import os
import uuid
import base64
import shutil
import time
from flask import Flask, request, jsonify, render_template, session
from werkzeug.utils import secure_filename
from unstructured.partition.pdf import partition_pdf
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

app = Flask(__name__)
load_dotenv()

groq_key = os.getenv("GROQ_API_KEY")
if groq_key:
    os.environ["GROQ_API_KEY"] = groq_key
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

app.secret_key = os.getenv("SECRET_KEY", "docpilot-secret-key-2024")
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('./raw_elements', exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

# Per-user storage
user_dbs = {}
user_progress = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def get_user_id():
    if "user_id" not in session:
        session["user_id"] = str(uuid.uuid4())
    return session["user_id"]

def set_progress(user_id, step, message):
    user_progress[user_id] = {"step": step, "message": message}
    print(f"[{user_id[:8]}] {message}")

def is_scanned_pdf(file_path):
    try:
        elements = partition_pdf(filename=file_path, strategy="fast")
        text_elements = [e for e in elements if "Image" not in str(type(e))]
        return len(text_elements) < 3
    except:
        return True

def create_document(text, text_summary, image_base64_list, image_summary, table, table_summary):
    documents = []
    for t, ts in zip(text, text_summary):
        doc = Document(page_content=ts, metadata={"id": str(uuid.uuid4()), "type": "text", "original_content": t})
        documents.append(doc)
    for img, ims in zip(image_base64_list, image_summary):
        doc = Document(page_content=ims, metadata={"id": str(uuid.uuid4()), "type": "image", "original_content": img})
        documents.append(doc)
    for tb, tbs in zip(table, table_summary):
        doc = Document(page_content=tbs, metadata={"id": str(uuid.uuid4()), "type": "text", "original_content": tb})
        documents.append(doc)
    return documents

def cleanup_old_users():
    if len(user_dbs) > 50:
        oldest = list(user_dbs.keys())[0]
        del user_dbs[oldest]
        if oldest in user_progress:
            del user_progress[oldest]

@app.route('/')
def upload_file():
    get_user_id()
    return render_template('upload.html')

@app.route('/progress')
def get_progress():
    user_id = get_user_id()
    return jsonify(user_progress.get(user_id, {"step": 0, "message": ""}))

@app.route('/upload', methods=['POST'])
def handle_upload():
    user_id = get_user_id()

    # Prevent double upload
    if user_progress.get(user_id, {}).get("step", 0) in [1, 2, 3, 4]:
        return jsonify({"error": "A file is already being processed. Please wait."}), 400

    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        set_progress(user_id, 1, "Uploading file...")

        # Per-user folders
        user_upload_dir = f"./uploads/{user_id}"
        user_raw_dir = f"./raw_elements/{user_id}"
        os.makedirs(user_upload_dir, exist_ok=True)
        if os.path.exists(user_raw_dir):
            shutil.rmtree(user_raw_dir)
        os.makedirs(user_raw_dir, exist_ok=True)

        filename = secure_filename(file.filename)
        file_path = os.path.join(user_upload_dir, filename)
        file.save(file_path)

        set_progress(user_id, 2, "Extracting content from PDF...")
        scanned = is_scanned_pdf(file_path)
        strategy = "ocr_only" if scanned else "fast"

        raw_element = partition_pdf(
            filename=file_path,
            strategy=strategy,
            extract_images_in_pdf=True,
            extract_image_block_types=["Image", "Table"],
            extract_image_block_to_payload=False,
            extract_image_block_output_dir=user_raw_dir
        )

        Text, Images, Table = [], [], []
        for element in raw_element:
            t = str(type(element))
            if "Text" in t or "NarrativeText" in t or "ListItem" in t or "FigureCaption" in t or "Title" in t:
                Text.append(str(element))
            elif "Table" in t:
                Table.append(str(element))
            elif "Image" in t:
                Images.append(str(element))

        # Handle empty PDF
        if not Text and not Table and not Images:
            set_progress(user_id, 0, "")
            return jsonify({"error": "No content could be extracted. PDF may be empty or password protected."}), 400

        set_progress(user_id, 3, "Summarizing content with AI...")
        model = ChatGroq(temperature=0, model="llama-3.3-70b-versatile")

        text_chain = ChatPromptTemplate.from_template("Summarize this text concisely: {element}") | model | StrOutputParser()
        if Text:
            combined_text = "\n\n".join(Text)
            try:
                text_summary = [text_chain.invoke({"element": combined_text})]
            except Exception:
                time.sleep(60)
                text_summary = [text_chain.invoke({"element": combined_text})]
            Text = [combined_text]
        else:
            text_summary = []

        table_chain = ChatPromptTemplate.from_template("Summarize this table concisely: {element}") | model | StrOutputParser()
        if Table:
            combined_table = "\n\n".join(Table)
            try:
                table_summary = [table_chain.invoke({"element": combined_table})]
            except Exception:
                time.sleep(60)
                table_summary = [table_chain.invoke({"element": combined_table})]
            Table = [combined_table]
        else:
            table_summary = []

        image_base64_list = []
        image_summaries = []
        for img_path in os.listdir(user_raw_dir):
            if img_path.endswith(".jpg"):
                with open(os.path.join(user_raw_dir, img_path), "rb") as f:
                    image_base64_list.append(base64.b64encode(f.read()).decode("utf-8"))
                image_summaries.append("Image extracted from document.")

        set_progress(user_id, 4, "Building vector store...")
        document = create_document(Text, text_summary, image_base64_list, image_summaries, Table, table_summary)
        user_dbs[user_id] = FAISS.from_documents(
            documents=document,
            embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        )

        # Cleanup files and old users
        shutil.rmtree(user_upload_dir, ignore_errors=True)
        shutil.rmtree(user_raw_dir, ignore_errors=True)
        cleanup_old_users()

        set_progress(user_id, 5, "Done!")
        return jsonify({"message": f"File processed successfully using {'OCR' if scanned else 'fast'} extraction. Ready for questions!"})

    return jsonify({"error": "Invalid file type"}), 400

@app.route('/query', methods=['POST'])
def handle_query():
    user_id = get_user_id()
    db = user_dbs.get(user_id)

    if not db:
        return jsonify({"error": "No document has been uploaded yet."}), 400

    data = request.get_json()
    query = data.get("query", "")
    if not query:
        return jsonify({"error": "Query is required"}), 400

    try:
        model = ChatGroq(temperature=0, model="llama-3.3-70b-versatile")
        prompt_text = """
        You are an AI assistant.
        Answer the question based only on the following context:
        {context}
        Question: {question}
        If unsure, say "Sorry, I don't have much information about it."
        Answer:
        """
        prompt = ChatPromptTemplate.from_template(prompt_text)
        chain = prompt | model | StrOutputParser()

        relevant_documents = db.similarity_search(query)
        context = ""
        relevant_images = []
        for doc in relevant_documents:
            if doc.metadata["type"] == "text":
                context += doc.metadata["original_content"]
            elif doc.metadata["type"] == "image":
                context += doc.page_content
                relevant_images.append(doc.metadata["original_content"])

        answer = chain.invoke({"context": context, "question": query})

        html_image = ""
        if relevant_images:
            for image_base64 in relevant_images:
                html_image = f'<img src="data:image/jpeg;base64,{image_base64}" alt="Image" style="width:300px;"/>'

        return jsonify({"answer": answer, "html_image": html_image})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
