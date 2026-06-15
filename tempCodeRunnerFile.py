import os
import uuid
import base64
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from unstructured.partition.pdf import partition_pdf
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.messages import HumanMessage
from PIL import Image
from IPython.display import display, HTML

# Initialize Flask app
app = Flask(__name__)

# Load environment variables
load_dotenv()

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Configure upload folder and allowed extensions
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

# Global variables
global db  # Vectorstore database
db = None

# Function: Check if file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Function: Create documents for retriever
def create_document(text, text_summary, image_base64_list, image_summary, table, table_summary):
    documents = []

    # Add text documents
    for text, text_summary in zip(text, text_summary):
        id=str(uuid.uuid4())
        doc=Document(
            page_content=text_summary,
            metadata={
                "id":id,
                "type":"text",
                "original_content":text
            }
        )
        documents.append(doc)

    # Add image documents
    for image_base64,image_summary in zip(image_base64_list,image_summary):
        id=str(uuid.uuid4())
        doc=Document(
            page_content=image_summary,
            metadata={
                "id":id,
                "type":"image",
                "original_content":image_base64
            }
        )
        documents.append(doc)

    # Add table documents
    for table,table_summary in zip(table,table_summary):
        id=str(uuid.uuid4())
        doc=Document(
            page_content=table_summary,
            metadata={
                "id":id,
                "type":"text",
                "original_content":table
            }
        )
        documents.append(doc)

    return documents

# Route: Home page (upload form)
@app.route('/')
def upload_file():
    return render_template('upload.html')

# Route: Handle file upload
@app.route('/upload', methods=['POST'])
def handle_upload():
    global db

    # Validate uploaded file
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        # Save the uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Partition PDF
        raw_element = partition_pdf(
            filename=file_path,
            strategy="hi_res",
            extract_images_in_pdf=True,
            extract_image_block_types=["Image", "Table"],
            extract_image_block_to_payload=False,
            extract_image_block_output_dir="./raw_elements"
        )

        Text=[]
        Image=[]
        Table=[]

        for element in raw_element:
            if "unstructured.documents.elements.Text" in str(type(element)):
                Text.append(str(element))
            elif "unstructured.documents.elements.NarrativeText" in str(type(element)):
                Text.append(str(element))
            elif "unstructured.documents.elements.ListItem" in str(type(element)):
                Text.append(str(element))
            elif "unstructured.documents.elements.FigureCaption" in str(type(element)):
                Text.append(str(element))
            elif "unstructured.documents.elements.Table" in str(type(element)):
                Table.append(str(element))
            elif "unstructured.documents.elements.Image" in str(type(element)):
                Image.append(str(element))

        # Summarization using OpenAI
        model = ChatOpenAI(temperature=0, model="gpt-4")

        # Summarize text
        #creating prompt
        prompt_text="""You are an assistant tasked with summarizing text that has been extracted from a document in concise form. \n
                       The document's content has been embedded in a vector store, and the text for summarization is provided below: {element}"""
        text_prompt=ChatPromptTemplate.from_template(prompt_text)
        text_summarizing_chain = text_prompt | model | StrOutputParser()
        text_summary = [text_summarizing_chain.invoke({"element": t}) for t in Text]

        # Summarize tables
        prompt_text="""You are an assistant tasked with summarizing table content that has been extracted from a document in concise form. \n
              The document's content has been embedded in a vector store, and the table for summarization is provided below: {element}"""
        table_prompt=ChatPromptTemplate.from_template(prompt_text)
        table_summarizing_chain = table_prompt | model | StrOutputParser()
        table_summary = [table_summarizing_chain.invoke({"element": t}) for t in Table]

        # Process images (mock summary)
        #encode image to base64
        def encode_image(image_path):
            with open(image_path,"rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")

        #image summarising function
        def summarize_image(image_base64,prompt):

            #initiating gpt-4o for image summarisation
            model=ChatOpenAI(temperature=0, model="gpt-4o")

            image_summarising_chain=model.invoke(
                [
                HumanMessage(
                    content=[
                            {"type":"text","text":prompt},
                            {
                                "type":"image_url",
                                "image_url":{"url": f"data:image/jpg;base64,{image_base64}"}}
                            ]
                            )
                ]
            )
            return image_summarising_chain.content

        # Prompt
        prompt = """You are an assistant tasked with summarizing images for retrieval. \n
        These summaries will be embedded and used to retrieve the raw image. \n
        Give a concise summary of the image that is well optimized for retrieval."""
        
        image_summary={"path":[],"summary":[]}
        image_base64_list=[]
        for img_path in os.listdir("/content/raw_elements"):
            if img_path.endswith(".jpg"):
                image_base64=encode_image(os.path.join("/content/raw_elements",img_path))
                image_summary["path"].append(os.path.join("/content/raw_elements",img_path))
                image_summary["summary"].append(summarize_image(image_base64,prompt))
                image_base64_list.append(image_base64)
        
        # Create documents for retrieval
        documents = create_document(Text, text_summary, image_base64_list, image_summary, Table, table_summary)

        # Create vector database
        db = FAISS.from_documents(documents=documents, embedding=OpenAIEmbeddings())

        return jsonify({"message": "File uploaded and processed successfully. Ready for questions!"})

    return jsonify({"error": "Invalid file type"}), 400

# Route: Query processing
@app.route('/query', methods=['POST'])
def handle_query():
    global db
    if not db:
        return jsonify({"error": "No document has been uploaded yet."}), 400

    data = request.get_json()
    query = data.get("query", "")
    if not query:
        return jsonify({"error": "Query is required"}), 400

    try:
        #using gpt-4 as a llm model
        model=ChatOpenAI(temperature=0, model="gpt-4")

        #prompt
        prompt_text="""
        You are a AI assistant.
        Answer the question based only on the following context, which can include text, images and tables:
        {context}
        Question: {question}
        Don't answer if you are not sure and decline to answer and say "Sorry, I don't have much information about it."
        Just return the helpful answer in as much as detailed possible.
        Answer:
        """
        prompt=ChatPromptTemplate.from_template(prompt_text)
        
        #creating chain
        multimodal_rag_chain=prompt|model|StrOutputParser()

        def content_to_display(query):
            relevant_documents=db.similarity_search(query)
            relevant_images=[]
            context=""
            for doc in relevant_documents:
                if doc.metadata["type"]=="text":
                    context+=doc.metadata["original_content"]
                elif doc.metadata["type"]=="image":
                    context+=doc.page_content
                    relevant_images.append(doc.metadata["original_content"])
                elif doc.metadata["type"]=="table":
                    context+=doc.metadata["original_content"]


            result=multimodal_rag_chain.invoke({"context":context,"question":query})
            return result,relevant_images
        
        def answer(query):
            answer,relevant_images=content_to_display(query)

            #displaying Image
            if relevant_images:
                for image_base64 in relevant_images:
                    html_img = f'<img src="data:image/jpeg;base64,{image_base64}" alt="Base64 Image" style="width:300px;"/>'
            
            return answer,html_img        
        
        answer,html_img=answer(query)

        return jsonify({"answer": answer,
                        "html_image":html_img})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
