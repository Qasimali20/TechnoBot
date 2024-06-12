import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY="AIzaSyAlop3bs-K4EES9L06HLCIrvnXIJtzcLDY"
genai.configure(api_key=GOOGLE_API_KEY)

# HTML and CSS for custom styling
html_header = """
    <style>
        body {
            background-color: #1a1a1a;
            color: #ffffff;
            font-family: 'Arial', sans-serif;
        }
        .header {
            background-color: #8B0000;
            color: #ffffff;
            padding: 15px;
            text-align: center;
            font-size: 36px;
            font-weight: bold;
            border-radius: 10px;
            margin-bottom: 30px;
        }
        .container {
            display: flex;
            flex-direction: row;
            justify-content: space-between;
            align-items: flex-start;
            padding: 20px;
        }
        .main-content {
            width: 70%;
            background-color: #333333;
            padding: 20px;
            border-radius: 10px;
        }
        .sidebar {
            width: 25%;
            background-color: #2a2a2a;
            padding: 20px;
            border-radius: 10px;
        }
        .sidebar img {
            width: 100%;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .question-input {
            background-color: #4d4d4d;
            color: #ffffff;
            border-radius: 10px;
            padding: 10px;
            width: 100%;
            margin-bottom: 20px;
        }
        .reply-container {
            margin-top: 20px;
            padding: 10px;
            background-color: #4d4d4d;
            border-radius: 10px;
        }
    </style>
"""
# Define the directory containing your PDFs
PDF_DIRECTORY = "pdfs"

def get_pdf_text(pdf_directory):
    text = ""
    for filename in os.listdir(pdf_directory):
        if filename.endswith(".pdf"):
            filepath = os.path.join(pdf_directory, filename)
            pdf_reader = PdfReader(filepath)
            for page in pdf_reader.pages:
                text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def user_input(user_question):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        chain = get_conversational_chain()
        response = chain.invoke({"input_documents": docs, "question": user_question})
        st.write("Reply: ", response["output_text"])
    except Exception as e:
        st.error(f"An error occurred: {e}")

def main():
    # Render HTML header
    st.markdown(html_header, unsafe_allow_html=True)
    
    # Render header
    st.markdown("<div class='header'>Welcome to TechnoBot 🤖</div>", unsafe_allow_html=True)
    
    # Render main content and sidebar
    col1, col2 = st.columns([3, 1])
    with col1:
        # Main content
        st.markdown("<div class='main-content'>", unsafe_allow_html=True)
        user_question = st.text_input("Ask a question:", label_visibility="collapsed", placeholder="Type your question here...", key="question_input")
        if user_question:
            user_input(user_question)
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        # Sidebar
        st.markdown("<div class='sidebar'>", unsafe_allow_html=True)
        st.image("pdfs/logo.jpg", use_column_width=True)
        st.title("TechnoBot Menu")
        st.markdown("""
            <p>Welcome to TechnoBot! This assistant can help you with information related to our robotics team.</p>
            <p>Please ask any questions you have about our team's domains, modules, competitions, and more!</p>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Check if faiss_index exists
    if not os.path.exists("faiss_index"):
        with st.spinner("Processing..."):
            raw_text = get_pdf_text(PDF_DIRECTORY)
            text_chunks = get_text_chunks(raw_text)
            get_vector_store(text_chunks)
            st.success("Initial processing done. You can now ask questions.")

if __name__ == "__main__":
    main()
