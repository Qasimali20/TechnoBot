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

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

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
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
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

    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain.invoke({"input_documents": docs, "question": user_question})
    st.write("Reply: ", response["output_text"])

def main():
    st.set_page_config(page_title="TechnoBot", page_icon="🤖", layout="wide")
    st.markdown(
        """
        <style>
        .main {
            background-color: #1a1a1a;
            color: white;
            font-family: 'Arial', sans-serif;
        }
        header {
            background-color: #8B0000;
            color: white;
            padding: 10px 0;
            text-align: center;
            font-size: 30px;
            font-weight: bold;
            border-radius: 8px;
        }
        .sidebar .sidebar-content {
            background-color: #000000;
            color: white;
        }
        .stButton button {
            background-color: #8B0000;
            color: white;
            border-radius: 8px;
        }
        .stTextInput>div>div>input {
            background-color: #333333;
            color: white;
        }
        </style>
        """, unsafe_allow_html=True
    )

    st.markdown("<header>Welcome to Techno Bot 🤖</header>", unsafe_allow_html=True)

    user_question = st.text_input("Ask a question:", label_visibility="collapsed", placeholder="Type your question here...")

    if user_question:
        user_input(user_question)

    if not os.path.exists("faiss_index"):
        with st.spinner("Processing..."):
            raw_text = get_pdf_text(PDF_DIRECTORY)
            text_chunks = get_text_chunks(raw_text)
            get_vector_store(text_chunks)
            st.success("Initial processing done. You can now ask questions.")

    with st.sidebar:
        st.image(r"pdfs/logo.jpg", use_column_width=True)
        st.title("TechnoBot Menu")
        st.markdown("""
            <p>Welcome to TechnoBot! This assistant can help you with information related to our robotics team.</p>
            <p>Please ask any questions you have about our team's domains, modules, competitions, and more!</p>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
