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

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# Define constants
PDF_DIRECTORY = "pdfs"

# Function to extract text from PDFs
def get_pdf_text(pdf_directory):
    text = ""
    for filename in os.listdir(pdf_directory):
        if filename.endswith(".pdf"):
            filepath = os.path.join(pdf_directory, filename)
            pdf_reader = PdfReader(filepath)
            for page in pdf_reader.pages:
                text += page.extract_text()
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

# Function to create and save a vector store from text chunks
def create_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Function to load QA chain
def load_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in the provided context, just say, "answer is not available in the context". Don't provide a wrong answer.\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# Function to process user input and generate a response
def process_user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = vector_store.similarity_search(user_question)
    chain = load_conversational_chain()
    response = chain.invoke({"input_documents": docs, "question": user_question})
    return response["output_text"]

# Main function to run the Streamlit app
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
            background-color: #6B0000; /* Darker shade of red for header */
            color: white;
            padding: 10px 0;
            text-align: center;
            font-size: 30px;
            font-weight: bold;
            border-radius: 8px;
        }
        .sidebar .sidebar-content {
            background-color: #333333; /* Changed sidebar background color to dark grey */
            color: white;
        }
        .stButton button {
            background-color: #8B0000; /* Darker shade of red */
            color: white;
            border-radius: 8px;
        }
        .stTextInput>div>div>input {
            background-color: #333333;
            color: white;
        }
        .chat-message {
            border-radius: 8px;
            padding: 10px;
            margin: 10px 0;
        }
        .user-message {
            background-color: #6B0000; /* Darker shade of red for user messages */
            color: white;
            text-align: left;
        }
        .bot-message {
            background-color: #333333;
            color: white;
        }
        </style>
        """, unsafe_allow_html=True
    )

    st.markdown("<header>Welcome to Techno Bot 🤖</header>", unsafe_allow_html=True)

    # Initialize session state for chat history
    if 'history' not in st.session_state:
        st.session_state.history = []

    user_question = st.text_input("Ask a question:", label_visibility="collapsed", placeholder="Type your question here...")

    if user_question:
        response = process_user_input(user_question)
        st.session_state.history.append({"question": user_question, "answer": response})

    if not os.path.exists("faiss_index"):
        with st.spinner("Processing..."):
            raw_text = get_pdf_text(PDF_DIRECTORY)
            text_chunks = get_text_chunks(raw_text)
            create_vector_store(text_chunks)
            st.success("Initial processing done. You can now ask questions.")

    # Display chat history
    for chat in st.session_state.history:
        with st.container():
            st.markdown(f"<div class='chat-message user-message'><strong>You:</strong> {chat['question']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='chat-message bot-message'><strong>TechnoBot:</strong> {chat['answer']}</div>", unsafe_allow_html=True)

    with st.sidebar:
        st.image(r"pdfs/logo.jpg", use_column_width=True)
        st.title("TechnoBot Menu")
        st.markdown("""
            <p>Welcome to TechnoBot! This assistant can help you with information related to our robotics team.</p>
            <p>Please ask any questions you have about our team's domains, modules, competitions, and more!</p>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
