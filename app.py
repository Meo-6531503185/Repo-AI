import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
#from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from InstructorEmbedding import INSTRUCTOR
from langchain_huggingface import HuggingFaceEndpoint
from langchain.llms import HuggingFaceHub
from langchain.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css,bot_template,user_template
from langchain_core.runnables import RunnableLambda
import google.generativeai as genai
import requests
import os
import vertexai
import google.auth 
from google.auth.transport.requests import Request
from google.oauth2 import service_account
from langchain_google_vertexai import VertexAI
from langchain_google_vertexai import VertexAIEmbeddings
import google.generativeai as palm
from langchain.embeddings import GooglePalmEmbeddings
import google.auth
credentials, project_id = google.auth.default()
load_dotenv()

st.set_page_config(page_title="Pdf Reader", page_icon=":books:")

def extractText_Pdf(pdf_files):
    text=""
    for pdf in pdf_files:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_textChuncks(extracted_text):
    text_splitter = CharacterTextSplitter(
        separator= "\n",
        chunk_size = 150,
        chunk_overlap = 100,
        length_function = len,
    )
    chunks = text_splitter.split_text(extracted_text)
    return chunks

def get_vectorStores(text_chunks):
    #embeddings = OpenAIEmbeddings()
    #embeddings = HuggingFaceEmbeddings(model_name="hkunlp/instructor-xl")
    #embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    PROJECT_ID = "coderefactoringai"  
    LOCATION = "us-central1"  
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    embeddings = VertexAIEmbeddings(model_name="text-embedding-004")
    vectorStore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorStore

def get_conversation_chain(vector_store):
    #llm = ChatOpenAI()
    #llm = HuggingFaceHub(repo_id="google/flan-t5-small", model_kwargs={"temperature":0.5, "max_length":512})
    llm = VertexAI(
        project="coderefactoringai",
        location="us-central1",  # Common location for Vertex AI
        model="gemini-1.5-flash-002",  # Specify the Gemini model
        model_kwargs={
            "temperature": 0.7,
            "max_length": 600,
            "top_p": 0.95,
            "top_k": 50
        }
    )

    #llm = HuggingFaceHub(
    #   repo_id="google/flan-t5-base", 
    #  model_kwargs={ 
    #    "temperature": 0.6,
    #    "max_length": 100,
    #    "top_p": 0.9,
    #    "top_k": 50
    #    }
    #)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):

    response = st.session_state.conversation({'question': user_question})
    #st.write(response)
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():

    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    
    st.header("Uploads Your PDF & Just Ask :books:")  
    user_question = st.text_input("Ask a question about your Pdf:")
    if user_question:
        handle_userinput(user_question)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    

    with st.sidebar:
        st.subheader("1. Upload files")
        pdf_files = st.file_uploader("accept multiple PDFs", accept_multiple_files=True)
        if st.button("Upload"):
            with st.spinner("Reading"):
                extracted_text = extractText_Pdf(pdf_files)
                
                text_chunks = get_textChuncks(extracted_text)
                
                vector_store = get_vectorStores(text_chunks)
                
                if vector_store is None:
                    st.error("Failed to create embeddings. Please check your input data.")
                else:
                    st.session_state.conversation = get_conversation_chain(vector_store)


if __name__ == "__main__":  
    main()
