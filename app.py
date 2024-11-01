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
from transformers import GPT2Tokenizer



#tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

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

def get_vectorStores(text_chuncks):
    # embeddings = OpenAIEmbeddings()
    embeddings = HuggingFaceEmbeddings(model_name="hkunlp/instructor-xl")
    #embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    #embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorStore = FAISS.from_texts(texts=text_chuncks, embedding=embeddings)
    return vectorStore

def get_conversation_chain(vector_store):
    #llm = ChatOpenAI()
    #llm = HuggingFaceHub(repo_id="google/flan-t5-small", model_kwargs={"temperature":0.5, "max_length":512})
    
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-base", 
        model_kwargs={ 
        "temperature": 0.6,
        "max_length": 100,
        "top_p": 0.9,
        "top_k": 50
        }
    )
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    
    """input_tokens = tokenizer.encode(user_question, return_tensors='pt').shape[1]
    max_new_tokens = max(0, 1024 - input_tokens)  # Calculate max_new_tokens based on input length"""

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
    load_dotenv()
    st.set_page_config(page_title="Pdf Reader", page_icon=":books:")

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
                
                text_chuncks = get_textChuncks(extracted_text)
                
                vector_store = get_vectorStores(text_chuncks)
                
                #retriever = RunnableLambda(vector_store.similarity_search).bind(k=1)  # select top result
                #st.write(retriever.batch(["GPA", "volunteer"]))

                st.session_state.conversation = get_conversation_chain(vector_store)



if __name__ == "__main__":  
    main()
