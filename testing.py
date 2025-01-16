import os
import streamlit as st
from git import Repo
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_text_splitters import Language
import vertexai
from langchain_text_splitters import RecursiveCharacterTextSplitter
import google.auth 
from langchain_google_vertexai import VertexAI
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_chroma import Chroma
from urllib.parse import urlparse 
from langchain.chains import create_retrieval_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from htmlTemplates import css, bot_template, user_template
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

test_data = [
    {
        "question": "What is the purpose of this repository?",
        "expected_answer": "The purpose of this repository is to provide a tool for interacting with code repositories and PDF documents. It allows users to upload a GitHub repository or PDF files, extract and process the content (e.g., Python code or text), create vector embeddings using Vertex AI, and facilitate question-answering through a conversational interface. The application leverages Langchain for document processing, embeddings, and retrieval, enabling users to ask specific questions about the content of the repository or PDF."
    },
    {
        "question": "What does repoReader.py do?",
        "expected_answer": "The repoReader is a Streamlit application that allows users to upload a GitHub repository URL, extract Python code, and generate embeddings using Vertex AI. It splits the code into smaller chunks, stores them in a vector store, and enables users to interact with the repository through a conversational interface."
    },
    {
        "question": "How many programming languages are in this repository?",
        "expected_answer": "This repository uses only Python"
    },
    {
        "question": "Explain the functions inside the repoReader.py",
        "expected_answer": "The repoReader.py is a Streamlit application designed to read and interact with GitHub repositories. Here's an explanation of its key functions: generate_repo_path(url): Takes a GitHub repository URL, extracts the repository name, and generates a local path where the repository will be cloned. get_parsedDocuments(repo_url): Clones the GitHub repository to the generated path. Uses the GenericLoader to parse Python files (.py) and return documents. get_splittedContents(documents): Splits the parsed Python documents into smaller chunks using RecursiveCharacterTextSplitter to ensure the documents are manageable for embedding. get_vectorStores(texts): Uses VertexAI to initialize an embedding model, creates embeddings from the split document texts, and stores them in a FAISS vector store for fast retrieval. handle_userinput(user_question): Handles the user input question, invokes the conversation chain to retrieve the answer, and displays the conversation history in the app. get_conversation_chain(vector_store): Creates a retrieval-based conversational chain using the vector store, VertexAI for model-based responses, and memory to maintain the context of the conversation."
    }
]

# Load a pre-trained sentence transformer
model = SentenceTransformer('all-MiniLM-L6-v2')

def calculate_similarity(expected, generated):

    #Calculate cosine similarity between expected and generated answers.

    embeddings = model.encode([expected, generated])
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return similarity

def get_generatedResponse(question, conversation_chain):
    chat_history = conversation_chain.memory.chat_memory.messages

    for i in range(0, len(chat_history), 2):  # Iterate through user messages
        if chat_history[i].content.strip() == question:  # Match user question
            # Return the corresponding bot response
            return chat_history[i + 1].content.strip() if i + 1 < len(chat_history) else None
    return None

def evaluate_answers(conversation_chain, test_data, user_question):

        # Get the generated answer from the conversation chain
        generated_response = get_generatedResponse(user_question, conversation_chain)
        expected_response = None  # Default to None if no match is found


        matching_test_case = next((test for test in test_data if test["question"] == user_question), None)

        if matching_test_case:
            expected_response = matching_test_case["expected_answer"] 
        # Calculate similarity
        similarity_score = calculate_similarity(expected_response, generated_response)


        # Print the result to the terminal
        print("Question:", user_question)
        print("Expected Answer:", expected_response)
        print("Generated Answer:", generated_response)
        print("Similarity Score:", similarity_score)
        print("-" * 50)  # Separator for readability



def generate_repo_path(url, custom_path):
    
    #Generate the repository path based on the user's desired location and the repository name.

    repo_name = urlparse(url).path.split('/')[-1].replace(".git", "")
    return os.path.join(custom_path, repo_name)


def get_parsed_documents(repo_url, custom_path):
    
    #Clone the repository and parse its Python files.
    
    repo_path = generate_repo_path(repo_url, custom_path)
    repo = Repo.clone_from(repo_url, to_path=repo_path)

    loader = GenericLoader.from_filesystem(
        repo_path,
        glob="**/*",
        suffixes=[".py", ".txt", ".md"],
        exclude=["**/non-utf8-encoding.py"],
        parser=LanguageParser(language=Language.PYTHON, parser_threshold=500),
    )
    documents = loader.load()
    st.write(f"Number of Parsed Documents: {len(documents)}")
    return documents


def get_splitted_contents(documents):
    
    #Split parsed documents into smaller chunks.
    
    python_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON, chunk_size=2000, chunk_overlap=200
    )
    contents = python_splitter.split_documents(documents)
    st.write(f"Number of Split Texts: {len(contents)}")
    return contents


def get_vector_stores(texts):
    
    #Generate vector stores using VertexAI embeddings.
    
    PROJECT_ID = "coderefactoringai"  
    LOCATION = "us-central1"  
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    embedding = VertexAIEmbeddings(model_name="text-embedding-004")
    vector_store = FAISS.from_documents(texts, embedding)
    return vector_store


def handle_user_input(user_question):
    
    #Handle user input for questions and display the chat conversation.
    
    response = st.session_state.conversation.invoke({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)


def get_conversation_chain(vector_store):
    
    #Initialize the conversation chain with the retriever and memory.
    
    retriever = vector_store.as_retriever(
       search_type="mmr",  
        search_kwargs={"k": 8}, 
    )
    llm = VertexAI(
        project="coderefactoringai",
        location="us-central1",  
        model="gemini-1.5-flash-002",  
        model_kwargs={
            "temperature": 0.7,
            "max_length": 600,
            "top_p": 0.95,
            "top_k": 50
        }
    )

    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory
    )
    st.success("Reading Completed. You can now ask questions!")
    return conversation_chain


def main():
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.write("""
    <div style="display: flex; align-items: center;">
        <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" alt="GitHub Logo" width="40" style="margin-right: 10px;">
        <h1 style="display: inline;">Github Repository Reader</h1>
    </div>
    """, unsafe_allow_html=True)

    user_question = st.text_input("2. Ask about the repository you just provided.")
    if user_question:
         handle_user_input(user_question)
         if any(test["question"] == user_question for test in test_data):
            evaluate_answers(st.session_state.conversation, test_data,user_question)


    with st.sidebar:
        st.subheader("1. Upload URL First")
        repo_url = st.text_input("Enter the GitHub repository URL you want to read")
        custom_path = st.text_input("Enter the path to clone the repository (e.g., /path/to/your/folder):")

        if st.button("Submit"):
            if not repo_url or not custom_path:
                st.error("Both the repository URL and custom path are required.")
            else:
                with st.spinner("Reading...."):
                    parsed_documents = get_parsed_documents(repo_url, custom_path)
                    splitted_contents = get_splitted_contents(parsed_documents)
                    vector_store = get_vector_stores(splitted_contents)
                    
                    if vector_store is None:
                        st.error("Failed to create embeddings. Please check your input data.")
                    else:
                        st.session_state.conversation = get_conversation_chain(vector_store)



if __name__ == "__main__":
    main()

