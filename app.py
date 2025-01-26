from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
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
import black
import autopep8

load_dotenv()
github_token = os.getenv("GITHUB_TOKEN")
if github_token:
    print("GitHub Token loaded successfully")
else:
    print("GitHub Token not found")


#Modified 0.2
#Untouch
def fetch_file_content(file_url):
    """Fetch the content of a single file."""
    file_response = requests.get(file_url)
    if file_response.status_code == 200:
        return file_response.text
    else:
        st.warning(f"Failed to retrieve file: {file_url}")
        return ""
    
#Untouch
def fetch_repo_contents(api_url, headers):
    """Fetch the contents of the entire repository."""
    response = requests.get(api_url, headers=headers)
    
    if response.status_code != 200:
        st.error(f"Failed to retrieve files from GitHub. Status code: {response.status_code}")
        return []

    return response.json()
    
#Untouch
def fetch_all_files_from_repo(api_url, headers):
    """Fetch all files from a GitHub repository, including files inside subfolders."""
    all_files = []  # This will hold the paths of all files in the repository
    repo_contents = fetch_repo_contents(api_url, headers)

    for file_info in repo_contents:
        if file_info["type"] == "file":
            # For files, add them to the list
            all_files.append({
                "name": file_info["name"],
                "url": file_info["download_url"],
                "path": file_info["path"]
            })
        elif file_info["type"] == "dir":
            # If it's a directory, recurse into it
            subfolder_url = file_info["url"]
            all_files += fetch_all_files_from_repo(subfolder_url, headers)

    return all_files


#Untouch
def read_all_repo_files(github_url, github_token):
    """Read all files in a repository and store them for later queries."""
    parts = github_url.rstrip("/").split("/")
    owner, repo = parts[-2], parts[-1]
    api_url = f"https://api.github.com/repos/{owner}/{repo}/contents"
    headers = {"Authorization": f"token {github_token}"}

    all_files = fetch_all_files_from_repo(api_url, headers)

    repo_data = {}
    all_file_chunks = []
    for file in all_files:
        file_content = fetch_file_content(file["url"])
        #Touch ( changed from get_text_chunks to get_text_chunks_for_refactoring)
        # chunks = get_text_chunks_for_refactoring(file_content)

        chunks = get_text_chunks(file_content)
        all_file_chunks.extend(chunks)
        repo_data[file["path"]] = file_content
    
    # Generate the vector store from the chunks
    vector_store = get_vector_store(all_file_chunks)
    return repo_data, vector_store

#Touch
# def get_text_chunks_for_refactoring(file_content):
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=3000,  # Larger chunks for code refactoring
#         chunk_overlap=200,
#         separators=["\n\n", "\n", " "],
#     )
#     return text_splitter.split_text(file_content)



#Original
def get_text_chunks(extracted_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1500,
        chunk_overlap=100,
        length_function=len,
    )
    chunks = text_splitter.split_text(extracted_text)
    return chunks


#Untouch
def get_vector_store(text_chunks):
    PROJECT_ID = "coderefactoringai"
    LOCATION = "us-central1"
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    embeddings = VertexAIEmbeddings(model_name="text-embedding-004")
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vector_store

def get_conversation_chain(vector_store):

    retriever = vector_store.as_retriever(
        search_type = "mmr",
        search_kwargs = {"k":8},
    )
    llm = VertexAI(
        project="coderefactoringai",
        location="us-central1",
        model="gemini-1.5-flash-002",
        model_kwargs={
            "temperature": 0.7,
            "max_length": 600,
            "top_p": 0.95,
            "top_k": 50,
        },
    )
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
    )
    return conversation_chain


#New fun for code 
# Utility function to split explanation and code
def split_explanation_and_code(response_content):
    """
    Split the bot's response into explanation and code parts.
    Assumes the response has a section for explanation and another for code.
    """
    parts = response_content.split("```")
    if len(parts) > 1:
        explanation = parts[0].strip()  # Text before the code block
        code_snippet = parts[1].strip()  # Code block content
    else:
        explanation = response_content
        code_snippet = ""

    return explanation, code_snippet



#Orginal handle_user_input
# def handle_user_input(user_question):
#     if "conversation" in st.session_state:
#         conversation_chain = st.session_state.conversation
#         response = conversation_chain({"question": user_question})
#         st.session_state.chat_history = response['chat_history']

#         for i, message in enumerate(st.session_state.chat_history):
#             if i % 2 == 0:
#                 st.write(f"User: {message.content}")
#             else:
#                 st.write(f"Bot: {message.content}")
#     else:
#         st.error("Conversation chain is not initialized. Please provide a repository URL first.")

#New funs

# Function to explain code in detail
def explain_code_detailed(file_content):
    """
    Use AI to explain the given code in detail.
    """
    st.subheader("Detailed Explanation of the Code:")
    explanation_query = f"Explain in detail what the following code does, line by line:\n{file_content}"
    response = st.session_state.conversation({"question": explanation_query})
    st.write(response["answer"])

# Function to refactor code
def refactor_code(file_content):
    """
    Use AI to refactor the given code.
    """
    st.subheader("Refactored Code:")
    refactor_query = f"Refactor the following code and fix any bugs if present:\n{file_content}"
    response = st.session_state.conversation({"question": refactor_query})
    explanation, code_snippet = split_explanation_and_code(response["answer"])
    
    # Display explanation and refactored code
    st.write("### Explanation")
    st.write(explanation)
    if code_snippet:
        st.write("### Refactored Code")
        st.code(code_snippet, language='python')

# Function to suggest features
def suggest_features(file_content):
    """
    Use AI to suggest features to add and provide modified code suggestions.
    """
    st.subheader("Suggested Features and Code Modifications:")
    suggestion_query = f"Suggest features to add to the following code and provide modified code:\n{file_content}"
    response = st.session_state.conversation({"question": suggestion_query})
    explanation, code_snippet = split_explanation_and_code(response["answer"])

    # Display explanation and suggested code
    st.write("### Explanation")
    st.write(explanation)
    if code_snippet:
        st.write("### Modified Code")
        st.code(code_snippet, language='python')

#New fun for code
def handle_user_input(user_question):
    if "conversation" in st.session_state:
        conversation_chain = st.session_state.conversation

        # Check if the question is about a specific file
        import re
        match = re.search(r'["\']?([\w\-/]+(\.\w+))["\']?', user_question)
        file_name = match.group(1) if match else None

        if file_name:
            # Ensure repo_data is loaded
            if not st.session_state.repo_data:
                st.error("Repository data is empty. Please ensure the repository was loaded correctly.")
                return

            # Search for the file in the repository data
            matching_files = [
                key for key in st.session_state.repo_data.keys() if file_name in key
            ]
            if matching_files:
                # Get the first matching file content
                file_content = st.session_state.repo_data[matching_files[0]]

                # 1. Display the source code first
                st.subheader(f"Source Code in '{file_name}':")
                st.code(file_content, language='python')  # Adjust language based on file type


                #Original
                # 2. Pass the file content to the AI for a detailed explanation
                # st.subheader("Detailed Explanation of the Code:")
                # explanation_query = f"Explain in detail what the following code does:\n{file_content}"
                # response = conversation_chain({"question": explanation_query})

                # # Display the explanation from the AI
                # st.write(response["answer"])

                #New 
                 # Handle user request types
                if "explain" in user_question.lower():
                    explain_code_detailed(file_content)
                elif "refactor" in user_question.lower():
                    refactor_code(file_content)
                elif "suggest" in user_question.lower():
                    suggest_features(file_content)
                else:
                    st.error("Unsupported operation. Please use 'explain', 'refactor', or 'suggest' in your question.")

            else:
                st.error(f"File '{file_name}' not found in the repository.")
        else:
             # Handle non-file-specific questions as usual
            response = conversation_chain({"question": user_question})
            st.session_state.chat_history = response['chat_history']

            for i, message in enumerate(st.session_state.chat_history):
                if i % 2 == 0:
                    st.write(f"User: {message.content}")
                else:
                    st.write(f"Bot: {message.content}")
    else:
        st.error("Conversation chain is not initialized. Please provide a repository URL first.")



# def handle_user_input(user_question):
#     if "conversation" in st.session_state:
#         conversation_chain = st.session_state.conversation

#         # Check if the question is about a specific file
#         if any(ext in user_question.lower() for ext in [".py", ".js", ".json", ".md", ".html"]):

#             # Extract the file name from the question (optional: use regex for better parsing)
#             file_name = user_question.split("file")[-1].strip().strip('"').strip("'")
            
#             # Search for the file in the repository data
#             if file_name in st.session_state.repo_data:
#                 file_content = st.session_state.repo_data[file_name]

#                 # 1. Display the source code first
#                 st.subheader(f"Source Code in '{file_name}':")
#                 st.code(file_content, language='python')  # Adjust language based on file type

#                 # 2. Pass the file content to the AI for a detailed explanation
#                 st.subheader("Detailed Explanation of the Code:")
#                 explanation_query = f"Explain in detail what the following code does:\n{file_content}"
#                 response = conversation_chain({"question": explanation_query})

#                 # Display the explanation from the AI
#                 st.write(response["answer"])
#             else:
#                 st.error(f"File '{file_name}' not found in the repository.")
#         else:
#             # Handle non-file-specific questions as usual
#             response = conversation_chain({"question": user_question})
#             st.session_state.chat_history = response['chat_history']

#             for i, message in enumerate(st.session_state.chat_history):
#                 if i % 2 == 0:
#                     st.write(f"User: {message.content}")
#                 else:
#                     st.write(f"Bot: {message.content}")
#     else:
#         st.error("Conversation chain is not initialized. Please provide a repository URL first.")

# Function to display the chat history with a background color
def display_chat_history():
    st.markdown(
        """
        <style>
        .chat-container {
            background-color: #f0f4f7;
            padding: 10px;
            border-radius: 8px;
            margin-bottom: 20px;
            max-height: 300px;
            overflow-y: auto;
        }
        .user-message {
            color: #333333;
            font-weight: bold;
            margin-bottom: 8px;
        }
        .bot-message {
            color: #0056b3;
            margin-bottom: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.markdown(f'<div class="user-message"> User: {message.content}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="bot-message"> Bot: {message.content}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

#Modified 0.2
# Main Streamlit application function
def main():
    st.set_page_config(page_title="GitHub Repositories Reader")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        #Newly added one for code
    if "repo_data" not in st.session_state:  # Add repo_data initialization
        st.session_state.repo_data = {}

    st.header("Provide GitHub Repository URL & Ask :books:")

    # Display chat history if available
    if st.session_state.chat_history:
        display_chat_history()


    github_url = st.text_input("Enter GitHub repository URL:")
    user_question = st.text_input("Ask a question about your GitHub repository:")

    if github_url:
        with st.spinner("Fetching repository contents..."):
            repo_data, vector_store = read_all_repo_files(github_url, github_token)
    


            #Original

            # if repo_data:
            #     # Initialize the conversation chain with the vector store
            #     st.session_state.conversation = get_conversation_chain(vector_store)

            #     # Handle user query related to the repository
            #     if user_question:
            #         handle_user_input(user_question)
            # else:
            #     st.error("Failed to fetch content from GitHub.")

            if repo_data:
                st.session_state.repo_data = repo_data
                # Initialize the conversation chain with the vector store
                if st.session_state.conversation is None:
                    st.session_state.conversation = get_conversation_chain(vector_store)

                # Handle user query related to the repository
                if user_question:
                    handle_user_input(user_question)
                    # st.rerun  # To clear the input box and refresh the UI
            else:
                st.error("Failed to fetch content from GitHub.")

if __name__ == "__main__":
    main()





#Modified 0.1

# def fetch_file_content(file_url):
#     """Fetch the content of a single file."""
#     file_response = requests.get(file_url)
#     if file_response.status_code == 200:
#         return file_response.text
#     else:
#         st.warning(f"Failed to retrieve file: {file_url}")
#         return ""

# def fetch_repo_contents(api_url, headers):
#     """Fetch the contents of the entire repository."""
#     response = requests.get(api_url, headers=headers)
    
#     if response.status_code != 200:
#         st.error(f"Failed to retrieve files from GitHub. Status code: {response.status_code}")
#         return []

#     return response.json()

# def fetch_all_files_from_repo(api_url, headers):
#     """Fetch all files from a GitHub repository, including files inside subfolders."""
#     all_files = []  # This will hold the paths of all files in the repository
#     repo_contents = fetch_repo_contents(api_url, headers)

#     for file_info in repo_contents:
#         if file_info["type"] == "file":
#             # For files, add them to the list
#             all_files.append({
#                 "name": file_info["name"],
#                 "url": file_info["download_url"],
#                 "path": file_info["path"]
#             })
#         elif file_info["type"] == "dir":
#             # If it's a directory, recurse into it
#             subfolder_url = file_info["url"]
#             all_files += fetch_all_files_from_repo(subfolder_url, headers)

#     return all_files

# def read_all_repo_files(github_url, github_token):
#     """Read all files in a repository and store them for later queries."""
#     parts = github_url.rstrip("/").split("/")
#     owner, repo = parts[-2], parts[-1]
#     api_url = f"https://api.github.com/repos/{owner}/{repo}/contents"
#     headers = {"Authorization": f"token {github_token}"}

#     # Fetch all files
#     all_files = fetch_all_files_from_repo(api_url, headers)

#     # Store the files' content
#     repo_data = {}
#     for file in all_files:
#         file_content = fetch_file_content(file["url"])
#         repo_data[file["path"]] = file_content
    
#     return repo_data

# def query_repo_content(repo_data, query):
#     """Query the content of the repository."""
#     query_lower = query.lower()
    
#     # If query is about a file in the repository
#     results = []
#     for file_path, content in repo_data.items():
#         if query_lower in file_path.lower():
#             results.append(f"File: {file_path}\n\n{content}")

#     if results:
#         return "\n\n".join(results)
#     else:
#         return "No file matching the query found."

# def fetch_github_files(github_url, github_token, file_name=None):
#     """Fetch and search for files in a GitHub repository."""
#     parts = github_url.rstrip("/").split("/")
#     owner, repo = parts[-2], parts[-1]
#     api_url = f"https://api.github.com/repos/{owner}/{repo}/contents"
#     headers = {"Authorization": f"token {github_token}"}
    
#     # Read all files and store their content
#     repo_data = read_all_repo_files(github_url, github_token)

#     # If a file_name is provided, query for it
#     if file_name:
#         return query_repo_content(repo_data, file_name)
#     else:
#         return "No file name provided for search."




#Modified 0.0
# def fetch_file_content(file_url):
#     """Fetch the content of a single file."""
#     file_response = requests.get(file_url)
#     if file_response.status_code == 200:
#         return file_response.text
#     else:
#         st.warning(f"Failed to retrieve file: {file_url}")
#         return ""
    
# def search_for_file_in_repo(file_name, api_url, headers):
#     """Search for a specific file (e.g., db.py) in the entire repository."""
#     response = requests.get(api_url, headers=headers)
    
#     if response.status_code != 200:
#         st.error(f"Failed to retrieve files from GitHub. Status code: {response.status_code}")
#         return ""
    
#     repo_contents = response.json()
    
#     for file_info in repo_contents:
#         if file_info["type"] == "file" and file_info["name"] == file_name:
#             # If the file is found, fetch its content
#             file_url = file_info["download_url"]
#             return fetch_file_content(file_url)
        
#         elif file_info["type"] == "dir":
#             # If it's a directory, recurse into it
#             subfolder_url = file_info["url"]
#             file_content = search_for_file_in_repo(file_name, subfolder_url, headers)
#             if file_content:
#                 return file_content
    
#     return f"{file_name} not found in the repository."

# def fetch_github_files_recursive(api_url, headers, search_file_name=None):
#     """Fetch all files recursively from a GitHub repository, optionally searching for a specific file."""
#     response = requests.get(api_url, headers=headers)
    
#     if response.status_code != 200:
#         st.error(f"Failed to retrieve files from GitHub. Status code: {response.status_code}")
#         return ""

#     repo_contents = response.json()
#     all_text = ""

#     for file_info in repo_contents:
#         if file_info["type"] == "file":
#             # If searching for a specific file, check if this is the file
#             if search_file_name:
#                 if file_info["name"] == search_file_name:
#                     file_url = file_info["download_url"]
#                     all_text += fetch_file_content(file_url) + "\n\n"
#             else:
#                 # If not searching for a specific file, fetch all files' content
#                 file_url = file_info["download_url"]
#                 all_text += fetch_file_content(file_url) + "\n\n"

#         elif file_info["type"] == "dir":
#             # If it's a directory, recurse into it
#             subfolder_url = file_info["url"]
#             all_text += fetch_github_files_recursive(subfolder_url, headers, search_file_name)
    
#     return all_text

# def fetch_github_files(github_url, file_name=None):
#     parts = github_url.rstrip("/").split("/")
#     owner, repo = parts[-2], parts[-1]
#     api_url = f"https://api.github.com/repos/{owner}/{repo}/contents"
#     headers = {"Authorization": f"token {github_token}"}
    
#     if file_name:
#         return search_for_file_in_repo(file_name, api_url, headers)
#     else:
#         return fetch_github_files_recursive(api_url, headers)













#Original Version
# def fetch_github_files_recursive(api_url, headers):
#     response = requests.get(api_url, headers=headers)
#     if response.status_code != 200:
#         st.error(f"Failed to retrieve file from GitHub. Status code: {response.status_code}")
#         st.error(f"Error details: {response.text}")
#         return ""

#     repo_contents = response.json()
#     all_text = ""

#     for file_info in repo_contents:
#         if file_info["type"] == "file":
#             file_url = file_info["download_url"]
#             file_response = requests.get(file_url)

#             if file_response.status_code == 200:
#                 all_text += file_response.text + "\n\n"
#             else:
#                 st.warning(f"Failed to retrieve file: {file_info['name']}")

#         elif file_info["type"] == "dir":
#             subfolder_url = file_info["url"]
#             all_text += fetch_github_files_recursive(subfolder_url, headers)

#     return all_text


# def fetch_github_files(github_url):
#     parts = github_url.rstrip("/").split("/")
#     owner, repo = parts[-2], parts[-1]
#     api_url = f"https://api.github.com/repos/{owner}/{repo}/contents"
#     headers = {"Authorization": f"token {github_token}"}
#     return fetch_github_files_recursive(api_url, headers)

#Original
# def main():
#     # st.write("<style>body{background-color:#f5f5f5;}</style>", unsafe_allow_html=True)
#     st.set_page_config(page_title="GitHub Repositories Reader")

#     if "conversation" not in st.session_state:
#         st.session_state.conversation = None
#     if "chat_history" not in st.session_state:
#         st.session_state.chat_history = []

#     st.header("Provide GitHub Repository URL & Ask :books:")
#     github_url = st.text_input("Enter GitHub repository URL:")
#     user_question = st.text_input("Ask a question about your GitHub repository:")

#     if github_url:
#         with st.spinner("Fetching repository contents..."):
#             #Modified 0.1
#             # extracted_text = fetch_github_files(github_url)
#             extracted_text = fetch_github_files(github_url, github_token)

#             if extracted_text:
#                 text_chunks = get_text_chunks(extracted_text)
#                 vector_store = get_vector_store(text_chunks)
#                 if vector_store:
#                     st.session_state.conversation = get_conversation_chain(vector_store)
#                 else:
#                     st.error("Failed to create embeddings. Please check your repository content.")
#             else:
#                 st.error("Failed to fetch content from GitHub.")

#     if user_question:
#         handle_user_input(user_question)

