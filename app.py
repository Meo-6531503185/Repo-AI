from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.llms import HuggingFaceHub
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css,bot_template,user_template
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

# from langchain_core.runnables import RunnableLambda
# import google.generativeai as genai
# from langchain_community.chat_models import ChatOpenAI
# from langchain.embeddings import HuggingFaceEmbeddings
# from sentence_transformers import SentenceTransformer
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from InstructorEmbedding import INSTRUCTOR
# from langchain_huggingface import HuggingFaceEndpoint
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.embeddings import HuggingFaceInstructEmbeddings
# from PyPDF2 import PdfReader

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
    # PROJECT_ID = "coderefactoringai"
    PROJECT_ID = "repoai-440607"
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
        # project="coderefactoringai",
        project = "repoai-440607",
        location="us-central1",
        # model="gemini-1.5-flash-002",
        model = "gemini-1.5-pro",
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
# def split_explanation_and_code(response_content):
#     """
#     Split the bot's response into explanation and code parts.
#     Assumes the response has a section for explanation and another for code.
#     """
#     parts = response_content.split("```")
#     if len(parts) > 1:
#         explanation = parts[0].strip()  # Text before the code block
#         code_snippet = parts[1].strip()  # Code block content
#     else:
#         explanation = response_content
#         code_snippet = ""

#     return explanation, code_snippet

import re

def split_explanation_and_code(response_content):
    """
    Splits the AI response into explanation and code.
    Works with responses that use triple backticks for code blocks.
    """

    # Match everything before and inside a code block
    match = re.search(r"```(?:python)?\n(.*?)```", response_content, re.DOTALL)

    if match:
        code_snippet = match.group(1).strip()  # Extracted code
        explanation = response_content.replace(match.group(0), "").strip()  # Remove the code block
    else:
        explanation = response_content  # If no code block, return entire response as explanation
        code_snippet = ""

    return explanation, code_snippet


#New funs
# Function to explain code in detail
def explain_code_detailed(file_content, user_question):
    """
    Use AI to explain the given code in detail.
    """
    st.subheader("Detailed Explanation of the Code:")
    if "line by line" in user_question.lower() or "each line" in user_question.lower():
        explain_type = "line_by_line"
    elif "block by block" in user_question.lower() or "each block" in user_question.lower():
        explain_type = "block_by_block"
    else:
        explain_type = "general"

    # Generate the appropriate prompt
    if explain_type == "line_by_line":
        explanation_query = f"""
Provide a detailed, line-by-line explanation of the following code. For each line, include:
- What the line does.
- Why it is needed in the code.
- Any potential side effects or important considerations related to that line.

Please format your response as follows:
1. Line of code: Explanation
2. Line of code: Explanation

Here is the code:
{file_content}
"""
    elif explain_type == "block_by_block":
        explanation_query = f"""
Analyze the following code and explain it **block by block**.  
For each block:
- Extract a meaningful snippet of code (group related lines together).
- Explain what the block does.
- Describe why it is necessary.
- Highlight any important considerations or potential optimizations.

Format your response as:
### Code Block:
```python
<code_snippet>
"""
        
    else:
        explanation_query = f"""
Provide a high-level summary of the following code. Focus on:
- The overall purpose of the code.
- How the code is structured.
- The key functionalities it provides.
- Any important design choices.

Here is the code:
{file_content}
"""

    response = st.session_state.conversation({"question": explanation_query})
    st.write(response["answer"])


def refactor_code(file_content):
    """
    Use AI to refactor the given code.
    """
    st.subheader("Refactored Code:")

    # First, display the original code
    st.write("### Original Code:")
    st.code(file_content, language='python')  # Display the original code snippet


    # Create the refactor query
    refactor_query = f"""
    Instruction:
        "Act as an expert in code optimization and refactoring. Analyze the code that are got from fetching data from github repository link and generate a refactored version that:
        Improves efficiency (performance)
        Enhances readability and maintainability
        Boosts scalability
        DO NOT change function names or external dependencies.
        Keep the functionality identical to the original code but you can modify the source code to improve better effiency, readability, maintainability and scalability and keep the output be the same.
        Original Code:
        {file_content}
        Important feature: (To let the users know what changes made to their source code)
        After refactoring, include a summary explaining what changes were made and how they help achieve the improvements and the modified code.
    """
    
    # Request the refactor from the AI model
    response = st.session_state.conversation({"question": refactor_query})
    # refactored_code

    # Split the response into explanation and refactored code
    explanation, code_snippet = split_explanation_and_code(response["answer"])

    # Display explanation and refactored code
    st.write("### Explanation of Refactored Code:")
    st.write(explanation)  # Display the explanation of why the code was refactored this way

    st.write("### Refactored Code:")
    st.code(code_snippet, language='python')  # Display the refactored code


# Function to suggest features
def suggest_features(file_content):
    """
    Use AI to suggest features to add and provide modified code suggestions.
    """
    st.subheader("Suggested Features and Code Modifications:")
    suggestion_query = f"""
Analyze the following code and suggest features that would improve its functionality, usability, and performance.  
Provide:
- A list of suggested features.
- An explanation of why each feature is beneficial.
- The modified code incorporating those features.

Here is the code:
{file_content}
"""

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
                    explain_code_detailed(file_content, user_question)
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

    #New one
    if github_url:
    # Fetch data only if it's not already fetched for the current session
        if not st.session_state.get("repo_data_fetched", False):
            with st.spinner("Fetching repository contents..."):
            # Fetch repository contents and create a vector store
                repo_data, vector_store = read_all_repo_files(github_url, github_token)

            if repo_data:
                # Save the fetched repository data and vector store in session state
                st.session_state.repo_data = repo_data
                st.session_state.vector_store = vector_store
                st.session_state.repo_data_fetched = True  # Mark as fetched

                # Initialize the conversation chain
                st.session_state.conversation = get_conversation_chain(vector_store)

            else:
                st.error("Failed to fetch content from GitHub.")
    else:
        st.info("Please put a GitHub Repository Link for Data fetching")

    # Process the user's question
    if user_question:
        if st.session_state.get("conversation"):
            handle_user_input(user_question)
        else:
            st.error("Conversation chain not initialized. Please check the repository link.")

if __name__ == "__main__":
    main()