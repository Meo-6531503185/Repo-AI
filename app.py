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
from githubTest import *
from data_fetch.fetching import fetch_file_content, fetch_all_files_from_repo, fetch_repo_contents

load_dotenv()
github_token = os.getenv("GITHUB_TOKEN")
if github_token:
    print("GitHub Token loaded successfully")
else:
    print("GitHub Token not found")

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
        #chunks = get_text_chunks_for_refactoring(file_content)

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
    # st.subheader("Detailed Explanation of the Code:")
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
    return response["answer"]
    # st.write(response["answer"])


def refactor_code(file_content):
    """
    Use AI to refactor the given code.
    """
    # st.subheader("Refactored Code:")

    # # First, display the original code
    # st.write("### Original Code:")
    # st.code(file_content, language='python')  # Display the original code snippet


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

    return f"### Explanation of Refactored Code:\n{explanation}\n\n### Refactored Code:\n```python\n{code_snippet}\n```"

    # Display explanation and refactored code
    # st.write("### Explanation of Refactored Code:")
    # st.write(explanation)  # Display the explanation of why the code was refactored this way

    # st.write("### Refactored Code:")
    # st.code(code_snippet, language='python')  # Display the refactored code


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

    return f"### Explanation\n{explanation}\n\n### Modified Code\n```python\n{code_snippet}\n```" if code_snippet else f"### Explanation\n{explanation}"


    # Display explanation and suggested code
    # st.write("### Explanation")
    # st.write(explanation)
    # if code_snippet:
    #     st.write("### Modified Code")
    #     st.code(code_snippet, language='python')

#New fun for code
def handle_user_input(user_question):
    if "conversation" in st.session_state:
        conversation_chain = st.session_state.conversation

        # Check if the question is about a specific file
        import re
        match = re.search(r'["\']?([\w\-/]+(\.\w+))["\']?', user_question)
        file_name = match.group(1) if match else None

        special_query = any(keyword in user_question.lower() for keyword in ["explain", "refactor", "suggest"])

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

                #New 
                 # Handle user request types
                if "explain" in user_question.lower():
                    bot_reply = explain_code_detailed(file_content, user_question)
                elif "refactor" in user_question.lower():
                    bot_reply= refactor_code(file_content)
                elif "suggest" in user_question.lower():
                    bot_reply= suggest_features(file_content)
                else:
                    st.error("Unsupported operation. Please use 'explain', 'refactor', or 'suggest' in your question.")
                    return
                
                if special_query and bot_reply:
                    st.session_state.messages.append({"role": "user", "content": user_question})
                    st.session_state.messages.append({"role": "assistant", "content": bot_reply})

            else:
                st.error(f"File '{file_name}' not found in the repository.")
        else:
             # Handle non-file-specific questions as usual
            response = conversation_chain({"question": user_question})
            bot_reply = response.get("chat_history", ["No response."])[-1].content

            # Append to chat history
            # if not st.session_state.messages or st.session_state.messages[-1]["content"] != bot_reply:
            if bot_reply:
                st.session_state.messages.append({"role": "user", "content": user_question})
                st.session_state.messages.append({"role": "assistant", "content": bot_reply})
            return bot_reply
            # st.session_state.chat_history = response['chat_history']

            # for i, message in enumerate(st.session_state.chat_history):
            #     if i % 2 == 0:
            #         st.write(f"User: {message.content}")
            #     else:
            #         st.write(f"Bot: {message.content}")
    else:
        st.error("Conversation chain is not initialized. Please provide a repository URL first.")


# Function to display the chat history with a background color
def display_chat_history():
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.chat_message("user").markdown(f"{message['content']}")
        else:
            if message["content"]:
                st.chat_message("assistant").markdown(f"{message['content']}")
            else:
                st.chat_message("assistant").markdown("*No response available*")  
        # with st.chat_message(message["role"]):
        #     st.markdown(message["content"];


    # st.markdown(
    #     """
    #     <style>
    #     .chat-container {
    #         background-color: #f0f4f7;
    #         padding: 10px;
    #         border-radius: 8px;
    #         margin-bottom: 20px;
    #         max-height: 300px;
    #         overflow-y: auto;
    #     }
    #     .user-message {
    #         color: #333333;
    #         font-weight: bold;
    #         margin-bottom: 8px;
    #     }
    #     .bot-message {
    #         color: #0056b3;
    #         margin-bottom: 8px;
    #     }
    #     </style>
    #     """,
    #     unsafe_allow_html=True,
    # )

    # st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    # for i, message in enumerate(st.session_state.chat_history):
    #     if i % 2 == 0:
    #         st.markdown(f'<div class="user-message"> User: {message.content}</div>', unsafe_allow_html=True)
    #     else:
    #         st.markdown(f'<div class="bot-message"> Bot: {message.content}</div>', unsafe_allow_html=True)
    # st.markdown('</div>', unsafe_allow_html=True)

#Modified 0.2
# Main Streamlit application function
def main():
    st.set_page_config(page_title="GitHub Repositories Reader")


    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    # if "chat_history" not in st.session_state:
    #     st.session_state.chat_history = []
    if "messages" not in st.session_state:
        st.session_state.messages = []
        #Newly added one for code
    if "repo_data" not in st.session_state:  # Add repo_data initialization
        st.session_state.repo_data = {}

    if "GITHUB_APP_ID" not in st.session_state:
        st.session_state.GITHUB_APP_ID = ""
    if "GITHUB_PRIVATE_KEY_PATH" not in st.session_state:
        st.session_state.GITHUB_PRIVATE_KEY_PATH = ""
    if "GITHUB_REPOSITORY" not in st.session_state:
        st.session_state.GITHUB_REPOSITORY = ""

    st.header(":red[REPO AI]")
    st.subheader(":blue[Superduper performance] :rocket:")

    # Display chat history if available
    # if st.session_state.chat_history:
    # display_chat_history()

    #For Side bar ( Github Extra features)
    with st.sidebar:
        st.header("GitHub Credentials (Optional)")
        st.session_state.GITHUB_APP_ID = st.text_input("GitHub App ID", value=st.session_state.GITHUB_APP_ID)
        st.session_state.GITHUB_PRIVATE_KEY_PATH = st.text_input("Private Key Path", value=st.session_state.GITHUB_PRIVATE_KEY_PATH)
        st.session_state.GITHUB_REPOSITORY = st.text_input("GitHub Repository (e.g., owner/repo)", value=st.session_state.GITHUB_REPOSITORY)


    github_url = st.text_input("Enter GitHub repository URL:")
    user_question = st.chat_input("Ask a question about your GitHub repository:")

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

    #GitHub Credentials Handling
    elif st.session_state.GITHUB_APP_ID and st.session_state.GITHUB_PRIVATE_KEY_PATH and st.session_state.GITHUB_REPOSITORY:
        try:
            with open(st.session_state.GITHUB_PRIVATE_KEY_PATH, "r") as key_file:
                GITHUB_PRIVATE_KEY = key_file.read()
        except FileNotFoundError:
            st.error("Error: Private key file not found. Please provide a valid path.")
            return
        

        wrapper = GitHubAPIWrapper(
            github_repository=st.session_state.GITHUB_REPOSITORY,
            github_app_id=st.session_state.GITHUB_APP_ID,
            github_app_private_key=GITHUB_PRIVATE_KEY
        )

        st.success("Connected to GitHub successfully!")
        repo = wrapper.github.get_repo(st.session_state.GITHUB_REPOSITORY)

        # Fetch and display issues & pull requests
        st.subheader("Open Issues")
        st.write(wrapper.parse_issues(repo.get_issues(state="open")))

        st.subheader("Pull Requests")
        st.write(wrapper.parse_pull_requests(repo.get_pulls()))

        st.subheader("Repository Files")
        st.write(wrapper.list_files_in_main_branch())

        # Create a branch button
        if st.button("Create Test Branch"):
            result = wrapper.create_branch("test_branch")
            st.write(result)

    else:
        st.info("Please put a GitHub Repository Link for Data fetching")

    # Process the user's question
    if user_question:
        # if st.session_state.get("conversation"):
            handle_user_input(user_question)
            display_chat_history()
        # else:
        #     st.error("Conversation chain not initialized. Please check the repository link.")
    # if user_question and st.session_state.get("conversation"):
    #     handle_user_input(user_question)
    # elif user_question:
    #     st.error("Conversation chain not initialized. Please check the repository link.")

if __name__ == "__main__":
    main()