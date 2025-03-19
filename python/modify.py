import subprocess
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

from langchain_core.prompts import ChatPromptTemplate

from fetching import fetch_file_content, fetch_all_files_from_repo, fetch_repo_contents
import jwt
import time




load_dotenv()

github_token = os.getenv("GITHUB_APP_ID")
github_private_file = os.getenv("GITHUB_PRIVATE_KEY_PATH")

if github_token and github_private_file:

    print("GitHub Token loaded successfully")

else:

    print("GitHub Token not found")



#Untouch

import time
import jwt
import requests
import streamlit as st


def fetch_repo_contents(api_url, headers):
    """Fetch the contents of the entire repository."""
    response = requests.get(api_url, headers=headers)
    
    if response.status_code != 200:
        st.error(f"Failed to retrieve files from GitHub. Status code: {response.status_code}")
        return []
    return response.json()


def fetch_all_files_from_repo(api_url, headers):
    """Fetch all files from a GitHub repository, including files inside subfolders."""
    all_files = []
    repo_contents = fetch_repo_contents(api_url, headers)

    for file_info in repo_contents:
        if file_info["type"] == "file":
            all_files.append({
                "name": file_info["name"],
                "download_url": file_info["download_url"],  # Using download_url directly
                "path": file_info["path"]
            })
        elif file_info["type"] == "dir":
            # If it's a directory, recurse into it
            subfolder_url = file_info["url"]
            all_files += fetch_all_files_from_repo(subfolder_url, headers)

    return all_files


def fetch_file_content(file_url):
    """Fetch the content of a single file."""
    response = requests.get(file_url)
    if response.status_code == 200:
        return response.text
    else:
        st.warning(f"Failed to retrieve file: {file_url}")
        return ""


def read_all_repo_files(github_url, github_token, github_private_file):
    """Read all files in a repository and store them for later queries."""
    # Extract the owner and repo from the URL
    parts = github_url.rstrip("/").split("/")
    owner, repo = parts[-2], parts[-1]

    # GitHub API URL for repository contents
    api_url = f"https://api.github.com/repos/{owner}/{repo}/contents"

    # Load GitHub App private key
    with open(github_private_file, "r") as f:
        private_key = f.read()

    # Create a JWT for authentication
    payload = {
        "iat": int(time.time()),  # Issued at time
        "exp": int(time.time()) + (10 * 60),  # Expires in 10 minutes
        "iss": github_token,  # GitHub App ID
    }
    jwt_token = jwt.encode(payload, private_key, algorithm="RS256")

    # Get the installation ID
    headers = {
        "Authorization": f"Bearer {jwt_token}",
        "Accept": "application/vnd.github+json"
    }
    response = requests.get("https://api.github.com/app/installations", headers=headers)
    installations = response.json()
    installation_id = installations[0]["id"]  # Use the first installation

    # Exchange JWT for an installation access token
    response = requests.post(
        f"https://api.github.com/app/installations/{installation_id}/access_tokens",
        headers=headers
    )
    installation_token = response.json()["token"]

    # Use the installation token for API requests
    headers = {
        "Authorization": f"token {installation_token}",
        "Accept": "application/vnd.github+json"
    }

    # Fetch all files from the repository
    all_files = fetch_all_files_from_repo(api_url, headers)

    # Prepare data and chunking
    repo_data = {}
    all_file_chunks = []

    for file in all_files:
        file_content = fetch_file_content(file.get("download_url"))
        chunks = get_text_chunks(file_content)  # Assuming get_text_chunks exists
        all_file_chunks.extend(chunks)
        repo_data[file["path"]] = file_content

    # Generate vector store from chunks
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

    PROJECT_ID = "langchain-iris-chatbot"

    #PROJECT_ID = "repoai-440607"

    LOCATION = "us-central1"

    vertexai.init(project=PROJECT_ID, location=LOCATION)

    embeddings = VertexAIEmbeddings(model_name="text-embedding-004")

    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)

    return vector_store



# GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def get_conversation_chain(vector_store):



    retriever = vector_store.as_retriever(

        search_type = "mmr",

        search_kwargs = {"k":8},

    )



    # llm = ChatGroq(

    #      model= "deepseek-r1-distill-llama-70b",

    #      api_key = GROQ_API_KEY

    #  )

    llm = VertexAI(

        project="langchain-iris-chatbot",

        #project = "repoai-440607",

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







def extract_class_name(java_code):
    """Extract the first class name from Java code."""
    match = re.search(r'\bclass\s+(\w+)', java_code)
    return match.group(1) if match else "GeneratedCode"

model = VertexAI(model="gemini-1.5-pro")

def generate_java_code_and_test(user_code):
    """Generate Java source and test files based on user input."""
    output_dir = "../java/src/main/java/"
    os.makedirs(output_dir, exist_ok=True)

    class_name = extract_class_name(user_code)
    with open(os.path.join(output_dir, f"{class_name}.java"), "w") as file:
        file.write(user_code)
   
    system_template = f"""Generate JUnit test cases for the following user's Java program based on the official guidelines from https://junit.org/junit5/docs/current/user-guide/#writing-tests. 
Ensure the test cases cover valid inputs, invalid inputs, and edge cases. 

Follow these best practices:
- **Do NOT call private methods directly**; instead, test them indirectly through **public methods** such as getters and setters.
- Use **assertions** to verify expected behavior.
- Focus on **state validation** rather than implementation details.
- If necessary, use setter methods to modify the object state before calling the tested methods.
- Ensure edge cases, such as extreme values, null inputs (if applicable), and boundary conditions, are tested.
  
Provide only the raw Java code without any markdown or formatting symbols.  
Do NOT add the package name and class name at the top of the code.
"""
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("user", "{text}")
    ])
    prompt = prompt_template.invoke({"text": user_code})
    test_case = model.invoke(prompt)
    
    test_class_name = extract_class_name(test_case)
    output_dir = "../java/src/test/java/"
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, f"{test_class_name}.java"), "w") as file:
        file.write(test_case)

def run_maven_tests():
    """Run Maven tests and return the output."""
    maven_path = "C:\\Program Files\\apache-maven-3.9.5\\bin\\mvn.cmd"
    project_path = r"C:\\Users\\user\\Documents\\RepoAI_Github\\Repo-AI\\java"
    try:
        result = subprocess.run([maven_path, "test"], cwd=project_path, capture_output=True, text=True, check=True)
        return result.stdout, 0
    except subprocess.CalledProcessError as e:
        return e.stdout, e.returncode



def analyze_test_failures(test_output, user_code, requirements):
    system_template = """
    Analyze the following Maven test output and determine the cause of failure. 
    Explain why the test failed and classify whether the issue is in the class implementation or the test cases.
    
    Use logical reasoning rather than relying on specific keywords. Identify:
    - Whether the failure is due to missing methods, incorrect method signatures, or unexpected exceptions (likely requiring class refactoring).
    - Whether the failure is due to incorrect test expectations, improper assertions, or flawed test logic (likely requiring test case refactoring).
    - If the failure is ambiguous, suggest manual review.

    Based on your reasoning, determine the appropriate action:
    - If the class is incorrect, return "Regenerate Class File".
    - If the test cases need fixing, return "Regenerate Test Cases".
    - If it is unclear, return "Manual Review Needed".

    Provide a clear and structured explanation before your decision.
    """

    prompt_template = ChatPromptTemplate.from_messages(
        [("system", system_template), ("user", "{text}")]
    )

    prompt = prompt_template.invoke({"text": test_output})

    try:
        explanation = model.invoke(prompt)
        

        # Automatically trigger refactoring based on AI's analysis
        if "Regenerate Class File" in explanation:
            refined_code = refactor_code_with_ai(user_code, requirements, test_output)
            st.write("Class file has been regenerated. Running new tests...")
            generate_java_code_and_test(refined_code)

        elif "Regenerate Test Cases" in explanation:
            st.write("Test cases have issues. Generating updated test cases...")
            generate_java_code_and_test(user_code)

        else:
            st.write("Manual review is needed.")

    except Exception as e:
        print(f"An error occurred while processing test output: {e}")

def refactor_code_with_ai(user_code, requirements, reasons):
    """Use Vertex AI to refine the user code based on test output."""
    system_template = f"""
    Refactor the following Java code to meet the requirements and address the issues and potential threats.
    Ensure that the refactored code adheres to best practices and satisfies the user requirements. 
    Do not explain the changes made. Just refactor the code. Do not add test cases or file name.
    Provide only the raw Java code without any markdown or formatting symbols. Do not add the package name.

    ### Requirements ###
    {requirements}
    
    ### Reasons ###
    {reasons} are the test outputs why the previous AI-generated Java code did not pass the Maven tests. 
    Refactor the code accordingly to satisfy these issues.
    """
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("user", "{text}"),
        ("user", "{text2}")
    ])
    prompt = prompt_template.invoke({"requirements": requirements, "text": user_code, "text2": reasons})
    
    try:
        refined_code = model.invoke(prompt)
        print("AI successfully refined the code.")
        return refined_code
    except Exception as e:
        print(f"An error occurred while calling Vertex AI: {e}")
        return user_code

def handle_refactor_and_test(user_question):
    print("Handling refactor and test...")
    match = re.search(r'["\']?([\w\-/]+(\.\w+))["\']?', user_question)

    file_name = match.group(1) if match else None



    

    if file_name:

            if not st.session_state.repo_data:

                st.error("Repository data is empty. Please ensure the repository was loaded correctly.")

                return



            matching_files = [key for key in st.session_state.repo_data.keys() if file_name in key]

            if matching_files:

                file_content = st.session_state.repo_data[matching_files[0]]



                st.subheader(f"Source Code in '{file_name}':")

                st.code(file_content, language='python')

   
                """Main workflow for refactoring and testing."""
                refined_code = refactor_code_with_ai(file_content, user_question, reasons="")
                generate_java_code_and_test(refined_code)
    
                while True:
                    test_output, test_code = run_maven_tests()

                    if test_code == 0:
                        print("All tests passed!")
                        st.write(refined_code)
                        break
                    else:
                        print("Tests failed. Analyzing reasons...")
                        analyze_test_failures(test_output, refined_code, user_question)



def load_private_key():

    private_key_path = "//Users//pyaephyopaing//Pdf-Reader//repoai-api.2025-02-22.private-key.pem"

    try:

        with open(private_key_path, "r") as key_file:

            return key_file.read()

    except FileNotFoundError:

        print(f"Error: Private key file not found at {private_key_path}")

        return None

    except Exception as e:

        print(f"Error loading private key: {str(e)}")

        return None

    

def extract_repo_info(url):

    """Extracts the owner and repo name from a GitHub URL."""

    pattern = r"https://github\.com/([^/]+)/([^/]+)"

    match = re.match(pattern, url)

    

    if match:

        return match.group(1), match.group(2)  # Returns (owner, repo)

    return None, None  # Invalid URL format

    



def main():

    st.set_page_config(page_title="GitHub Repositories Reader")

    

    if "conversation" not in st.session_state:

        st.session_state.conversation = None

    if "messages" not in st.session_state:

        st.session_state.messages = []

    if "repo_data" not in st.session_state:

        st.session_state.repo_data = {}

    if "GITHUB_REPOSITORY" not in st.session_state:

        st.session_state.GITHUB_REPOSITORY = None





    
    INSTALL_URL = "https://github.com/apps/repoai-api"



    st.header(":red[REPO AI]")

    st.subheader(":blue[Superduper performance] :rocket:")



    with st.sidebar:

        st.header("GitHub Integration")

        st.markdown(f"[Click here to install RepoAI]({INSTALL_URL}) :link:")



    github_url = st.text_input("Enter GitHub repository URL (e.g., https://github.com/owner/repo):")



    user_question = st.chat_input("Ask a question about your GitHub repository:")



    if github_url:

        owner, repo = extract_repo_info(github_url)

        

        if owner and repo:

            full_repo_name = f"{owner}/{repo}"

            st.session_state.GITHUB_REPOSITORY = full_repo_name

            

            if not st.session_state.get("repo_data_fetched", False):

                with st.spinner(f"Fetching repository contents for {full_repo_name}..."):

                    repo_data, vector_store = read_all_repo_files(full_repo_name, github_token, github_private_file)



                if repo_data:

                    st.session_state.repo_data = repo_data

                    st.session_state.vector_store = vector_store

                    st.session_state.repo_data_fetched = True

                    st.session_state.conversation = get_conversation_chain(vector_store)

                    st.success(f"Successfully loaded repository: {full_repo_name}")

                else:

                    st.error("Failed to fetch content from GitHub.")

        else:

            st.error("Invalid GitHub URL. Please use the format: https://github.com/owner/repo")



    if user_question:
        
        refined_code = handle_refactor_and_test(user_question)
        
        if "push" in user_question.lower():
            print("Push command detected, executing push process...")
            st.write("Processing push request...")
            # push_refactored_code_to_github(refined_code)
    
          

       

        



if __name__ == "__main__":

    main() 
    