import subprocess
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
import streamlit as st
from dotenv import load_dotenv
from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter,
)
from langchain_community.vectorstores import FAISS
from langchain.llms import HuggingFaceHub
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import tiktoken
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
from langchain_groq import ChatGroq
from githubTest import *
from langchain_core.prompts import ChatPromptTemplate
import jwt
import time
import re
from google.cloud import aiplatform
from vertexai.language_models import TextEmbeddingModel
from langchain.docstore.document import Document


load_dotenv()
github_token = os.getenv("GITHUB_APP_ID")
github_private_file = os.getenv("GITHUB_PRIVATE_KEY_PATH")
if github_token and github_private_file:
    print("GitHub Token loaded successfully")
else:
    print("GitHub Token not found")


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
        content = response.text
        return content
    else:
        print(f"Failed to retrieve file: {file_url}, Status Code: {response.status_code}")
        return ""
    
    
def read_all_repo_files(github_url):
    """Read all files in a repository and store them for later queries."""
    # Extract the owner and repo from the URL
    parts = github_url.rstrip("/").split("/")
    owner, repo = parts[-2], parts[-1]
    # GitHub API URL for repository contents
    api_url = f"https://api.github.com/repos/{owner}/{repo}/contents"
    # Load GitHub App private key
    
    installation_token = get_installation_token()
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
        
        if not isinstance(file_content, str):
            st.write(f"ERROR: Expected string, got {type(file_content)} for file {file['path']}")
            continue  # Skip this file if it's not a string
        chunks = get_text_chunks(file_content)  # Assuming get_text_chunks exists
        all_file_chunks.extend(chunks)
        repo_data[file["path"]] = file_content
    # Generate vector store from chunks
    vector_store = get_vector_store(all_file_chunks)
    return repo_data, vector_store

    
cached_jwt = None
jwt_expiry = 0   
def get_jwt():
        global cached_jwt, jwt_expiry
        now = int(time.time())

        with open(github_private_file, "r") as f:
            private_key = f.read()
        
        # Check if JWT is still valid
        if cached_jwt and now < jwt_expiry - 60:  # Refresh 1 min before expiry
            return cached_jwt  # Return cached JWT

        # Generate a new JWT
        payload = {
            "iat": now,
            "exp": now + 600,  # Expires in 10 min
            "iss": github_token,  # GitHub App ID
        }
        
        cached_jwt = jwt.encode(payload, private_key, algorithm="RS256")
        jwt_expiry = now + 600  # Update expiry time
        # st.write(f"Now: {now}, Expiration Time: {now + 600}")

        return cached_jwt  # Return new JWT

def get_installation_token():
        jwt_token = get_jwt()
        headers = {
            "Authorization": f"Bearer {jwt_token}",
            "Accept": "application/vnd.github+json",
        }

        # Fetch installation ID
        url = "https://api.github.com/app/installations"
        response = requests.get(url, headers=headers)


        installations = response.json()
        installation_id = installations[0]["id"]

        # Generate installation token
        token_url = f"https://api.github.com/app/installations/{installation_id}/access_tokens"
        token_response = requests.post(token_url, headers=headers).json()

        return token_response["token"]  # This lasts 1 hour
    

def count_tokens(text, model_name="text-embedding-004"):
    enc = tiktoken.get_encoding("cl100k_base")  # Good approximation
    return len(enc.encode(text))

def get_text_chunks(extracted_text,max_tokens = 20000, overlap = 200):
    java_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.JAVA, chunk_size=1500, chunk_overlap=50
    )

    if isinstance(extracted_text, list):
        extracted_text = " ".join(extracted_text)
        
    words = java_splitter.split_text(extracted_text)
    
    chunks = []
    start = 0

    while start < len(words):
        end = min(start + (max_tokens // 1500), len(words))
        chunk = " ".join(words[start:end])
        
        # Ensure the chunk size is within the limit
        while count_tokens(chunk) > max_tokens and end > start:
            end -= 100  # Reduce chunk size if too large
            chunk = " ".join(words[start:end])
        
        chunks.append(chunk)
        start = end - (overlap // 1500)  # Ensure overlap for continuity

    return chunks


def get_vector_store(text_chunks):
    chunks = get_text_chunks(text_chunks)
    PROJECT_ID = "coderefactoringai"
    LOCATION = "us-central1"
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    model = VertexAIEmbeddings(model_name="text-embedding-004")

    # Wrap chunks into LangChain Document objects
    documents = []
    for chunk in chunks:
        token_count = count_tokens(chunk)
        print(f"Chunk token count: {token_count}")
        documents.append(Document(page_content=chunk))

    # Create FAISS vector store
    vector_store = FAISS.from_documents(documents, model)
    return vector_store

# GROQ_API_KEY = os.getenv("GROQ_API_KEY")
def get_conversation_chain(vector_store):
    retriever = vector_store.as_retriever(
        search_type = "mmr",
        search_kwargs = {"k":8},
    )

    llm = VertexAI(
        project = "coderefactoringai",
        location="us-central1",
        model = "gemini-1.5-pro",
        model_kwargs={
            "temperature": 0.7,
            "max_length": 600,
            "top_p": 0.95,
            "top_k": 3,
        },
    )
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
    )
    return conversation_chain

def extract_class_name(java_code):
    """Extract the first class name from Java code."""
    match = re.search(r'\bclass\s+(\w+)', java_code)
    return match.group(1) if match else "GeneratedCode"

#model
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
    maven_path = "/opt/homebrew/opt/maven/bin/mvn"
    project_path = "/Users/soemoe/MFU/3rd Year/Seminar/Repo Ai/java"
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
            st.write("Class file has some issues. Class file has been regenerated. Running new tests...")
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
    Explain the changes made, and refactor the code. Do not add test cases or file name.
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
        return f"```java\n{refined_code}\n```"
        # return refined_code
    except Exception as e:
        print(f"An error occurred while calling Vertex AI: {e}")
        return user_code
    
def handle_refactor_and_test(user_question):
    st.write("Handling refactor and test...")
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
                st.code(file_content, language='java')
                """Main workflow for refactoring and testing."""
                refined_code = refactor_code_with_ai(file_content, user_question, reasons="")
                generate_java_code_and_test(refined_code)
                while True:
                    test_output, test_code = run_maven_tests()
                    if test_code == 0:
                        print("All tests passed!")
                        # st.code(refined_code)
                        return f"\n{refined_code}"
                    else:
                        print("Tests failed. Analyzing reasons...")
                        analyze_test_failures(test_output, refined_code, user_question)
            else:
                st.write("No file detected...")
                
    else:
        st.write("No file detected...")


def extract_repo_info(url):
    """Extracts the owner and repo name from a GitHub URL."""
    pattern = r"https://github\.com/([^/]+)/([^/]+)"
    match = re.match(pattern, url)
    if match:
        return match.group(1), match.group(2)  # Returns (owner, repo)
    return None, None  # Invalid URL format


def detect_push_command(user_input):
    """Detects if the user wants to push code based on various phrases."""
    push_patterns = [
        r"\bpush\b",
        r"\bcommit\b",
        r"\bupload\b",
        r"\bdeploy\b"
    ]
    for pattern in push_patterns:
        if re.search(pattern, user_input, re.IGNORECASE):
            return True
    return False 
  
def push_refactored_code_to_github(refined_code, github_url):
    if not github_url:
        st.error("GitHub repository or token is missing.")
        return
    # Authenticate with GitHub API
    wrapper = GitHubAPIWrapper(
    github_repository = github_url
,
    github_app_id= github_token,
   github_app_private_key = github_private_file
)
    wrapper.create_branch("refactored_code")
    # Define file details
    file_path = "src/main/java/RefactoredClass.java"

    refactored_code = refined_code
    file_query = f"{file_path}\n\n{refactored_code}"
    # Push new file
    wrapper.create_file(file_query)
    st.success("Refactored code has been pushed to the repository.")
 
#reasoning_parts

def reasoning_user_questions(user_question, full_repo_name):
    system_template = """
    Analyze the user’s request and determine the appropriate action based on logical reasoning. Identify whether the user is asking for a repository explanation, a file-specific refactor, or a full project enhancement. Use structured analysis rather than relying on specific keywords. Identify:
If the request is for a repository explanation, analyze the project structure, dependencies, key modules, and overall architecture.
If the request is to refactor a specific file, analyze the file’s implementation, dependencies, and how changes will impact the project. Ensure entity classes, method signatures, and imports remain valid.
If the request is for a full project enhancement, assess overall code quality, detect redundant or inefficient code, and suggest large-scale structural improvements.
If refactoring impacts multiple files, ensure that related components (e.g., entity classes, services, repositories) are updated accordingly.
Determine the appropriate action based on your analysis:
If the request is for explanation, return "Provide Repository Overview".
If a specific file needs refactoring, return "Refactor File".
If the entire project requires enhancement, return "Enhance Full Project".
If the request is unclear, return "Manual Review Needed".
Provide a structured and clear explanation before your decision.
    """
    prompt_template = ChatPromptTemplate.from_messages(
        [("system", system_template), ("user", "{text}")]
    )
    prompt = prompt_template.invoke({"text": user_question})
    try:
        explanation = model.invoke(prompt)
        # Automatically trigger refactoring based on AI's analysis
        if "Provide Repository Overview" in explanation:
            st.write("Generating repository overview...")
            response = Overview_of_repository(user_question)
        elif "Refactor File" in explanation:
            st.write("Refactoring...")
            response = handle_refactor_and_test(user_question)
            if detect_push_command(user_question):
                
                print("Push command detected, executing push process...")
                st.write("Processing push request...")
            
                push_refactored_code_to_github(response, full_repo_name)
                
            
        elif "Enhance Full Project" in explanation:
            st.write("Enhancing full project...")
            response = Enhance_full_project(user_question)
        
        else:
            st.write("Manual review is needed.")
        return response
    except Exception as e:
        print(f"An error occurred while processing test output: {e}")

def Overview_of_repository(user_question):
    # st.write("Overview being called...")

    repo_data = st.session_state.repo_data

    # Inspect the repo_data structure
    # st.write("repo_data contents:", repo_data)

    # Ensure that repo_data contains files
    if not repo_data:
        st.write("Error: No files found in repo_data")
        return None

    # Extract file names and their contents
    file_contents = []
    for filename, content in repo_data.items():
        file_contents.append(f"File: {filename}\n{content}\n\n")  # Adding filename for clarity

    if not file_contents:
        st.write("No files were found in repo_data.")
        return None
    
    # st.write("File contents:", file_contents)

    # Escape the curly braces in the system template
    system_template = f"""
    Analyze the structure of the given GitHub repository files, which contains the following files and modules:
    {{file_contents}}

    Based on this information, please provide a clear, high-level explanation of the repository. 
    Identify the purpose of the project, its key modules, and dependencies. 
    Describe the main components, such as entity classes, services, controllers, and utility functions. 
    Summarize how different parts of the codebase interact, highlighting any important frameworks or technologies used (e.g., Spring Boot, Hibernate, REST APIs). 
    Keep the explanation concise and developer-friendly.
    """


    # Create the prompt with user input and system context
    prompt_template = ChatPromptTemplate.from_messages(
        [("system", system_template), ("user", "{text}")]
    )

    # Prepare the prompt with the user's question
    prompt_input = prompt_template.invoke({"file_contents":file_contents,"text": user_question})

    try:
        # Assuming 'model' is the language model you're using to get the response
        explanation = model.invoke(prompt_input)
        # st.write("Model Response:\n", explanation)  # Check the model's response
        return explanation
    except Exception as e:
        st.write(f"An error occurred while processing the output: {e}")
        return None


        
def Enhance_full_project(user_question):
    repo_data = st.session_state.repo_data
    if not repo_data:
        st.write("Error: No files found in repo_data")
        return None
    file_contents = []
    for filename, content in repo_data.items():
        file_contents.append(f"File: {filename}\n{content}\n\n")  # Adding filename for clarity

    if not file_contents:
        st.write("No files were found in repo_data.")
        return None
    
    
    system_template = f"""
    Perform a comprehensive enhancement of the entire project {{file_contents}} by analyzing and refactoring code while maintaining all existing functionality. Follow a structured approach:*

1️ Codebase Analysis:

Identify code duplication, inefficiencies, and areas that violate best practices.

Ensure consistent formatting, proper naming conventions, and modularization.

2️ Dependency & Configuration Improvements:

Optimize pom.xml (remove unused dependencies, update outdated libraries).

Improve project configurations (application.properties, environment settings).

3️ Performance Enhancements:

Optimize database queries, caching strategies, and avoid redundant computations.

Improve asynchronous processing where applicable.

4️ Architecture & Design Patterns:

Ensure the project follows clean architecture principles (e.g., layered architecture, microservices).

Refactor tightly coupled classes to improve maintainability.

5️ Testing & Validation:

Ensure all modifications pass unit tests, integration tests, and build verification (mvn test, mvn verify).

If test coverage is low, suggest or generate missing test cases.

6️ Final Verification & Recommendations:

Summarize all enhancements and their impact.

Ensure the project builds successfully and all improvements align with best practices.

Return a structured summary of enhancements and confirm that the project remains functional after modifications.
    """
    prompt_template = ChatPromptTemplate.from_messages(
        [("system", system_template), ("user", "{text}")]
    )
    prompt_input = prompt_template.invoke({"file_contents":file_contents,"text": user_question})
    
    try:
        explanation = model.invoke(prompt_input)
        st.write(explanation)
        return explanation
        
    except Exception as e:
        print(f"An error occurred while processing test output: {e}")
         
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
                    repo_data, vector_store = read_all_repo_files(full_repo_name)
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
        # Save user message
        st.session_state.messages.append({"role": "user", "content": user_question})

        # Get AI response
        response = reasoning_user_questions(user_question, full_repo_name)

        if detect_push_command(user_question):
            push_refactored_code_to_github(response, full_repo_name)

        # Save AI response
        st.session_state.messages.append({"role": "AI", "content": response})


    for message in st.session_state.messages:
        if message["role"] == "user":
            st.chat_message("user").markdown(f"**User:** {message['content']}")
        else:
            content = message.get("content", "")
            if content:
                # Simple heuristic to check if message is code-like
                is_code = (
                    "\n" in content or
                    content.strip().startswith(("class ", "def ", "import ", "public ", "function ", "#include")) or
                    any(x in content for x in ["{", "}", ";", "=>"])
                )
                with st.chat_message("assistant"):
                    if is_code:
                        st.code(content, language='java')  # Change 'java' based on your output
                    else:
                        st.markdown(f"**AI:** {content}")
            else:
                st.chat_message("assistant").markdown("*No response available*")


if __name__ == "__main__":
    main()
