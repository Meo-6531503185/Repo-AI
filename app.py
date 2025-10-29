# import streamlit as st
# from dotenv import load_dotenv
# import os
# from langchain_google_vertexai import VertexAI
# import google.generativeai as palm
# from githubTest import *
# import re
# from fetching import *
# from processing import *
# from testing import *
# from push import *
# from langchain_core.prompts import ChatPromptTemplate


# load_dotenv()
# github_token = os.getenv("GITHUB_APP_ID")
# github_private_file = os.getenv("GITHUB_PRIVATE_KEY")
# if github_token and github_private_file:
#    print("GitHub Token loaded successfully")
# else:
#    print("GitHub Token not found")






# def extract_repo_info(url):
#    """Extracts the owner and repo name from a GitHub URL."""
#    pattern = r"https://github\.com/([^/]+)/([^/]+)"
#    match = re.match(pattern, url)
#    if match:
#        return match.group(1), match.group(2)  # Returns (owner, repo)
#    return None, None  # Invalid URL format

# ##New prompts
# def reasoning_user_questions(user_question, full_repo_name):
#    system_template = f"""
#    You are an expert AI code assistant. Your task is to analyze a user's request regarding a GitHub repository and classify their intent into ONE of the following precise action tags. Do NOT perform the action; only choose the tag.

# **AVAILABLE ACTION TAGS:**
# 1.  **REPO_OVERVIEW:** Request for a high-level summary, purpose, architecture, or key modules of the project.
# 2.  **REFACTOR_FILE:** Request to improve, fix bugs, or clean up an existing, specific file (e.g., "Refactor User.java").
# 3.  **ADD_FEATURE:** Request to implement a new feature, a new class, a new method, or add new functionality.
# 4.  **GENERATE_TESTS:** Request to create or update unit tests (e.g., "Generate tests for the AuthService class").
# 5.  **DOCS_GENERATE:** Request to create or update documentation (e.g., ReadMe, Javadoc).
# 6.  **DEPENDENCY_MGR:** Request to manage dependencies (e.g., update a library in pom.xml, fix a conflict).
# 7.  **PROJECT_ENHANCEMENT:** Broad, high-level request for non-specific, general improvement to the entire codebase.
# 8.  **DEBUG_TROUBLESHOOT:** Request to analyze an error, stack trace, or log to find the root cause.
# 9.  **PUSH_CODE:** Request includes an explicit command to save or push the resulting code to the repository.

# **OUTPUT FORMAT:**
# 1.  Provide a brief, structured explanation of your reasoning.
# 2.  On a new line, output only the chosen action tag enclosed in angle brackets, for machine parsing.

# **Example:** "The user wants to implement a new 'reset password' method, which is a new feature."
# **Decision:** <ADD_FEATURE>

# **User Request:** {user_question}
#    """
#    prompt_template = ChatPromptTemplate.from_messages(
#        [("system", system_template), ("user", "{{text}}")]
#    )
#    prompt = prompt_template.invoke({"text": user_question})
#    try:
#        explanation = model.invoke(prompt)
#        # Automatically trigger refactoring based on AI's analysis
#        if "<REPO_OVERVIEW>" in explanation:
#            #st.write("Generating repository overview...")
#            response = Overview_of_repository(user_question)
           
#        elif "<REFACTOR_FILE>" in explanation:
#            #st.write("Refactoring...")
#            file_content = handle_refactor_and_test(user_question)
#            response =  refactor_code_with_ai(file_content, user_question)
#         #    if detect_push_command(user_question):
#         #     #refined_code, test_case = handle_refactor_and_test(user_question)
#         #     response = push_refactored_code_to_github(response, test_case, full_repo_name)

#        elif "<PROJECT_ENHANCEMENT>" in explanation:
#            #st.write("Enhancing full project...")
#            response = Enhance_full_project(user_question)
           
#        elif "<ADD_FEATURE>" in explanation:
#            file_content = handle_refactor_and_test(user_question)

#            response = Add_feature(user_question,file_content)
      
#        else:
#            st.write("Manual review is needed.")
#        return response
#    except Exception as e:
#        print(f"An error occurred while processing test output: {e}")

# # def reasoning_user_questions(user_question, full_repo_name):
# #    system_template = """
# #    Analyze the user’s request and determine the appropriate action based on logical reasoning. Identify whether the user is asking for a repository explanation, a file-specific refactor, or a full project enhancement. Use structured analysis rather than relying on specific keywords. Identify:
# # If the request is for a repository explanation, analyze the project structure, dependencies, key modules, and overall architecture.
# # If the request is to refactor a specific file, analyze the file’s implementation, dependencies, and how changes will impact the project. Ensure entity classes, method signatures, and imports remain valid.
# # If the request is for a full project enhancement, assess overall code quality, detect redundant or inefficient code, and suggest large-scale structural improvements.
# # If refactoring impacts multiple files, ensure that related components (e.g., entity classes, services, repositories) are updated accordingly.
# # Determine the appropriate action based on your analysis:
# # If the request is for explanation, return "Provide Repository Overview".
# # If a specific file needs refactoring, return "Refactor File".
# # If the entire project requires enhancement, return "Enhance Full Project".
# # If the request is unclear, return "Manual Review Needed".
# # Provide a structured and clear explanation before your decision.
# #    """
# #    prompt_template = ChatPromptTemplate.from_messages(
# #        [("system", system_template), ("user", "{text}")]
# #    )
# #    prompt = prompt_template.invoke({"text": user_question})
# #    try:
# #        explanation = model.invoke(prompt)
# #        # Automatically trigger refactoring based on AI's analysis
# #        if "Provide Repository Overview" in explanation:
# #            #st.write("Generating repository overview...")
# #            response = Overview_of_repository(user_question)
# #        elif "Refactor File" in explanation:
# #            #st.write("Refactoring...")
# #            response, test_case= handle_refactor_and_test(user_question)
# #            if detect_push_command(user_question):
# #             #refined_code, test_case = handle_refactor_and_test(user_question)
# #             response = push_refactored_code_to_github(response, test_case, full_repo_name)

          
# #        elif "Enhance Full Project" in explanation:
# #            #st.write("Enhancing full project...")
# #            response = Enhance_full_project(user_question)
      
# #        else:
# #            st.write("Manual review is needed.")
# #        return response
# #    except Exception as e:
# #        print(f"An error occurred while processing test output: {e}")
       
# def handle_refactor_and_test(user_question):
#     #st.write("Handling refactor and test...")
#     match = re.search(r'["\']?([\w\-/]+(\.\w+))["\']?', user_question)
#     file_name = match.group(1) if match else None
#     if file_name:
#             if not st.session_state.repo_data:
#                 st.error("Repository data is empty. Please ensure the repository was loaded correctly.")
#                 return
#             matching_files = [key for key in st.session_state.repo_data.keys() if file_name in key]
#             if matching_files:
#                 file_content = st.session_state.repo_data[matching_files[0]]
#                 return  file_content
#                 # refined_code = refactor_code_with_ai(file_content, user_question, reasons="")
#                 # test_case = generate_java_code_and_test(refined_code)
#                 # while True:
#                 #     test_output, test_code = run_maven_tests()
#                 #     if test_code == 0:
#                 #         print("All tests passed!")
#                 #         #st.write("Refactored code has been successfully generated.")
#                 #         #st.code(refined_code, language='java')
#                 #         return refined_code, test_case
#                 #     else:
#                 #         print("Tests failed. Analyzing reasons...")
#                 #         analyze_test_failures(test_output, refined_code, user_question)
#             else:
#                 st.write("No file detected...")
#     else:
#         st.write("No file detected...")




# def main():
#    st.set_page_config(page_title="GitHub Repositories Reader")
#    if "conversation" not in st.session_state:
#        st.session_state.conversation = None
#    if "messages" not in st.session_state:
#        st.session_state.messages = []
#    if "repo_data" not in st.session_state:
#        st.session_state.repo_data = {}
#    if "GITHUB_REPOSITORY" not in st.session_state:
#        st.session_state.GITHUB_REPOSITORY = None
#    INSTALL_URL = "https://github.com/apps/repoai-api"



#    col1, col2 = st.columns([5, 1])  # Swap the ratios

#    with col1:
#     st.markdown(
#         """
#         <div style='display: flex; align-items: center; height: 100%;'>
#             <h1 style='color: #23a445; margin: 0;'>REPO AI</h1>
#         </div>
#         """,
#         unsafe_allow_html=True
#     )

# #    with col2:
# #     st.image("repoAi Logo.png", width=100)  # Logo on the right




#    #st.markdown(
#     #"<h2 style='color: #28a745; font-size: 50px;'>Fast & Reliable performance 🚀</h2>",
#     #unsafe_allow_html=True
#     #)
#    with st.sidebar:
#        st.header("GitHub Integration")
#        st.markdown(f"[Click here to install RepoAI]({INSTALL_URL}) :link:")
#    github_url = st.text_input("Enter GitHub repository URL (e.g., https://github.com/owner/repo):")
#    user_question = st.chat_input("Ask a question about your GitHub repository:")
#    if github_url:
#        owner, repo = extract_repo_info(github_url)
#        if owner and repo:
#            full_repo_name = f"{owner}/{repo}"
#            st.session_state.GITHUB_REPOSITORY = full_repo_name
#            if not st.session_state.get("repo_data_fetched", False):
#                with st.spinner(f"Fetching repository contents for {full_repo_name}..."):
#                    repo_data, vector_store = read_all_repo_files(full_repo_name)
#                if repo_data:
#                    st.session_state.repo_data = repo_data
#                    st.session_state.vector_store = vector_store
#                    st.session_state.repo_data_fetched = True
#                    st.session_state.conversation = get_conversation_chain(vector_store)
#                    st.success(f"Successfully loaded repository: {full_repo_name}")
#                else:
#                    st.error("Failed to fetch content from GitHub.")
#        else:
#            st.error("Invalid GitHub URL. Please use the format: https://github.com/owner/repo")

#    if user_question:
#        # Save user message
#        st.session_state.messages.append({"role": "user", "content": user_question})


#        # Get AI response
#        response = reasoning_user_questions(user_question, full_repo_name)



#        st.session_state.messages.append({"role": "AI", "content": response})

#    for message in st.session_state.messages:
#     if message["role"] == "user":
#         st.chat_message("user").markdown(message["content"])
#     else:
#         raw_content = message.get("content", "")

#         # Handle tuple content (e.g., (response, push_message))
#         if isinstance(raw_content, tuple):
#             content = "\n\n".join(str(part) for part in raw_content if part)
#         else:
#             content = str(raw_content)

#         if content:
#             is_code = (
#                 content.strip().startswith(("class ", "def ", "import ", "public ", "function ", "#include")) or
#                 any(x in content for x in ["{", "}", ";", "=>"])
#             )

#             with st.chat_message("assistant"):
#                 if is_code:
#                     st.code(content, language="java")
#                 else:
#                     st.markdown(content.replace("\n", "  \n"))  # For line breaks
#         else:
#             st.chat_message("assistant").markdown("*No response available*")



# if __name__ == "__main__":
#    main()

#############################Autogen######################################
import streamlit as st
from dotenv import load_dotenv
import os, re, asyncio
from langchain_google_vertexai import VertexAI
from langchain_core.prompts import ChatPromptTemplate
from autogen_core import (
    RoutedAgent,
    message_handler,
    MessageContext, SingleThreadedAgentRuntime, AgentId
)
from githubTest import *
from fetching import *
from processing import *
from testing import *
from push import *
from agents.sub_agents import (
    OverviewAgent,
    MultiFileRefactorAgent,
    EnhanceAgent
)
from agents.push_agent import PushAgent

# === Load environment ===
load_dotenv()
github_token = os.getenv("GITHUB_APP_ID")
github_private_file = os.getenv("GITHUB_PRIVATE_KEY")
if github_token and github_private_file:
    print("✅ GitHub Token loaded successfully")
else:
    print("⚠️ GitHub Token not found")

# === Helper: Extract repo info ===
def extract_repo_info(url: str):
    pattern = r"https://github\.com/([^/]+)/([^/]+)"
    match = re.match(pattern, url)
    return match.groups() if match else (None, None)

from dataclasses import dataclass

@dataclass
class ReasoningRequest:
    question: str

# === Reasoning Agent ===
class ReasoningAgent(RoutedAgent):
    def __init__(self):
        super().__init__("reasoning_agent")

    @message_handler
    async def handle_reasoning(self, message: ReasoningRequest, ctx: MessageContext) -> str:
        """Agent handler for reasoning about repo questions."""
        user_question = message.question

        system_template = f"""
You are an expert AI code assistant. Analyze the user's request and choose ONE of the following action tags.

**TAGS:**
REPO_OVERVIEW, REFACTOR_FILE, ADD_FEATURE, GENERATE_TESTS, DOCS_GENERATE,
DEPENDENCY_MGR, PROJECT_ENHANCEMENT, DEBUG_TROUBLESHOOT, PUSH_CODE.

**OUTPUT FORMAT:**
1. Provide a short reasoning explanation.
2. On a new line, output ONLY the chosen tag for machine parsing.

**User Request:** {user_question}
"""

        prompt_template = ChatPromptTemplate.from_messages(
            [("system", system_template), ("user", "{text}")]
        )
        prompt = prompt_template.invoke({"text": user_question})

        try:
            model = VertexAI(model_name="gemini-2.5-flash")
            explanation = model.invoke(prompt)

            lines = [line.strip() for line in explanation.strip().splitlines() if line.strip()]
            chosen_tag = lines[-1] if lines else None

            if chosen_tag == "REPO_OVERVIEW":
                response = OverviewAgent().run(user_question)
            elif chosen_tag == "REFACTOR_FILE":
                file_results = MultiFileRefactorAgent().run(user_question, st.session_state.repo_data)
                response = ""
                for f, content in file_results.items():
                    response += f"\n// === {f} ===\n{content}\n"
            elif chosen_tag == "PROJECT_ENHANCEMENT":
                response = EnhanceAgent().run(user_question)
            # elif chosen_tag == "ADD_FEATURE":
            #     response = MultiFileFeatureAgent().run(user_question, st.session_state.repo_data)
            else:
                response = "Manual review is needed."

            return response

        except Exception as e:
            return f"❌ Error in reasoning agent: {e}"


# === Main Streamlit UI ===
def main():
    st.set_page_config(page_title="GitHub Repositories Reader")

    # --- Initialize session ---
    for key, default in {
        "conversation": None,
        "messages": [],
        "repo_data": {},
        "GITHUB_REPOSITORY": None,
    }.items():
        if key not in st.session_state:
            st.session_state[key] = default

    INSTALL_URL = "https://github.com/apps/repoai-api"

    # --- Header ---
    col1, col2 = st.columns([5, 1])
    with col1:
        st.markdown(
            """
            <div style='display: flex; align-items: center; height: 100%;'>
                <h1 style='color: #23a445; margin: 0;'>REPO AI</h1>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --- Sidebar ---
    with st.sidebar:
        st.header("GitHub Integration")
        st.markdown(f"[Click here to install RepoAI]({INSTALL_URL}) :link:")

    github_url = st.text_input("Enter GitHub repository URL:")
    user_question = st.chat_input("Ask a question about your GitHub repository:")

    # --- GitHub handling ---
    if github_url:
        owner, repo = extract_repo_info(github_url)
        if owner and repo:
            full_repo_name = f"{owner}/{repo}"
            st.session_state.GITHUB_REPOSITORY = full_repo_name

            if not st.session_state.get("repo_data_fetched", False):
                with st.spinner(f"Fetching {full_repo_name}..."):
                    repo_data, vector_store = read_all_repo_files(full_repo_name)
                if repo_data:
                    st.session_state.repo_data = repo_data
                    st.session_state.vector_store = vector_store
                    st.session_state.repo_data_fetched = True
                    st.session_state.conversation = get_conversation_chain(vector_store)
                    st.success(f"Loaded repository: {full_repo_name}")
                else:
                    st.error("Failed to fetch content from GitHub.")
        else:
            st.error("Invalid GitHub URL format.")

    # === Run the reasoning agent ===
    if user_question:
        st.session_state.messages.append({"role": "user", "content": user_question})

        async def run_reasoning_agent():
            runtime = SingleThreadedAgentRuntime()
            await ReasoningAgent.register(runtime, "reasoning_agent", lambda: ReasoningAgent())
            runtime.start()
            response = await runtime.send_message(
                ReasoningRequest(user_question),
                AgentId("reasoning_agent", "default"),
            )
            await runtime.stop_when_idle()
            await runtime.close()
            return response

        response = asyncio.run(run_reasoning_agent())
        st.session_state.messages.append({"role": "AI", "content": response})

    # === Display chat history ===
    # for message in st.session_state.messages:
    #     role = message["role"]
    #     content = message.get("content", "")
    #     with st.chat_message(role):
    #         if role == "assistant" and any(
    #             x in content for x in ["class ", "def ", "import ", "public ", "{", "}", ";"]
    #         ):
    #             st.code(content, language="java")
    #         else:
    #             st.markdown(content.replace("\n", "  \n"))
    #         # === Push button for AI-generated code ===
    #         if message["role"] == "AI" and "// ===" in content:
    #             push_agent = PushAgent(
    #                 github_app_id=github_token,
    #                 github_private_key=github_private_file
    #             )

    #             if st.button("🚀 Push this refactored code to GitHub", key=f"push_{hash(content)}"):
    #                 with st.spinner("Pushing refactored code to GitHub..."):
    #                     result = push_agent.run(
    #                         ai_response=content,
    #                         repo_name=st.session_state.GITHUB_REPOSITORY
    #                     )
    #                 st.success(result)
    # === Display chat history ===
    for i, message in enumerate(st.session_state.messages):
        role = message["role"]
        content = message.get("content", "")

        with st.chat_message(role):
            # Show code if it looks like code
            if role == "AI" and any(
                kw in content for kw in ["class ", "public ", "def ", "import ", "{", "}", ";"]
            ):
                st.code(content, language="java")
            else:
                st.markdown(content.replace("\n", "  \n"))

            # === Detect refactored output (more reliable condition) ===
            if (
                role == "AI"
                and ("// ===" in content or "refactored" in content.lower() or "public class" in content)
            ):
                st.markdown("---")  # visual separator
                st.info("This looks like refactored Java code. You can push it to GitHub:")

                push_agent = PushAgent(
                    github_app_id=github_token,
                    github_private_key=github_private_file
                )

                if st.button("🚀 Push this refactored code to GitHub", key=f"push_{i}"):
                    with st.spinner("Pushing refactored code to GitHub..."):
                        result = push_agent.run(
                            ai_response=content,
                            repo_name=st.session_state.GITHUB_REPOSITORY
                        )
                    st.success(result)




if __name__ == "__main__":
    main()
