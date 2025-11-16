
#############################Autogen######################################
import streamlit as st
from dotenv import load_dotenv
import os, re, asyncio
from typing import Dict, Any, Tuple, List
import ast
import inspect
from dataclasses import dataclass

# --- Imports (Assuming these files/packages are available in your project structure) ---
# LLM/Embedding Imports
from langchain_google_vertexai import VertexAI, VertexAIEmbeddings 
from langchain_core.prompts import ChatPromptTemplate

# Agent Core Imports
from autogen_core import (
    RoutedAgent,
    message_handler,
    MessageContext, SingleThreadedAgentRuntime, AgentId
)

# Custom Agent/Utility Imports (Must be defined externally)
from githubTest import * # Assuming GitHubAPIWrapper is here
from fetching import * # Assuming read_all_repo_files is here
from processing import * # Assuming other utilities are here
from agents.sub_agents import (
    MultiFileRefactorAgent,
)
from agents.push_agent import PushAgent # Assuming PushAgent is in agents/sub_agents.py or a separate file

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ADD THESE IMPORTS AFTER EXISTING IMPORTS
from validators.code_validators import ValidationPipeline
from utils.error_handlers import ErrorHandler, UserFriendlyError, ErrorCategory
from utils.llm_normalizer import LLMOutputNormalizer, CodeExtractor


# === Load environment ===
load_dotenv()
github_token = os.getenv("GITHUB_APP_ID")
github_private_file = os.getenv("GITHUB_PRIVATE_KEY")
if github_token and github_private_file:
    print("✅ GitHub Token loaded successfully")
else:
    print("⚠️ GitHub Token not found")

# === Initialize embedding model once ===
try:
    # Use the official Google Vertex AI Embeddings client
    model = VertexAIEmbeddings(model_name="text-embedding-004")
except Exception as e:
    st.error(f"Error initializing VertexAI Embeddings: {e}")
    model = None # Handle case where model fails to initialize

# === Helper: Extract repo info ===
def extract_repo_info(url: str):
    """Extracts owner and repo name from a GitHub URL."""
    pattern = r"https://github\.com/([^/]+)/([^/]+)"
    match = re.match(pattern, url)
    return match.groups() if match else (None, None)

def extract_code_chunks(code_string: str) -> Dict[str, str]:
    """
    Splits a Python code string into meaningful chunks (functions and classes) 
    using the Abstract Syntax Tree (AST).
    """
    chunks = {}
    if not code_string or not isinstance(code_string, str):
        return {"Full Code Result": ""}
    
    try:
        # We must use inspect.getsource, which requires the AST node to be compiled/executed 
        # within a temporary module structure. This is often complex and prone to errors 
        # unless the code is perfectly formatted/self-contained.
        # Fallback to simple line-based splitting if AST fails on complex input.
        
        tree = ast.parse(code_string)
        
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = node.name
                source_code = inspect.getsource(
                    compile(ast.Module(body=[node], type_ignores=[]), '<string>', 'exec')
                )
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    chunks[f"Function: {name}"] = source_code.strip()
                elif isinstance(node, ast.ClassDef):
                    chunks[f"Class: {name}"] = source_code.strip()
            
    except Exception as e:
        # Fallback for non-Python code or complex syntax errors
        # print(f"AST parsing failed: {e}. Using single chunk.")
        if code_string.strip():
            chunks["Full Code Result"] = code_string.strip()
        
    if not chunks and code_string.strip():
        chunks["Full Code Result"] = code_string.strip()

    return chunks

# === Intent Similarity Function ===
def check_intent_similarity(user_request: str, full_refactored_code: str, embedding_model: VertexAIEmbeddings) -> float:
    """
    Checks intent fulfillment by finding the MAXIMUM cosine similarity 
    between the user intent and ALL logical code chunks.
    """
    if not embedding_model:
        return 0.0 # Return 0 if the embedding model failed to load
        
    try:
        # 1. Get Intent Embedding
        user_emb = embedding_model.embed_query(user_request)
        
        # 2. Extract Code Chunks
        code_chunks = extract_code_chunks(full_refactored_code)
        
        max_score = 0.0
        
        # 3. Compare Intent to Each Code Chunk
        for chunk_code in code_chunks.values():
            if not chunk_code: continue
            
            # Get Chunk Embedding
            chunk_emb = embedding_model.embed_query(chunk_code)
            
            # Calculate Similarity
            # Ensure embeddings are numpy arrays for cosine_similarity
            user_array = np.array(user_emb).reshape(1, -1)
            chunk_array = np.array(chunk_emb).reshape(1, -1)
            
            current_score = cosine_similarity(user_array, chunk_array)[0][0]
            
            if current_score > max_score:
                max_score = current_score

        # 4. Format Output
        percentage = round(max_score * 100, 2)
        return percentage
        
    except Exception as e:
        print(f"Error calculating intent similarity: {e}")
        return 0.0


@dataclass
class ReasoningRequest:
    question: str


def _parse_tag(explanation_output: str) -> str:
    """Robustly extracts the action tag from the LLM response."""
    lines = [line.strip() for line in explanation_output.strip().splitlines() if line.strip()]
    if lines:
        last_line = lines[-1]
        if last_line in ["REPO_OVERVIEW", "REFACTOR_FILE"]:
            return last_line
    return "MANUAL_REVIEW"


class ReasoningAgent(RoutedAgent):
    def __init__(self):
        super().__init__("reasoning_agent")
        # Use a more capable model for better planning
        self.router_model = VertexAI(
            model_name="gemini-2.0-flash-exp",
            temperature=0.2
        )
        
    @message_handler
    async def handle_reasoning(self, message: ReasoningRequest, ctx: MessageContext) -> str:
        """Agent handler for reasoning about repo questions."""
        user_question = message.question
        repo_data = st.session_state.get("repo_data", {})
        
        # --- Step 1: Enhanced Action Tagging (Routing) ---
        system_template_route = f"""You are an expert AI code assistant analyzing a user's request.

**Repository Files Available:**
{', '.join(list(repo_data.keys())[:20])}

**User Request:**
"{user_question}"

**Your Task:**
Determine if this is a:
1. REPO_OVERVIEW - User wants to understand/query the repository (questions like "what does", "explain", "show me")
2. REFACTOR_FILE - User wants to modify/add/change code (requests like "add", "implement", "refactor", "create")

**Decision Criteria:**
- If the request contains action verbs (add, create, implement, modify, refactor, update, fix), choose REFACTOR_FILE
- If the request is a question (what, how, why, explain, show), choose REPO_OVERVIEW
- Default to REFACTOR_FILE if the intent is to make changes

**Output Format:**
Line 1: Brief reasoning (1 sentence)
Line 2: Your chosen tag (REPO_OVERVIEW or REFACTOR_FILE)

Example:
The user wants to add a feedback feature, which requires code modification.
REFACTOR_FILE"""
        
        raw = self.router_model.predict(system_template_route.strip())
        
        # Normalize output
        explanation = ""
        if isinstance(raw, list):
            if len(raw) > 0:
                item = raw[0]
                explanation = item.get("text") or str(item) if isinstance(item, dict) else str(item)
        elif isinstance(raw, dict):
            explanation = raw.get("text") or str(raw)
        elif hasattr(raw, "text"):
            explanation = raw.text
        else:
            explanation = str(raw)
        
        chosen_tag = _parse_tag(explanation)
        
        # --- Step 2: Execution based on Tag ---
        response = ""
        st.session_state.current_commit_message = user_question
        st.session_state.reasoning_agent_output = user_question
        
        if chosen_tag == "REPO_OVERVIEW":
            from agents.sub_agents import OverviewAgent
            response = OverviewAgent().run(user_question)
            
        elif chosen_tag == "REFACTOR_FILE":
            if not repo_data:
                return "❌ Error: Repository data not loaded. Cannot perform refactoring."
            
            # --- Enhanced Planning and Commit Message Generation ---
            planning_prompt = f"""You are planning a code refactoring task.

**User Request:**
"{user_question}"

**Available Files:**
{chr(10).join([f'  - {f}' for f in list(repo_data.keys())[:20]])}

**Generate:**
1. **Commit Message:** A concise, professional Git commit message (max 50 characters)
   - Use imperative mood (e.g., "Add feature" not "Added feature")
   - Be specific but brief
   
2. **Implementation Plan:** A clear 2-4 bullet point plan of what changes will be made
   - Which files will be modified
   - What functionality will be added/changed
   - Any dependencies or prerequisites

**Output Format:**
COMMIT_MESSAGE: [Your commit message]

PLAN:
• [First step]
• [Second step]
• [Additional steps as needed]

**Example:**
COMMIT_MESSAGE: Add user feedback form to sales forecasting

PLAN:
• Modify pages/sales_forecasting.py to add feedback form after analysis results
• Add Streamlit form widgets (text input, rating, submit button)
• Implement form submission handler to store feedback
• Add visual confirmation message on submission

Now generate the commit message and plan:"""
            
            planning_raw = self.router_model.predict(planning_prompt)
            
            # Normalize planning output
            planning_output = ""
            if isinstance(planning_raw, list):
                if len(planning_raw) > 0:
                    item = planning_raw[0]
                    planning_output = item.get("text") or str(item) if isinstance(item, dict) else str(item)
            elif isinstance(planning_raw, dict):
                planning_output = planning_raw.get("text") or str(planning_raw)
            elif hasattr(planning_raw, "text"):
                planning_output = planning_raw.text
            else:
                planning_output = str(planning_raw)
            
            # Extract commit message with better regex
            commit_match = re.search(r"COMMIT_MESSAGE:\s*(.+?)(?:\n|$)", planning_output, re.IGNORECASE)
            if commit_match:
                commit_message = commit_match.group(1).strip()
                # Clean up any trailing periods or extra punctuation
                commit_message = commit_message.rstrip('.').strip()
            else:
                # Fallback: create commit message from user question
                commit_message = user_question[:50].strip()
                if not commit_message[0].isupper():
                    commit_message = commit_message.capitalize()
            
            st.session_state.current_commit_message = commit_message
            st.session_state.current_refactor_plan = planning_output
            
            # Display the plan
            st.session_state.messages.append({
                "role": "AI", 
                "content": f"**📋 Refactoring Plan:**\n\n{planning_output}"
            })
            
            # --- Execute Refactoring ---
            from agents.sub_agents import MultiFileRefactorAgent
            
            file_results = MultiFileRefactorAgent().run(user_question, repo_data)
            
            # Assemble response with proper markers
            response_parts = []
            refactor_agent = MultiFileRefactorAgent()
            
            for file_path, content in file_results.items():
                lang, _ = refactor_agent._get_file_info(file_path)
                marker_char = "#" if lang in ["python", "ruby", "shell"] else "//"
                response_parts.append(f"{marker_char} === {file_path} ===\n{content}")
            
            response = "\n\n".join(response_parts)
        
        else:
            response = f"Manual review needed. Router output:\n{explanation}"
        
        return response

# === Main Streamlit UI ===
def main():
    st.set_page_config(page_title="GitHub Repositories Refactorer")

    # --- Initialize session ---
    for key, default in {
        "messages": [],
        "repo_data": {},
        "GITHUB_REPOSITORY": None,
        "repo_data_fetched": False,
        "current_commit_message": "AI Refactoring based on user request",
        "reasoning_agent_output": "", # Stores the user request for intent check
        "vector_store": None
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
                <h1 style='color: #23a445; margin: 0;'>REPO AI Refactorer</h1>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --- Sidebar ---
    with st.sidebar:
        st.header("GitHub Integration")
        st.markdown(f"[Click here to install RepoAI]({INSTALL_URL}) :link:")

    github_url = st.text_input("Enter GitHub repository URL:")
    user_question = st.chat_input("Ask a question or request a refactoring:")

    # --- GitHub handling ---
    if github_url:
        owner, repo = extract_repo_info(github_url)
        if owner and repo:
            full_repo_name = f"{owner}/{repo}"
            st.session_state.GITHUB_REPOSITORY = full_repo_name

            if not st.session_state.repo_data_fetched or st.session_state.GITHUB_REPOSITORY != full_repo_name:
                st.session_state.repo_data_fetched = False # Reset if URL changes
                with st.spinner(f"Fetching {full_repo_name}..."):
                    try:
                        # Assuming read_all_repo_files fetches content and creates the vector store
                        repo_data, vector_store = read_all_repo_files(full_repo_name) 
                        
                        if repo_data:
                            st.session_state.repo_data = repo_data
                            st.session_state.vector_store = vector_store
                            st.session_state.repo_data_fetched = True
                            st.success(f"Loaded repository: {full_repo_name} with {len(repo_data)} files.")
                        else:
                            st.error("Failed to fetch content from GitHub.")
                    except Exception as e:
                        st.error(f"Failed to fetch content from GitHub: {e}")
            
        else:
            st.error("Invalid GitHub URL format.")

    # === Run the reasoning agent ===
    if user_question:
        
        # Check if refactoring is attempted without data
        if "refactor" in user_question.lower() and not st.session_state.repo_data_fetched:
            st.error("Please load a GitHub repository before requesting refactoring.")
            user_question = None
        
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

            # Clear temporary planning state before run
            st.session_state.pop("current_refactor_plan", None)

            with st.spinner("AI Agent is routing and executing the request..."):
                response = asyncio.run(run_reasoning_agent())
            
            # Only append the final output if it's not just the planning step (which is inserted earlier)
            if not "PLAN:" in response and response.strip():
                 st.session_state.messages.append({"role": "AI", "content": response})

    # === Display conversation ===

    refactor_agent_instance = MultiFileRefactorAgent()

    for i, message in enumerate(st.session_state.messages):
        role = message["role"]
        content = message.get("content", "")
        
        with st.chat_message(role):
            if role == "AI":
                
                # Check for the multi-file refactoring marker format
                if content.startswith("// ===") or content.startswith("# ==="):
                    
                    st.info(f"AI Refactoring Results:")
                    
                    # Use PushAgent's extraction logic for clean parsing
                    push_agent = PushAgent()
                    extracted_files = push_agent.extract_files_from_response(content) 

                    if extracted_files:
                        st.subheader(f"Files Modified ({len(extracted_files)})")
                        
                        full_refactored_code = ""
                        for file_name, file_content in extracted_files.items():
                            
                            # Determine language for syntax highlighting
                            lang, _ = refactor_agent_instance._get_file_info(file_name)
                            
                            st.markdown(f"**File:** `{file_name}`")
                            st.code(file_content, language=lang, line_numbers=True)
                            st.markdown("---")
                            
                            full_refactored_code += f"\n// File: {file_name}\n" + file_content
                        
                        # --- Validation and Diff ---
                        st.subheader("🔍 Code Review")
                        
                        # Initialize validation pipeline
                        validation_pipeline = ValidationPipeline()
                        
                        # Detect language helper
                        def detect_language(file_path: str) -> str:
                            lang, _ = refactor_agent_instance._get_file_info(file_path)
                            return lang
                        
                        # Validate and generate diffs
                        original_files = st.session_state.get("repo_data", {})
                        validation_results, diff_data = validation_pipeline.validate_and_diff(
                            original_files,
                            extracted_files,
                            detect_language
                        )
                        
                        # Display validation summary
                        all_valid = validation_pipeline.display_validation_summary(validation_results)
                        
                        # Display diffs
                        st.subheader("📊 Changes Preview")
                        for diff in diff_data:
                            validation_pipeline.diff_visualizer.render_diff_in_streamlit(diff)
                        
                        # --- Intent Fulfillment ---
                        if "reasoning_agent_output" in st.session_state and model:
                            st.subheader("🎯 Intent Fulfillment Check")
                            
                            score = check_intent_similarity(
                                st.session_state.reasoning_agent_output,
                                full_refactored_code,
                                embedding_model=model
                            )
                            st.metric(label="Intent Alignment Score", value=f"{score} %")

                        # --- Push to GitHub (ONLY IF VALIDATION PASSES) ---
                        commit_msg = st.session_state.get("current_commit_message", "AI Refactoring")
                        st.write(f"Proposed Commit: **{commit_msg}**")
                        
                        # FIXED: Create original_paths mapping correctly
                        # Map filename to full path from repo_data
                        original_paths = {}
                        for file_name in extracted_files.keys():
                            # Try to find the original path in repo_data
                            for original_path in original_files.keys():
                                if original_path.endswith(file_name) or original_path == file_name:
                                    original_paths[file_name] = original_path
                                    break
                        
                        # CHANGED: Only show button if validation passed
                        if all_valid:
                            if st.button("✅ Approve and Create PR", key=f"push_{i}"):
                                
                                if not st.session_state.GITHUB_REPOSITORY:
                                    error_handler = ErrorHandler()
                                    error = error_handler.handle_github_auth_error(
                                        Exception("Repository URL not set")
                                    )
                                    error_handler.display_error(error)
                                else:
                                    with st.spinner(f"Creating PR: {commit_msg}"):
                                        try:
                                            current_push_agent = PushAgent(
                                                github_app_id=github_token,
                                                github_private_key=github_private_file
                                            )
                                            
                                            # FIXED: Pass original_paths to maintain file structure
                                            result = current_push_agent.run(
                                                ai_response=content,
                                                repo_name=st.session_state.GITHUB_REPOSITORY,
                                                commit_message=commit_msg,
                                                original_paths=original_paths  # Pass the mapping
                                            )
                                            st.success(result)
                                            
                                        except Exception as e:
                                            error_handler = ErrorHandler()
                                            # Try to categorize the error
                                            if "rate limit" in str(e).lower():
                                                error = error_handler.handle_rate_limit_error(e)
                                            elif "auth" in str(e).lower():
                                                error = error_handler.handle_github_auth_error(e)
                                            else:
                                                # Generic error
                                                from utils.error_handlers import UserFriendlyError, ErrorCategory
                                                error = UserFriendlyError(
                                                    category=ErrorCategory.GITHUB_API,
                                                    title="GitHub Operation Failed",
                                                    message="Failed to create PR",
                                                    technical_details=str(e),
                                                    suggested_actions=[
                                                        "Check technical details below",
                                                        "Verify GitHub App permissions",
                                                        "Ensure repository name is correct",
                                                        "Try again in a few moments"
                                                    ],
                                                    can_retry=True
                                                )
                                            error_handler.display_error(error)
                        else:
                            st.warning("⚠️ Please fix validation errors before pushing to GitHub")
                            st.info("You can manually review and fix the code, then retry")
                    
                    else:
                        st.warning("Could not extract refactored files from AI output. Check logs.")
                        st.code(content)

                else:
                    # Display general text responses (QA, planning, errors)
                    st.markdown(content.replace("\n", "  \n"))

            else:
                # Display user messages normally
                st.markdown(f"**You:**  {content.replace('\n', '  \n')}")


if __name__ == "__main__":
    main()