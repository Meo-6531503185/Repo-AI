
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
from testing import * 
from push import * # Assuming push utilities are here
from agents.sub_agents import (
    OverviewAgent,
    MultiFileRefactorAgent,
)
from agents.push_agent import PushAgent # Assuming PushAgent is in agents/sub_agents.py or a separate file

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


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

# Helper to parse the action tag from LLM output
def _parse_tag(explanation_output: str) -> str:
    """Robustly extracts the action tag from the end of the LLM response."""
    lines = [line.strip() for line in explanation_output.strip().splitlines() if line.strip()]
    if lines:
        last_line = lines[-1]
        if last_line in ["REPO_OVERVIEW", "REFACTOR_FILE"]:
             return last_line
    return "MANUAL_REVIEW"


# === Reasoning Agent (Router + Planner) ===
class ReasoningAgent(RoutedAgent):
    def __init__(self):
        super().__init__("reasoning_agent")
        # Use a lightweight model for initial routing and planning
        self.router_model = VertexAI(model_name="gemini-2.5-flash")
        
    @message_handler
    async def handle_reasoning(self, message: ReasoningRequest, ctx: MessageContext) -> str:
        """Agent handler for reasoning about repo questions."""
        user_question = message.question
        repo_data = st.session_state.get("repo_data", {})
        
        # --- Step 1: Action Tagging (Routing) ---
        system_template_route = f"""
You are an expert AI router. Analyze the user's request.
Determine the primary action: Does this require a repository overview (reading/QA) or code refactoring (writing/changing)?

TAGS:
REPO_OVERVIEW, REFACTOR_FILE.

OUTPUT FORMAT:
1. Short reasoning (1-2 sentences).
2. The chosen tag on a new line (e.g., REFACTOR_FILE).

User Request: {user_question}
"""
        
        raw = self.router_model.predict(system_template_route.strip())

        # --- Normalize Gemini output to a plain string ---
        explanation = ""
        if isinstance(raw, list):
            # Handle list output - extract the first item
            if len(raw) > 0:
                item = raw[0]
                if isinstance(item, dict):
                    explanation = item.get("text") or item.get("output_text") or str(item)
                elif hasattr(item, "text"):
                    explanation = item.text
                else:
                    explanation = str(item)
            else:
                explanation = ""
        elif isinstance(raw, dict):
            explanation = raw.get("text") or raw.get("output_text") or str(raw)
        elif hasattr(raw, "text"):
            explanation = raw.text
        else:
            explanation = str(raw)

        chosen_tag = _parse_tag(explanation)


        # --- Step 2: Execution based on Tag ---
        response = ""
        st.session_state.current_commit_message = user_question # Default commit message
        st.session_state.reasoning_agent_output = user_question # Store intent

        if chosen_tag == "REPO_OVERVIEW":
            # Pass the request to the Overview agent (presumably handles QA via vector store)
            response = OverviewAgent().run(user_question)
            
        elif chosen_tag == "REFACTOR_FILE":
            
            if not repo_data:
                return "❌ Error: Repository data not loaded. Cannot perform refactoring."

            # --- Step 2b: Planning and Commit Message Generation ---
            planning_prompt = f"""
            The user wants to perform a multi-file refactoring. 
            
            First, generate a concise, professional commit message (max 12 words) that summarizes the core change.
            
            Second, provide a brief (1-3 line) plan of action detailing the approach (e.g., "Refactor class X, then update imports in file Y").
            
            User Request: {user_question}
            Files available: {list(repo_data.keys())}

            OUTPUT FORMAT:
            COMMIT_MESSAGE: [Concise summary]
            PLAN: [Brief plan of action]
            """
            
            planning_raw = self.router_model.predict(planning_prompt)
            
            # --- Normalize planning output the same way ---
            planning_output = ""
            if isinstance(planning_raw, list):
                if len(planning_raw) > 0:
                    item = planning_raw[0]
                    if isinstance(item, dict):
                        planning_output = item.get("text") or item.get("output_text") or str(item)
                    elif hasattr(item, "text"):
                        planning_output = item.text
                    else:
                        planning_output = str(item)
            elif isinstance(planning_raw, dict):
                planning_output = planning_raw.get("text") or planning_raw.get("output_text") or str(planning_raw)
            elif hasattr(planning_raw, "text"):
                planning_output = planning_raw.text
            else:
                planning_output = str(planning_raw)
            
            # Robust extraction of the commit message
            commit_match = re.search(r"COMMIT_MESSAGE:\s*(.*)", planning_output, re.IGNORECASE)
            commit_message = commit_match.group(1).strip() if commit_match else f"Automated refactoring: {user_question[:50]}..."
            
            st.session_state.current_commit_message = commit_message
            st.session_state.current_refactor_plan = planning_output # Store plan for display
            
            # --- Step 2c: Execution ---
            
            # Display the plan immediately
            st.session_state.messages.append({"role": "AI", "content": f"**Refactoring Plan:**\n\n{planning_output}"})
            
            file_results = MultiFileRefactorAgent().run(user_question, repo_data)
            
            # Reassemble the file results using the required marker format for easy PushAgent parsing
            response_parts = []
            
            # Use the MultiFileRefactorAgent's helper to ensure correct markers
            refactor_agent_instance = MultiFileRefactorAgent()
            
            for f, content in file_results.items():
                # Use a specific helper to ensure consistency
                lang, _ = refactor_agent_instance._get_file_info(f)
                marker_char = "#" if lang in ["python", "ruby", "shell"] else "//"
                response_parts.append(f"{marker_char} === {f} ===\n{content}")
            
            response = "\n\n".join(response_parts)

        else:
            response = f"Manual review is needed. Router output was:\n{explanation}"

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
    refactor_agent_instance = MultiFileRefactorAgent() # Instance needed for file info helper
    
    for i, message in enumerate(st.session_state.messages):
        role = message["role"]
        content = message.get("content", "")
        
        with st.chat_message(role):
            if role == "AI":
                
                # Check for the multi-file refactoring marker format (// === file === CODE)
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
                        
                        # --- Intent Fulfillment ---
                        if "reasoning_agent_output" in st.session_state and model:
                            st.subheader("🎯 Intent Fulfillment Check")
                            
                            score = check_intent_similarity(
                                st.session_state.reasoning_agent_output,
                                full_refactored_code,
                                embedding_model=model
                            )
                            st.metric(label="Intent Alignment Score", value=f"{score} %")

                        # --- Push to GitHub ---
                        commit_msg = st.session_state.get("current_commit_message", "AI Refactoring (No message generated)")
                        st.write(f"Proposed Commit Message: **{commit_msg}**")
                        
                        if st.button("🚀 Create Pull Request on GitHub", key=f"push_{i}"):
                            
                            if not st.session_state.GITHUB_REPOSITORY:
                                st.error("Cannot push: Repository URL not set.")
                            else:
                                with st.spinner(f"Creating branch and PR: {commit_msg}"):
                                    # Instantiate the push agent with credentials
                                    current_push_agent = PushAgent(
                                        github_app_id=github_token,
                                        github_private_key=github_private_file
                                    )
                                    result = current_push_agent.run(
                                        ai_response=content, # Pass the raw content for re-extraction
                                        repo_name=st.session_state.GITHUB_REPOSITORY,
                                        commit_message=commit_msg 
                                    )
                                st.success(result)
                    
                    else:
                        st.warning("Could not extract refactored files from AI output. Check logs.")
                        st.code(content) # Show raw content if parsing failed

                else:
                    # Display general text responses (QA, planning, errors)
                    st.markdown(content.replace("\n", "  \n"))

            else:
                # 🧍 Display user messages normally
                st.markdown(f"**You:**  {content.replace('\n', '  \n')}")


if __name__ == "__main__":
    main()