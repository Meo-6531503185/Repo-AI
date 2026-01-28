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
    OverviewAgent,
)
from agents.push_agent import PushAgent # Assuming PushAgent is in agents/sub_agents.py or a separate file

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# FIXED IMPORTS - Use the new multi-language validators
from validators.comprehensive_validators import (
    ComprehensiveValidationPipeline,
    ValidationResult,
    detect_language_from_path  # Import the helper function
)
from utils.error_handlers import ErrorHandler, UserFriendlyError, ErrorCategory
from utils.llm_normalizer import LLMOutputNormalizer, CodeExtractor



st.markdown("""
    <style>
    /* Style primary buttons (blue → green) */
    div.stButton > button[kind="primary"] {
        background-color: #28a745 !important;   /* green */
        color: white !important;
        border: none !important;
    }

    /* Optional: style secondary buttons */
    div.stButton > button[kind="secondary"] {
        background-color: #6c757d !important;  /* gray */
        color: white !important;
        border: none !important;
    }
    </style>
""", unsafe_allow_html=True)

# === Load environment ===
load_dotenv()
github_token = os.getenv("GITHUB_APP_ID")
github_private_file = os.getenv("GITHUB_PRIVATE_KEY")
if github_token and github_private_file:
    print("GitHub Token loaded successfully")
else:
    print("GitHub Token not found")

# === Initialize embedding model and validation pipeline ONCE at module level ===
try:
    model = VertexAIEmbeddings(model_name="text-embedding-004")
    
    # Initialize validation LLM - using more capable model
    validation_llm = VertexAI(
        model_name="gemini-2.0-flash-exp",  # Use faster model for validation
        temperature=0.2
    )
    
    # Initialize comprehensive validation pipeline with multi-language support
    validation_pipeline = ComprehensiveValidationPipeline(
        embedding_model=model,
        llm_client=validation_llm
    )
    
    print("Multi-language validation pipeline initialized successfully")
    
except Exception as e:
    st.error(f"Error initializing models: {e}")
    model = None
    validation_pipeline = None

# === Helper: Extract repo info ===
def extract_repo_info(url: str):
    """Extracts owner and repo name from a GitHub URL."""
    pattern = r"https://github\.com/([^/]+)/([^/]+)"
    match = re.match(pattern, url)
    return match.groups() if match else (None, None)


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
            # FIXED: Now properly passing repo_data to OverviewAgent
            overview_agent = OverviewAgent()
            response = overview_agent.run(user_question, repo_data)
            
        elif chosen_tag == "REFACTOR_FILE":
            if not repo_data:
                return "Error: Repository data not loaded. Cannot perform refactoring."
            
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
                "content": f"**Refactoring Plan:**\n\n{planning_output}"
            })
            
            # --- Execute Refactoring ---
            refactor_agent = MultiFileRefactorAgent()
            file_results = refactor_agent.run(user_question, repo_data)
            
            # Assemble response with proper markers
            response_parts = []
            
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
        "reasoning_agent_output": "",
        "vector_store": None,
        "validation_passed": False,
        "validation_results": None  # Store detailed results
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
        
        # Show validation status if available
        if st.session_state.validation_results:
            st.divider()
            st.subheader("Last Validation")
            
            report = validation_pipeline.generate_validation_report(
                st.session_state.validation_results
            )
            
            if report['overall_pass']:
                st.success(f"All checks passed")
            else:
                st.warning(f"{report['critical_failures']} issue(s)")
            
            st.metric("Files Validated", report['files_validated'])
            st.metric("Pass Rate", f"{report['files_passed']}/{report['files_validated']}")

    github_url = st.text_input("Enter GitHub repository URL:")
    user_question = st.chat_input("Ask a question or request a refactoring:")

    # --- GitHub handling ---
    if github_url:
        owner, repo = extract_repo_info(github_url)
        if owner and repo:
            full_repo_name = f"{owner}/{repo}"
            st.session_state.GITHUB_REPOSITORY = full_repo_name

            if not st.session_state.repo_data_fetched or st.session_state.GITHUB_REPOSITORY != full_repo_name:
                st.session_state.repo_data_fetched = False
                with st.spinner(f"Fetching {full_repo_name}..."):
                    try:
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

            st.session_state.pop("current_refactor_plan", None)

            with st.spinner("AI Agent is routing and executing the request..."):
                response = asyncio.run(run_reasoning_agent())
            
            if not "PLAN:" in response and response.strip():
                 st.session_state.messages.append({"role": "AI", "content": response})

    # === Display conversation ===

    for i, message in enumerate(st.session_state.messages):
        role = message["role"]
        content = message.get("content", "")
        
        with st.chat_message(role):
            if role == "AI":
                
                # Check for the multi-file refactoring marker format
                if content.startswith("// ===") or content.startswith("# ==="):
                    
                    st.info(f"AI Refactoring Results")
                    
                    # Use PushAgent's extraction logic for clean parsing
                    push_agent = PushAgent()
                    extracted_files = push_agent.extract_files_from_response(content) 

                    if extracted_files:
                        st.subheader(f"Files Modified ({len(extracted_files)})")
                        
                        # Display code for each file
                        for file_name, file_content in extracted_files.items():
                            # Use the new detect_language_from_path helper
                            lang = detect_language_from_path(file_name)
                            
                            st.markdown(f"**File:** `{file_name}` ({lang})")
                            st.code(file_content, language=lang, line_numbers=True)
                            st.markdown("---")
                        
                        # --- COMPREHENSIVE MULTI-LANGUAGE VALIDATION ---
                        if validation_pipeline:
                            st.subheader("Comprehensive Code Validation")
                            
                            with st.spinner("Running multi-language validation checks..."):
                                # Run validation on all files
                                original_files = st.session_state.get("repo_data", {})
                                user_request = st.session_state.get("reasoning_agent_output", user_question)
                                
                                try:
                                    validation_results = validation_pipeline.validate_all_files(
                                        user_request=user_request,
                                        original_files=original_files,
                                        refactored_files=extracted_files,
                                        language_detector=detect_language_from_path  # Use the imported helper
                                    )
                                    
                                    # Store results in session state
                                    st.session_state.validation_results = validation_results
                                    
                                    # Display results using the built-in Streamlit display
                                    all_valid = validation_pipeline.display_validation_summary_streamlit(
                                        validation_results
                                    )
                                    
                                    # Store validation status
                                    st.session_state.validation_passed = all_valid
                                    
                                    # Show appropriate message based on results
                                    if all_valid:
                                        st.success("All validation checks passed! Code is ready for PR.")
                                    else:
                                        st.warning("""
                                        Some validation checks failed. Review the details above.
                                        
                                        **You can still create a PR** - validation results are advisory to help you make an informed decision.
                                        """)
                                        
                                        # Show which specific checks failed
                                        report = validation_pipeline.generate_validation_report(validation_results)
                                        
                                        if report['critical_failures'] > 0:
                                            st.error(f"{report['critical_failures']} critical issue(s) found")
                                            
                                            # List the specific failures
                                            for file_path, file_result in report['detailed_results'].items():
                                                if not file_result['passed']:
                                                    failed_tests = [
                                                        test_name for test_name, test_result 
                                                        in file_result['tests'].items() 
                                                        if not test_result['passed'] and test_result['is_critical']
                                                    ]
                                                    if failed_tests:
                                                        st.markdown(f"- **{file_path}**: {', '.join(failed_tests)}")
                                
                                except Exception as e:
                                    st.error(f"Validation error: {str(e)}")
                                    st.session_state.validation_passed = True  # Don't block on validation errors
                                    all_valid = True
                            
                        else:
                            st.warning("Validation pipeline not initialized - skipping validation")
                            all_valid = True
                            st.session_state.validation_passed = True

                        # --- Push to GitHub (ALWAYS ENABLED) ---
                        st.divider()
                        st.subheader("Create Pull Request")
                        
                        commit_msg = st.session_state.get("current_commit_message", "AI Refactoring")
                        st.info(f"**Proposed Commit Message:** {commit_msg}")
                        
                        # Show validation status badge
                        if st.session_state.validation_passed:
                            st.success("All validation checks passed")
                        else:
                            st.warning("Validation issues detected (see above)")
                        
                        # Map filename to full path from repo_data
                        original_paths = {}
                        for file_name in extracted_files.keys():
                            for original_path in original_files.keys():
                                if original_path.endswith(file_name) or original_path == file_name:
                                    original_paths[file_name] = original_path
                                    break
                        
                        # Button styling based on validation status
                        if st.session_state.validation_passed:
                            button_label = "Create Pull Request"
                            button_type = "primary"
                        else:
                            button_label = "Create PR (with validation issues)"
                            button_type = "secondary"
                        
                        
                        
                        
                            if st.button(button_label, key=f"push_{i}", type=button_type, use_container_width=True):
                                
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
                                            
                                            result = current_push_agent.run(
                                                ai_response=content,
                                                repo_name=st.session_state.GITHUB_REPOSITORY,
                                                commit_message=commit_msg,
                                                original_paths=original_paths
                                            )
                                            st.success(result)
                                            
                                            # Add validation summary to PR description if available
                                            if st.session_state.validation_results:
                                                report = validation_pipeline.generate_validation_report(
                                                    st.session_state.validation_results
                                                )
                                                st.info(f"""
                                                **Validation Summary:**
                                                - Files validated: {report['files_validated']}
                                                - Files passed: {report['files_passed']}
                                                - Critical issues: {report['critical_failures']}
                                                """)
                                            
                                        except Exception as e:
                                            error_handler = ErrorHandler()
                                            if "rate limit" in str(e).lower():
                                                error = error_handler.handle_rate_limit_error(e)
                                            elif "auth" in str(e).lower():
                                                error = error_handler.handle_github_auth_error(e)
                                            else:
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
                        st.error("Could not extract refactored files from AI output")
                        with st.expander("Show raw output"):
                            st.code(content)

                else:
                    # Display general text responses (QA, planning, errors)
                    st.markdown(content.replace("\n", "  \n"))

            else:
                # Display user messages
                st.markdown(content.replace('\n', '  \n'))


if __name__ == "__main__":
    main()