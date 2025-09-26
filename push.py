import re
from githubTest import *
from testing import *

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

def push_refactored_code_to_github(refined_code,test_case, github_url):
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
    wrapper.create_branch("AI_code_branch")
    # Define file details
    file_path = "src/main/java/" + extract_class_name(refined_code) + ".java"
    test_path = "src/test/java/" + extract_class_name(test_case) + ".java"
    # commit_message = "Added refactored Java code"
    refactored_code = refined_code
    file_query = f"{file_path}\n\n{refactored_code}"
    test_query = f"{test_path}\n\n{test_case}"
    # Push new file
    wrapper.create_file(file_query)
    wrapper.create_file(test_query)
    return "Refactored code has been pushed to the repository."