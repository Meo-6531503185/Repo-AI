import re
import streamlit as st
from githubTest import GitHubAPIWrapper


class PushAgent:
    """
    Independent agent that pushes AI-generated refactored code to GitHub.
    Works even if the AI output doesn't follow strict markers.
    """

    def __init__(self, github_app_id=None, github_private_key=None):
        self.github_app_id = github_app_id
        self.github_private_key = github_private_key

    def extract_files_from_response(self, ai_response: str):
        """
        Extract multiple or single files from the AI response.
        Supports formats like:
            // === FileName.java ===
            <file content>
        or plain code blocks like:
            public class MyClass { ... }
        Returns: dict[str, str]
        """
        files = {}

        # 1️⃣ Try multi-file marker format first
        pattern = r"// === (.*?) ===\n(.*?)(?=(?:\n// === |\Z))"
        matches = re.findall(pattern, ai_response, re.DOTALL)
        if matches:
            for filename, content in matches:
                files[filename.strip()] = content.strip()
            return files

        # 2️⃣ Try to extract a single Java class if markers not found
        class_match = re.search(
            r"(public\s+class\s+([A-Za-z0-9_]+).*?})", ai_response, re.DOTALL
        )
        if class_match:
            class_code = class_match.group(1).strip()
            class_name = class_match.group(2).strip()
            files[f"{class_name}.java"] = class_code
            return files

        # 3️⃣ If no class found, just save all content as RefactoredOutput.java
        files["RefactoredOutput.java"] = ai_response.strip()
        return files

    def push_to_github(self, repo_name: str, files: dict):
        """
        Push all extracted files to the GitHub repository.
        Each file is committed under `src/main/java` or `src/test/java` based on name.
        """
        if not repo_name:
            st.error("❌ GitHub repository name is missing.")
            return

        # Initialize GitHub wrapper
        wrapper = GitHubAPIWrapper(
            github_repository=repo_name,
            github_app_id=self.github_app_id,
            github_app_private_key=self.github_private_key,
        )

        # Create a new branch for the AI changes
        wrapper.create_branch("AI_code_branch")

        pushed_files = []
        for file_name, code in files.items():
            if "test" in file_name.lower():
                file_path = f"src/test/java/{file_name}"
            else:
                file_path = f"src/main/java/{file_name}"

            wrapper.create_file(f"{file_path}\n\n{code}")
            pushed_files.append(file_path)

        return "✅ Successfully pushed files:\n" + "\n".join([f"- {f}" for f in pushed_files])

    def run(self, ai_response: str, repo_name: str):
        """
        Main entry point for the PushAgent.
        Extracts files and pushes them to GitHub.
        """
        files = self.extract_files_from_response(ai_response)
        if not files:
            st.warning("⚠️ No recognizable Java files found in the AI response.")
            return "No refactored files detected."

        result = self.push_to_github(repo_name, files)
        return result
