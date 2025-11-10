import re
import streamlit as st
from githubTest import GitHubAPIWrapper


# class PushAgent:
#     """
#     Independent agent that pushes AI-generated refactored code to GitHub.
#     Works across multiple programming languages (Java, Python, JS, etc.).
#     """

#     def __init__(self, github_app_id=None, github_private_key=None):
#         self.github_app_id = github_app_id
#         self.github_private_key = github_private_key

#     def extract_files_from_response(self, ai_response: str):
#         """
#         Extracts one or more files from AI response text.

#         Supports:
#         - Markers like:
#               // === FileName.java ===
#               # === FileName.py ===
#         - Plain code without markers (auto-detect language via syntax).
#         Returns: dict[str, str]
#         """
#         files = {}

#         # 1️⃣ Try multi-language file marker format
#         pattern = r"(?:\/\/|#)\s*===\s*(.*?)\s*===\n(.*?)(?=(?:\n(?:\/\/|#)\s*===|\Z))"
#         matches = re.findall(pattern, ai_response, re.DOTALL)
#         if matches:
#             for filename, content in matches:
#                 files[filename.strip()] = content.strip()
#             return files

#         # 2️⃣ Try to infer from file content (language-specific heuristics)
#         lang_patterns = {
#             "java": (r"public\s+class\s+([A-Za-z0-9_]+)", ".java"),
#             "python": (r"def\s+\w+\(", ".py"),
#             "javascript": (r"function\s+\w+\(|const\s+\w+\s*=\s*\(", ".js"),
#             "typescript": (r"export\s+class\s+([A-Za-z0-9_]+)", ".ts"),
#             "cpp": (r"#include\s*<.*>", ".cpp"),
#             "c": (r"#include\s*<.*>", ".c"),
#             "go": (r"package\s+\w+", ".go"),
#             "php": (r"<\?php", ".php"),
#             "ruby": (r"def\s+\w+", ".rb"),
#             "swift": (r"import\s+SwiftUI", ".swift"),
#             "kotlin": (r"fun\s+\w+", ".kt"),
#         }

#         for lang, (pattern, ext) in lang_patterns.items():
#             match = re.search(pattern, ai_response)
#             if match:
#                 name = match.group(1) if match.groups() else f"RefactoredOutput{ext}"
#                 files[f"{name}{ext}" if not name.endswith(ext) else name] = ai_response.strip()
#                 return files

#         # 3️⃣ Fallback: unknown content type
#         files["RefactoredOutput.txt"] = ai_response.strip()
#         return files

#     def detect_language_folder(self, file_name: str) -> str:
#         """Map file extensions to language-specific folders."""
#         extension_map = {
#             ".java": "java",
#             ".py": "python",
#             ".js": "javascript",
#             ".ts": "typescript",
#             ".cpp": "cpp",
#             ".c": "c",
#             ".go": "go",
#             ".php": "php",
#             ".rb": "ruby",
#             ".swift": "swift",
#             ".kt": "kotlin",
#         }

#         for ext, folder in extension_map.items():
#             if file_name.endswith(ext):
#                 return folder
#         return "other"

#     def push_to_github(self, repo_name: str, files: dict):
#         """
#         Push all extracted files to the GitHub repository.
#         Creates language-based folder structures automatically.
#         """
#         if not repo_name:
#             st.error("❌ GitHub repository name is missing.")
#             return

#         wrapper = GitHubAPIWrapper(
#             github_repository=repo_name,
#             github_app_id=self.github_app_id,
#             github_app_private_key=self.github_private_key,
#         )

#         wrapper.create_branch("AI_code_branch")

#         pushed_files = []
#         for file_name, code in files.items():
#             lang_folder = self.detect_language_folder(file_name)

#             # Distinguish between test and main code
#             if "test" in file_name.lower():
import uuid
class PushAgent:
    
    def __init__(self, github_app_id=None, github_private_key=None):
        self.github_app_id = github_app_id
        self.github_private_key = github_private_key

    def extract_files_from_response(self, ai_response: str):
        """
        Improved extraction using strict AI output markers, ensuring robustness.
        """
        files = {}
        
        # Use a single, powerful regex matching both // === and # === formats
        # and capture the content until the next marker or end of string.
        # This relies on the LLM adhering to the strict output format defined in MultiFileRefactorAgent.
        pattern = r"(?P<marker>\/\/|#)\s*===\s*(?P<filename>.*?)\s*===\s*\n(?P<content>.*?)(?=\n(?:\/\/|#)\s*===|\Z)"
        
        matches = re.finditer(pattern, ai_response, re.DOTALL)
        
        if matches:
            for match in matches:
                filename = match.group("filename").strip()
                content = match.group("content").strip()
                if filename and content:
                    files[filename] = content
            return files

        # 2️⃣ Fallback: Try to infer if only one file was returned (no markers found)
        # We rely on the MultiFileRefactorAgent to return *only* code.
        if not files:
            # If the response is primarily code-like and contains no separators, 
            # assume it's a single file output. We can't know the name, so we use a placeholder.
            if len(ai_response) > 100 and any(kw in ai_response for kw in ["def ", "class ", "public class", "function "]):
                 # Attempt to extract the original file name from session state if available, 
                 # or use a generic name. (This requires better agent state passing, but we'll use a safe fallback.)
                files["RefactoredOutput.code"] = ai_response.strip()
            
        return files
    
    # ... (rest of detect_language_folder and push_to_github/run remain largely the same)
    def detect_language_folder(self, file_name: str) -> str:
        """Map file extensions to language-specific folders."""
        extension_map = {
            ".java": "java",
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".cpp": "cpp",
            ".c": "c",
            ".go": "go",
            ".php": "php",
            ".rb": "ruby",
            ".swift": "swift",
            ".kt": "kotlin",
        }

        for ext, folder in extension_map.items():
            if file_name.endswith(ext):
                return folder
        return "other"
    def run(self, ai_response: str, repo_name: str, commit_message: str = "AI Refactoring based on user request"):
        """Main entry point for the PushAgent."""
        files = self.extract_files_from_response(ai_response)
        if not files:
            st.warning("⚠️ No recognizable source files found in the AI response.")
            return "No refactored files detected."

        result = self.push_to_github(repo_name, files, commit_message)
        return result

    def push_to_github(self, repo_name: str, files: dict, commit_message: str):
        """
        Push all extracted files to the GitHub repository.
        (Updated to accept a commit message)
        """
        # ... (Wrapper initialization remains the same)
        
        if not repo_name:
            st.error("❌ GitHub repository name is missing.")
            return
        
        wrapper = GitHubAPIWrapper(
            github_repository=repo_name,
            github_app_id=self.github_app_id,
            github_app_private_key=self.github_private_key,
        )

        branch_name = f"ai-refactor-{uuid.uuid4().hex[:6]}"
        wrapper.create_branch(branch_name)

        pushed_files = []
        # ... (file path determination remains the same)
        
        # Added loop for creating files with the branch
        for file_name, code in files.items():
            lang_folder = self.detect_language_folder(file_name)
            
            if "test" in file_name.lower() or "tests" in file_name.lower():
                file_path = f"src/test/{lang_folder}/{file_name}"
            else:
                file_path = f"src/main/{lang_folder}/{file_name}"
            
            wrapper.create_file(f"{file_path}\n\n{code}")
            pushed_files.append(file_path)
            
        return "✅ Successfully pushed files:\n" + "\n".join([f"- {f}" for f in pushed_files])#                 file_path = f"src/test/{lang_folder}/{file_name}"
#             else:
#                 file_path = f"src/main/{lang_folder}/{file_name}"

#             wrapper.create_file(f"{file_path}\n\n{code}")
#             pushed_files.append(file_path)

#         return "✅ Successfully pushed files:\n" + "\n".join([f"- {f}" for f in pushed_files])

#     def run(self, ai_response: str, repo_name: str):
#         """Main entry point for the PushAgent."""
#         files = self.extract_files_from_response(ai_response)
#         if not files:
#             st.warning("⚠️ No recognizable source files found in the AI response.")
#             return "No refactored files detected."

#         result = self.push_to_github(repo_name, files)
#         return result




        # # Assuming GitHubAPIWrapper handles the final commit on the branch
        # pr_url = wrapper.create_pull_request(
        #     title=f"AI Refactor: {commit_message[:50]}...",
        #     body=f"Automated refactoring based on the user request:\n\n{commit_message}",
        #     base_branch="main",
        #     head_branch=branch_name
        # )
        
        # return f"✅ Successfully pushed {len(pushed_files)} files to branch `{branch_name}`. Pull Request created: {pr_url}"