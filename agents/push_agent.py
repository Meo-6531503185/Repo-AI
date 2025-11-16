import re
import streamlit as st
from githubTest import GitHubAPIWrapper


import uuid
from typing import Dict, Optional


class PushAgent:
    """Enhanced agent for pushing refactored code to GitHub with PR creation."""
    
    def __init__(self, github_app_id=None, github_private_key=None):
        self.github_app_id = github_app_id
        self.github_private_key = github_private_key
    
    def extract_files_from_response(self, ai_response: str) -> Dict[str, str]:
        """Extract files from AI response with improved parsing."""
        files = {}
        
        # Match both // === and # === markers
        pattern = r"(?P<marker>\/\/|#)\s*===\s*(?P<filename>.*?)\s*===\s*\n(?P<content>.*?)(?=\n(?:\/\/|#)\s*===|\Z)"
        matches = re.finditer(pattern, ai_response, re.DOTALL)
        
        for match in matches:
            filename = match.group("filename").strip()
            content = match.group("content").strip()
            if filename and content:
                files[filename] = content
        
        # Fallback: single file without markers
        if not files and len(ai_response) > 100:
            if any(kw in ai_response for kw in ["def ", "class ", "public class", "function "]):
                files["RefactoredOutput.code"] = ai_response.strip()
        
        return files
    
    def detect_language_folder(self, file_name: str) -> str:
        """Map file extensions to language folders."""
        extension_map = {
            ".java": "java", ".py": "python", ".js": "javascript",
            ".ts": "typescript", ".cpp": "cpp", ".c": "c",
            ".go": "go", ".php": "php", ".rb": "ruby",
            ".swift": "swift", ".kt": "kotlin",
        }
        
        for ext, folder in extension_map.items():
            if file_name.endswith(ext):
                return folder
        return "other"
    
    def determine_file_path(self, file_name: str, original_path: Optional[str] = None) -> str:
        """
        Determine where to place the file, preserving original structure when possible.
        
        Args:
            file_name: Name of the file
            original_path: Original path from repo_data (if available)
        """
        # If we have the original path, use it
        if original_path:
            return original_path
        
        # Otherwise, use language-based folder structure
        lang_folder = self.detect_language_folder(file_name)
        
        if "test" in file_name.lower():
            return f"src/test/{lang_folder}/{file_name}"
        else:
            return f"src/main/{lang_folder}/{file_name}"
    
    def push_to_github(
        self, 
        repo_name: str, 
        files: Dict[str, str], 
        commit_message: str,
        original_paths: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Push files to GitHub and create a Pull Request.
        
        Args:
            repo_name: Full repository name (owner/repo)
            files: Dict of filename -> content
            commit_message: Commit message
            original_paths: Optional dict mapping filenames to original paths
        """
        if not repo_name:
            raise ValueError("GitHub repository name is missing")
        
        try:
            wrapper = GitHubAPIWrapper(
                github_repository=repo_name,
                github_app_id=self.github_app_id,
                github_app_private_key=self.github_private_key,
            )
            
            # Create unique branch name
            branch_name = f"ai-refactor-{uuid.uuid4().hex[:8]}"
            branch_result = wrapper.create_branch(branch_name)
            
            if "Error" in branch_result or "Unable" in branch_result:
                raise Exception(f"Branch creation failed: {branch_result}")
            
            st.success(f"✅ Created branch: {branch_name}")
            
            # Push files
            pushed_files = []
            failed_files = []
            
            for file_name, code in files.items():
                try:
                    # Determine correct path
                    if original_paths and file_name in original_paths:
                        file_path = original_paths[file_name]
                    else:
                        file_path = self.determine_file_path(file_name)
                    
                    # Check if file exists - update or create
                    existing_content = wrapper.read_file(file_path)
                    
                    if "File not found" not in existing_content:
                        # File exists - update it
                        update_query = f"{file_path}\nOLD <<<<\n{existing_content}\n>>>> OLD\nNEW <<<<\n{code}\n>>>> NEW"
                        result = wrapper.update_file(update_query)
                    else:
                        # File doesn't exist - create it
                        result = wrapper.create_file(f"{file_path}\n\n{code}")
                    
                    if "Error" in result or "Unable" in result:
                        failed_files.append((file_path, result))
                    else:
                        pushed_files.append(file_path)
                        st.success(f"✅ Pushed: {file_path}")
                        
                except Exception as e:
                    failed_files.append((file_name, str(e)))
            
            # Create Pull Request
            if pushed_files:
                try:
                    pr_result = self._create_pull_request(
                        wrapper,
                        branch_name,
                        commit_message,
                        pushed_files
                    )
                    
                    result_message = f"✅ Successfully pushed {len(pushed_files)} file(s) to branch `{branch_name}`\n\n"
                    result_message += f"📝 Files modified:\n" + "\n".join([f"  - {f}" for f in pushed_files])
                    
                    if pr_result:
                        result_message += f"\n\n{pr_result}"
                    
                    if failed_files:
                        result_message += f"\n\n⚠️ Failed files ({len(failed_files)}):\n"
                        result_message += "\n".join([f"  - {f}: {err}" for f, err in failed_files])
                    
                    return result_message
                    
                except Exception as pr_error:
                    return f"✅ Files pushed to branch `{branch_name}`, but PR creation failed: {str(pr_error)}\n\nYou can manually create a PR from the GitHub UI."
            else:
                return f"❌ No files were successfully pushed. Errors:\n" + "\n".join([f"- {f}: {err}" for f, err in failed_files])
        
        except Exception as e:
            raise Exception(f"GitHub push failed: {str(e)}")
    
    def _create_pull_request(
        self,
        wrapper: GitHubAPIWrapper,
        branch_name: str,
        commit_message: str,
        pushed_files: list
    ) -> str:
        """Create a pull request using the GitHub API."""
        try:
            # Access the GitHub instance directly
            repo = wrapper.github_repo_instance
            base_branch = wrapper.github_base_branch
            
            pr_title = commit_message[:72] + ("..." if len(commit_message) > 72 else "")
            pr_body = f"""## AI-Powered Refactoring

**Changes:**
{chr(10).join([f'- {f}' for f in pushed_files])}

**Original Request:**
{commit_message}

---
*This PR was automatically generated by REPO AI Refactorer*
"""
            
            pr = repo.create_pull(
                title=pr_title,
                body=pr_body,
                head=branch_name,
                base=base_branch
            )
            
            return f"🎉 Pull Request created: {pr.html_url}"
            
        except Exception as e:
            # If PR creation fails, at least the branch exists
            raise Exception(f"Could not create PR: {str(e)}")
    
    def run(
        self, 
        ai_response: str, 
        repo_name: str, 
        commit_message: str = "AI Refactoring",
        original_paths: Optional[Dict[str, str]] = None
    ) -> str:
        """Main entry point for pushing code to GitHub."""
        files = self.extract_files_from_response(ai_response)
        
        if not files:
            return "⚠️ No recognizable source files found in the AI response."
        
        return self.push_to_github(repo_name, files, commit_message, original_paths)