from langchain_google_vertexai import VertexAI
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st
import re
from typing import Dict, List, Any, Tuple # <-- ADD THIS LINE


class OverviewAgent:
    def __init__(self):
        self.model = VertexAI(model_name="gemini-2.5-pro")
    def run(self, user_question: str):
        repo_data = st.session_state.get("repo_data", {})
        if not repo_data: return "⚠️ No repository data available."
        files_summary = "\n".join([f"File: {f}" for f in repo_data.keys()])
        system_template = "You are an expert code analyst. Explain what the repository is about."
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_template),
            ("user", "User Question: {question}\n\nRepository Files:\n{files}")
        ]).invoke({"question": user_question, "files": files_summary})
        return self.model.invoke(prompt)

def escape_template_braces(text: str) -> str:
    return text.replace("{", "{{").replace("}", "}}")

class MultiFileRefactorAgent:
    def __init__(self):
        self.model = VertexAI(model_name="gemini-2.5-pro")
        self.extension_map = {
            ".java": "java", ".py": "python", ".js": "javascript",
            ".ts": "typescript", ".cpp": "cpp", ".c": "c", 
            ".go": "go", ".php": "php", ".rb": "ruby", 
            ".swift": "swift", ".kt": "kotlin", ".yaml": "yaml", 
            ".json": "json", ".html": "html", ".css": "css",
        }
        
        self.system_instruction_template = """
        You are an expert refactoring assistant specializing in {lang} and multi-file code changes.
        Your current task is to refactor the file: {file_path}.

        **Overall Repository Context:**
        The repository contains the following files (paths only):
        {all_files_list}
        
        **Refactoring Goal (Must maintain consistency across all files):**
        {user_question}

        **Strict Output Protocol:**
        1. **MUST START** with the file marker: `{marker_char} === {file_path} ===`
        2. Output ONLY the complete, refactored source code for {file_path}.
        3. DO NOT include explanations, reasoning, or markdown formatting.
        """

    def _get_file_info(self, file_path: str) -> Tuple[str, str]:
        file_lower = file_path.lower()
        file_ext = next((ext for ext in self.extension_map if file_lower.endswith(ext)), None)
        lang = self.extension_map.get(file_ext, "plaintext")
        return lang, file_ext

    def _identify_target_files(self, user_question: str, all_files: List[str]) -> List[str]:
        user_question_lower = user_question.lower()
        target_files = []
        for f in all_files:
            basename = f.split('/')[-1].lower()
            if f.lower() in user_question_lower or basename in user_question_lower:
                target_files.append(f)
        return sorted(list(set(target_files)))

    def run(self, user_question: str, repo_data: Dict[str, str]):
        all_files = list(repo_data.keys())
        target_files = self._identify_target_files(user_question, all_files)
        results = {}
        files_to_process = target_files if target_files else all_files
        all_files_list_str = "\n".join(all_files)

        for f in files_to_process:
            content = repo_data[f]
            lang, _ = self._get_file_info(f)
            escaped_content = escape_template_braces(content)

            marker_char = "#" if lang in ["python", "ruby", "shell"] else "//"
            file_marker = f"{marker_char} === {f} ==="

            system_instruction = self.system_instruction_template.format(
                lang=lang,
                file_path=f,
                all_files_list=all_files_list_str,
                user_question=user_question,
                marker_char=marker_char
            )

            prompt_messages = [
                ("system", system_instruction),
                ("user", f"Refactor the following code snippet for {f}:\n\n[CODE START]\n{escaped_content}\n[CODE END]")
            ]
            prompt = ChatPromptTemplate.from_messages(prompt_messages)

            try:
                raw = self.model.predict(prompt)

                # === NORMALIZE GEMINI OUTPUT TO STRING ===
                if isinstance(raw, dict):
                    response_str = raw.get("text") or raw.get("output_text") or str(raw)

                elif hasattr(raw, "text"):
                    response_str = raw.text

                elif isinstance(raw, list):
                    item = raw[0]
                    if hasattr(item, "text"):
                        response_str = item.text
                    elif isinstance(item, dict):
                        response_str = item.get("text") or item.get("output_text") or str(item)
                    else:
                        response_str = str(item)

                else:
                    response_str = str(raw)
                # =========================================

                # === CLEAN OUTPUT ===
                if response_str.startswith(file_marker):
                    cleaned_content = response_str[len(file_marker):].strip()
                else:
                    cleaned_content = re.sub(
                        r'```[a-z]*\s*|\s*```',
                        '',
                        response_str,
                        flags=re.MULTILINE
                    ).strip()

                results[f] = cleaned_content

            except Exception as e:
                print(f"Error processing file {f}: {e}")
                results[f] = content

        return results
