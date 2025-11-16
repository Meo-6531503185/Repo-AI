"""
Enhanced MultiFileRefactorAgent that actually refactors code properly.
Replace your existing MultiFileRefactorAgent in agents/sub_agents.py
"""

from langchain_google_vertexai import VertexAI
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st
from typing import Dict, List, Tuple
from utils.llm_normalizer import LLMOutputNormalizer, CodeExtractor


def escape_template_braces(text: str) -> str:
    """Escape braces for template formatting."""
    return text.replace("{", "{{").replace("}", "}}")


class MultiFileRefactorAgent:
    def __init__(self):
        self.model = VertexAI(
            model_name="gemini-2.0-flash-exp",  # Using newer, more capable model
            temperature=0.3,  # Slightly higher for creativity in refactoring
            max_output_tokens=8192
        )
        self.normalizer = LLMOutputNormalizer(provider="vertex_ai")
        self.code_extractor = CodeExtractor()
        self.extension_map = {
            ".java": "java", ".py": "python", ".js": "javascript",
            ".ts": "typescript", ".cpp": "cpp", ".c": "c", 
            ".go": "go", ".php": "php", ".rb": "ruby", 
            ".swift": "swift", ".kt": "kotlin", ".yaml": "yaml", 
            ".json": "json", ".html": "html", ".css": "css",
        }
        
    def _get_file_info(self, file_path: str) -> Tuple[str, str]:
        """Get language and extension for a file."""
        file_lower = file_path.lower()
        file_ext = next((ext for ext in self.extension_map if file_lower.endswith(ext)), None)
        lang = self.extension_map.get(file_ext, "plaintext")
        return lang, file_ext

    def _analyze_request(self, user_question: str) -> Dict[str, any]:
        """
        Analyze the user's request to understand what needs to be done.
        Returns a structured analysis.
        """
        analysis_prompt = f"""Analyze this refactoring request and provide a structured response.

Request: "{user_question}"

Provide:
1. Primary Goal: What is the main objective?
2. Required Changes: What specific modifications are needed?
3. Files Likely Affected: What types of files would need changes?
4. Implementation Strategy: How should this be implemented?

Output as JSON-like structure."""

        try:
            response = self.model.predict(analysis_prompt)
            normalized = self.normalizer.normalize(response)
            return {"analysis": normalized.text, "raw_request": user_question}
        except Exception as e:
            return {"analysis": user_question, "raw_request": user_question}

    def _identify_target_files(self, user_question: str, all_files: List[str]) -> List[str]:
        """
        Intelligently identify which files need modification.
        Uses both keyword matching and semantic understanding.
        """
        user_question_lower = user_question.lower()
        target_files = []
        
        # Direct file/path mentions
        for f in all_files:
            basename = f.split('/')[-1].lower()
            path_parts = f.lower().split('/')
            
            # Check if any part of the path is mentioned
            if any(part in user_question_lower for part in path_parts):
                target_files.append(f)
            elif basename in user_question_lower:
                target_files.append(f)
        
        # If specific files mentioned, use only those
        if target_files:
            return sorted(list(set(target_files)))
        
        # Semantic keyword-based detection
        keyword_patterns = {
            'sales': lambda f: any(x in f.lower() for x in ['sales', 'forecast', 'revenue']),
            'forecast': lambda f: 'forecast' in f.lower() or 'predict' in f.lower(),
            'dashboard': lambda f: 'dashboard' in f.lower() or 'home' in f.lower(),
            'auth': lambda f: any(x in f.lower() for x in ['auth', 'login', 'user']),
            'database': lambda f: any(x in f.lower() for x in ['db', 'database', 'model']),
            'api': lambda f: 'api' in f.lower() or 'endpoint' in f.lower(),
            'test': lambda f: 'test' in f.lower(),
            'config': lambda f: any(x in f.lower() for x in ['config', 'settings', '.env']),
            'feedback': lambda f: 'feedback' in f.lower() or 'form' in f.lower(),
            'analysis': lambda f: any(x in f.lower() for x in ['analysis', 'analytics', 'report']),
        }
        
        for keyword, matcher in keyword_patterns.items():
            if keyword in user_question_lower:
                matched_files = [f for f in all_files if matcher(f)]
                target_files.extend(matched_files)
        
        # Remove duplicates and filter out non-code files
        target_files = sorted(list(set(target_files)))
        
        # Filter out config files unless specifically mentioned
        if 'config' not in user_question_lower and 'settings' not in user_question_lower:
            target_files = [f for f in target_files if not any(
                x in f.lower() for x in ['.toml', '.json', '.yaml', '.gitignore', 'readme']
            )]
        
        return target_files[:10]  # Limit to 10 files max

    def _build_refactoring_prompt(
        self, 
        file_path: str, 
        file_content: str,
        user_question: str,
        request_analysis: Dict,
        all_files: List[str],
        lang: str
    ) -> str:
        """
        Build a comprehensive refactoring prompt with clear instructions.
        """
        marker_char = "#" if lang in ["python", "ruby", "shell"] else "//"
        
        # Build context about other files
        related_files = [f for f in all_files if f != file_path][:5]
        files_context = "\n".join([f"  - {f}" for f in related_files])
        
        prompt = f"""You are an expert {lang} developer performing a code refactoring task.

**CURRENT FILE: {file_path}**

**USER REQUEST:**
{user_question}

**REQUEST ANALYSIS:**
{request_analysis.get('analysis', 'No analysis available')}

**REPOSITORY CONTEXT:**
Other files in this repository:
{files_context}

**YOUR TASK:**
You must ACTUALLY IMPLEMENT the requested changes. Do not just return the original code.

**CRITICAL REQUIREMENTS:**
1. **MAKE REAL CHANGES** - The code MUST be different from the original
2. **IMPLEMENT THE FEATURE** - Add the requested functionality
3. **PRESERVE EXISTING FUNCTIONALITY** - Don't break what works
4. **FOLLOW {lang.upper()} BEST PRACTICES** - Use proper conventions
5. **ADD HELPFUL COMMENTS** - Explain significant changes
6. **MAINTAIN CODE STRUCTURE** - Keep imports, class definitions, etc.

**SPECIFIC INSTRUCTIONS FOR THIS REQUEST:**
- If adding a feature: Write the complete implementation
- If modifying behavior: Update the relevant functions/classes
- If adding UI elements: Include all necessary widgets and handlers
- If adding database features: Include schema changes and queries
- Add error handling where appropriate
- Add docstrings to new functions

**FORBIDDEN ACTIONS:**
- DO NOT return unchanged code
- DO NOT add only comments without functional changes
- DO NOT remove existing features unless explicitly requested
- DO NOT add placeholder comments like "# TODO: implement X"

**OUTPUT FORMAT:**
Start with: {marker_char} === {file_path} ===
Then output ONLY the COMPLETE, REFACTORED code.
NO explanations, NO markdown fences, NO preamble.

**EXAMPLE OF GOOD OUTPUT:**
{marker_char} === {file_path} ===
import streamlit as st
# ... COMPLETE WORKING CODE WITH ACTUAL CHANGES ...

Begin your refactoring now:"""

        return prompt

    def _validate_changes(self, original: str, refactored: str, file_path: str) -> Tuple[bool, str]:
        """
        Validate that meaningful changes were made.
        Returns (is_valid, reason)
        """
        # Remove whitespace for comparison
        original_stripped = ''.join(original.split())
        refactored_stripped = ''.join(refactored.split())
        
        # Check 1: Files should be different
        if original_stripped == refactored_stripped:
            return False, "No changes detected (code is identical)"
        
        # Check 2: Refactored should not be significantly shorter (unless deleting)
        if len(refactored_stripped) < len(original_stripped) * 0.7:
            return False, "Refactored code is too short (possible truncation)"
        
        # Check 3: Should have similar structure (imports, functions)
        if 'def ' in original and 'def ' not in refactored:
            return False, "Function definitions missing in refactored code"
        
        if 'class ' in original and 'class ' not in refactored:
            return False, "Class definitions missing in refactored code"
        
        # Check 4: Basic syntax check for Python
        if file_path.endswith('.py'):
            try:
                import ast
                ast.parse(refactored)
            except SyntaxError as e:
                return False, f"Syntax error in refactored code: {e}"
        
        return True, "Changes validated"

    def run(self, user_question: str, repo_data: Dict[str, str]) -> Dict[str, str]:
        """
        Main refactoring pipeline with validation and retry logic.
        """
        # Step 1: Analyze the request
        st.info("🔍 Analyzing your request...")
        request_analysis = self._analyze_request(user_question)
        
        # Step 2: Identify target files
        all_files = list(repo_data.keys())
        target_files = self._identify_target_files(user_question, all_files)
        
        if not target_files:
            st.warning("⚠️ No specific files identified. Applying to main code files...")
            # Fallback: use Python files in root or pages directory
            target_files = [f for f in all_files if f.endswith('.py') and 
                          ('pages/' in f or '/' not in f)][:5]
        
        st.info(f"📝 Identified {len(target_files)} file(s) for refactoring")
        
        results = {}
        successful_changes = 0
        
        # Step 3: Process each file
        for idx, file_path in enumerate(target_files, 1):
            st.info(f"🔄 Refactoring ({idx}/{len(target_files)}): {file_path}")
            
            original_content = repo_data[file_path]
            lang, _ = self._get_file_info(file_path)
            
            # Build the refactoring prompt
            prompt_text = self._build_refactoring_prompt(
                file_path=file_path,
                file_content=original_content,
                user_question=user_question,
                request_analysis=request_analysis,
                all_files=all_files,
                lang=lang
            )
            
            # Escape content for template
            escaped_content = escape_template_braces(original_content)
            
            full_prompt = prompt_text + f"\n\n**ORIGINAL CODE:**\n```{lang}\n{escaped_content}\n```"
            
            try:
                # Call the LLM
                raw_response = self.model.predict(full_prompt)
                
                # Normalize output
                normalized = self.normalizer.normalize(raw_response)
                response_str = normalized.text
                
                # Extract code
                marker_char = "#" if lang in ["python", "ruby", "shell"] else "//"
                file_marker = f"{marker_char} === {file_path} ==="
                
                if response_str.startswith(file_marker):
                    cleaned_content = response_str[len(file_marker):].strip()
                else:
                    cleaned_content = self.code_extractor.remove_markdown_fences(response_str)
                    cleaned_content = self.code_extractor.remove_ai_preamble(cleaned_content)
                
                # Validate changes
                is_valid, reason = self._validate_changes(original_content, cleaned_content, file_path)
                
                if is_valid:
                    results[file_path] = cleaned_content
                    successful_changes += 1
                    st.success(f"✅ {file_path} - Changes validated")
                else:
                    st.warning(f"⚠️ {file_path} - {reason}, keeping original")
                    results[file_path] = original_content
                
            except Exception as e:
                st.error(f"❌ Error processing {file_path}: {str(e)}")
                results[file_path] = original_content
        
        # Final summary
        if successful_changes == 0:
            st.error("❌ No files were successfully modified. The AI may need more context or a clearer request.")
        else:
            st.success(f"✅ Successfully modified {successful_changes}/{len(target_files)} file(s)")
        
        return results