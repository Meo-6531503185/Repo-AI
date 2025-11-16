"""
Universal MultiFileRefactorAgent - Works for ANY project type and language.
Replace the entire MultiFileRefactorAgent class in agents/sub_agents.py
"""

from langchain_google_vertexai import VertexAI
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st
from typing import Dict, List, Tuple
from utils.llm_normalizer import LLMOutputNormalizer, CodeExtractor
import re


def escape_template_braces(text: str) -> str:
    """Escape braces for template formatting."""
    return text.replace("{", "{{").replace("}", "}}")


class MultiFileRefactorAgent:
    def __init__(self):
        self.model = VertexAI(
            model_name="gemini-2.0-flash-exp",
            temperature=0.3,
            max_output_tokens=8192
        )
        self.normalizer = LLMOutputNormalizer(provider="vertex_ai")
        self.code_extractor = CodeExtractor()
        self.extension_map = {
            ".java": "java", ".py": "python", ".js": "javascript",
            ".ts": "typescript", ".jsx": "javascript", ".tsx": "typescript",
            ".cpp": "cpp", ".c": "c", ".h": "c", ".hpp": "cpp",
            ".go": "go", ".php": "php", ".rb": "ruby", 
            ".swift": "swift", ".kt": "kotlin", ".rs": "rust",
            ".yaml": "yaml", ".yml": "yaml", ".json": "json", 
            ".html": "html", ".css": "css", ".scss": "scss",
            ".sql": "sql", ".sh": "shell", ".bash": "shell",
            ".xml": "xml", ".md": "markdown", ".txt": "plaintext",
        }
        
    def _get_file_info(self, file_path: str) -> Tuple[str, str]:
        """Get language and extension for a file."""
        file_lower = file_path.lower()
        file_ext = next((ext for ext in self.extension_map if file_lower.endswith(ext)), None)
        lang = self.extension_map.get(file_ext, "plaintext")
        return lang, file_ext

    def _analyze_request(self, user_question: str, all_files: List[str]) -> Dict[str, any]:
        """
        Use AI to analyze the request and identify target files.
        This makes it truly universal - works for ANY type of project.
        """
        files_list = "\n".join([f"  - {f}" for f in all_files[:50]])  # First 50 files
        
        analysis_prompt = f"""You are a code refactoring expert. Analyze this request and identify which files need modification.

**User Request:**
"{user_question}"

**Available Files in Repository:**
{files_list}
{f"... and {len(all_files) - 50} more files" if len(all_files) > 50 else ""}

**Your Task:**
Analyze the request and identify:
1. **Target Files**: Which specific files need to be modified? List their exact paths.
2. **Change Type**: What kind of changes? (feature addition, refactoring, bug fix, etc.)
3. **Key Implementation Points**: What are the main things that need to be done?

**Guidelines:**
- If specific file names/paths are mentioned, use those exactly
- If no specific files mentioned, infer from the request context
- Consider file names, directory structure, and common patterns
- Prefer main source files over config/test files unless explicitly requested
- If unsure, identify 2-5 most likely files

**Output Format (strict JSON):**
{{
  "target_files": ["path/to/file1.py", "path/to/file2.js"],
  "change_type": "feature addition",
  "implementation_points": [
    "Add new function X to handle Y",
    "Modify existing function Z"
  ]
}}

Output ONLY valid JSON, no other text."""

        try:
            response = self.model.predict(analysis_prompt)
            normalized = self.normalizer.normalize(response)
            response_text = normalized.text.strip()
            
            # Extract JSON from response (handle markdown fences)
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(1)
            
            # Try to parse as JSON
            import json
            try:
                parsed = json.loads(response_text)
                return {
                    "target_files": parsed.get("target_files", []),
                    "change_type": parsed.get("change_type", "modification"),
                    "implementation_points": parsed.get("implementation_points", []),
                    "raw_request": user_question
                }
            except json.JSONDecodeError:
                # Fallback: extract file paths from text
                return self._fallback_file_extraction(response_text, user_question, all_files)
                
        except Exception as e:
            st.warning(f"⚠️ Analysis failed: {e}. Using fallback method.")
            return self._fallback_file_extraction("", user_question, all_files)

    def _fallback_file_extraction(self, response_text: str, user_question: str, all_files: List[str]) -> Dict:
        """Fallback method if AI analysis fails."""
        target_files = []
        user_lower = user_question.lower()
        
        # Method 1: Check if specific files are mentioned in the request
        for file_path in all_files:
            basename = file_path.split('/')[-1]
            if basename.lower() in user_lower or file_path.lower() in user_lower:
                target_files.append(file_path)
        
        # Method 2: Generic pattern matching
        if not target_files:
            # Common action keywords that suggest which files to modify
            patterns = {
                # Backend/API
                r'\b(api|endpoint|route|controller|service)\b': lambda f: any(x in f.lower() for x in ['api', 'route', 'controller', 'service', 'handler']),
                # Frontend/UI
                r'\b(ui|interface|component|page|view|screen)\b': lambda f: any(x in f.lower() for x in ['component', 'page', 'view', 'screen', 'ui', 'frontend']),
                # Database
                r'\b(database|db|model|schema|query|table)\b': lambda f: any(x in f.lower() for x in ['model', 'db', 'database', 'schema', 'migration']),
                # Authentication
                r'\b(auth|login|user|session|token)\b': lambda f: any(x in f.lower() for x in ['auth', 'login', 'user', 'session']),
                # Testing
                r'\b(test|spec|unit|integration)\b': lambda f: any(x in f.lower() for x in ['test', 'spec', '__test__']),
                # Configuration
                r'\b(config|setting|environment)\b': lambda f: any(x in f.lower() for x in ['config', 'setting', 'env']),
                # Utilities
                r'\b(util|helper|tool|common)\b': lambda f: any(x in f.lower() for x in ['util', 'helper', 'tool', 'common', 'lib']),
            }
            
            for pattern, matcher in patterns.items():
                if re.search(pattern, user_lower):
                    matched = [f for f in all_files if matcher(f)]
                    target_files.extend(matched[:3])  # Take first 3 matches per pattern
        
        # Method 3: If still no files, take main source files
        if not target_files:
            # Prioritize by common main file patterns
            priority_patterns = [
                r'(main|app|index)\.(py|js|ts|java|go|rb)$',
                r'src/.+\.(py|js|ts|java|go|rb)$',
                r'^[^/]+\.(py|js|ts|java|go|rb)$',  # Root level files
            ]
            
            for pattern in priority_patterns:
                matched = [f for f in all_files if re.search(pattern, f.lower())]
                target_files.extend(matched[:3])
                if target_files:
                    break
        
        # Remove duplicates and limit
        target_files = sorted(list(set(target_files)))[:10]
        
        return {
            "target_files": target_files,
            "change_type": "modification",
            "implementation_points": [user_question],
            "raw_request": user_question
        }

    def _build_refactoring_prompt(
        self, 
        file_path: str, 
        file_content: str,
        user_question: str,
        request_analysis: Dict,
        all_files: List[str],
        lang: str
    ) -> str:
        """Build a comprehensive refactoring prompt."""
        marker_char = "#" if lang in ["python", "ruby", "shell"] else "//"
        
        # Build context about other files
        related_files = [f for f in all_files if f != file_path][:5]
        files_context = "\n".join([f"  - {f}" for f in related_files])
        
        implementation_points = request_analysis.get('implementation_points', [user_question])
        if isinstance(implementation_points, list):
            impl_text = "\n".join([f"  • {point}" for point in implementation_points])
        else:
            impl_text = f"  • {implementation_points}"
        
        prompt = f"""You are an expert {lang} developer performing a code refactoring task.

**CURRENT FILE: {file_path}**

**USER REQUEST:**
{user_question}

**CHANGE TYPE:** {request_analysis.get('change_type', 'modification')}

**IMPLEMENTATION REQUIREMENTS:**
{impl_text}

**REPOSITORY CONTEXT:**
Other files in this repository:
{files_context}

**YOUR TASK:**
You MUST implement the requested changes. This is NOT a review - you need to WRITE THE CODE.

**CRITICAL REQUIREMENTS:**
1. ✅ **MAKE REAL CHANGES** - Code MUST be functionally different from original
2. ✅ **IMPLEMENT COMPLETELY** - No placeholders, TODOs, or incomplete code
3. ✅ **PRESERVE EXISTING CODE** - Don't remove working features
4. ✅ **FOLLOW {lang.upper()} CONVENTIONS** - Use language best practices
5. ✅ **ADD DOCUMENTATION** - Comment significant changes
6. ✅ **HANDLE ERRORS** - Add try-catch/error handling where appropriate

**SPECIFIC GUIDANCE BY CHANGE TYPE:**

**If adding a feature:**
- Write complete implementation with all necessary functions/classes
- Add imports if needed
- Include input validation
- Add comments explaining the new feature

**If refactoring:**
- Improve code structure while maintaining functionality
- Update variable/function names if clearer
- Add type hints/annotations
- Optimize logic where possible

**If fixing a bug:**
- Identify and fix the root cause
- Add validation to prevent recurrence
- Add comments explaining the fix

**If adding tests:**
- Write comprehensive test cases
- Cover edge cases and error scenarios
- Follow testing framework conventions

**ABSOLUTE PROHIBITIONS:**
❌ DO NOT return unchanged code
❌ DO NOT add comments without code changes
❌ DO NOT use placeholders like "# TODO: implement"
❌ DO NOT remove functionality unless explicitly requested
❌ DO NOT truncate or summarize code

**OUTPUT FORMAT:**
{marker_char} === {file_path} ===
[COMPLETE REFACTORED CODE HERE]

NO explanations before or after the code.
NO markdown fences.
NO "Here's the refactored code" preamble.
JUST the marker followed by complete working code.

**QUALITY CHECK:**
Before outputting, verify:
- ✓ Does this code implement what the user asked for?
- ✓ Is this functionally different from the original?
- ✓ Would this code run without errors?
- ✓ Are all imports and dependencies included?

Begin refactoring now:"""

        return prompt

    def _validate_changes(self, original: str, refactored: str, file_path: str) -> Tuple[bool, str]:
        """Validate that meaningful changes were made."""
        # Remove whitespace for comparison
        original_stripped = ''.join(original.split())
        refactored_stripped = ''.join(refactored.split())
        
        # Check 1: Files should be different
        if original_stripped == refactored_stripped:
            return False, "No changes detected (code is identical)"
        
        # Check 2: Refactored should not be significantly shorter (unless deleting)
        if len(refactored_stripped) < len(original_stripped) * 0.5:
            return False, "Refactored code is too short (possible truncation)"
        
        # Check 3: Should have similar structure (language-specific)
        lang, _ = self._get_file_info(file_path)
        
        if lang == "python":
            # Check for Python structure
            if 'def ' in original and 'def ' not in refactored and len(original) > 100:
                return False, "Function definitions missing in refactored code"
            if 'class ' in original and 'class ' not in refactored:
                return False, "Class definitions missing in refactored code"
            
            # Syntax check
            try:
                import ast
                ast.parse(refactored)
            except SyntaxError as e:
                return False, f"Syntax error: {e}"
        
        elif lang in ["javascript", "typescript"]:
            # Check for JS/TS structure
            if 'function ' in original and 'function ' not in refactored and '=>' not in refactored:
                return False, "Function definitions may be missing"
        
        elif lang == "java":
            # Check for Java structure
            if 'class ' in original and 'class ' not in refactored:
                return False, "Class definitions missing"
            # Basic brace matching
            if original.count('{') != refactored.count('{'):
                return False, "Brace mismatch detected"
        
        # Check 4: Should not have TODO placeholders if original didn't
        if 'TODO' not in original.upper() and 'TODO' in refactored.upper():
            todo_count = refactored.upper().count('TODO')
            if todo_count > 2:  # Allow a couple of TODOs for future improvements
                return False, f"Contains {todo_count} TODO placeholders - code may be incomplete"
        
        return True, "Changes validated"

    def run(self, user_question: str, repo_data: Dict[str, str]) -> Dict[str, str]:
        """Main refactoring pipeline - works for any project type."""
        
        # Step 1: Analyze the request using AI
        st.info("🔍 Analyzing your request with AI...")
        all_files = list(repo_data.keys())
        request_analysis = self._analyze_request(user_question, all_files)
        
        # Step 2: Get target files from analysis
        target_files = request_analysis.get("target_files", [])
        
        # Validate target files exist in repo
        target_files = [f for f in target_files if f in repo_data]
        
        if not target_files:
            st.warning("⚠️ No specific files identified by AI. Using fallback detection...")
            target_files = request_analysis.get("target_files", [])
            
            # Last resort: ask user or take first few source files
            if not target_files:
                source_files = [f for f in all_files if any(f.endswith(ext) for ext in ['.py', '.js', '.java', '.go', '.rb', '.php'])]
                target_files = source_files[:5]
        
        if not target_files:
            st.error("❌ Could not identify any files to modify. Please specify file names in your request.")
            return {}
        
        st.info(f"📝 Identified {len(target_files)} file(s) for refactoring:")
        for f in target_files:
            st.write(f"   • {f}")
        
        results = {}
        successful_changes = 0
        
        # Step 3: Process each file
        for idx, file_path in enumerate(target_files, 1):
            st.info(f"🔄 Processing ({idx}/{len(target_files)}): {file_path}")
            
            original_content = repo_data.get(file_path)
            if not original_content:
                st.warning(f"⚠️ File not found in repository: {file_path}")
                continue
            
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
                    st.warning(f"⚠️ {file_path} - {reason}")
                    # Try one more time with stronger emphasis
                    st.info(f"   🔄 Retrying with enhanced prompt...")
                    
                    retry_prompt = f"""CRITICAL: The previous attempt returned unchanged code or invalid changes.

This is attempt #2. You MUST make substantial, functional changes to the code.

{full_prompt}

**REMINDER:** Return code that is FUNCTIONALLY DIFFERENT from the original. Add real features, not just comments."""
                    
                    retry_response = self.model.predict(retry_prompt)
                    retry_normalized = self.normalizer.normalize(retry_response)
                    retry_str = retry_normalized.text
                    
                    if retry_str.startswith(file_marker):
                        retry_cleaned = retry_str[len(file_marker):].strip()
                    else:
                        retry_cleaned = self.code_extractor.remove_markdown_fences(retry_str)
                        retry_cleaned = self.code_extractor.remove_ai_preamble(retry_cleaned)
                    
                    retry_valid, retry_reason = self._validate_changes(original_content, retry_cleaned, file_path)
                    
                    if retry_valid:
                        results[file_path] = retry_cleaned
                        successful_changes += 1
                        st.success(f"✅ {file_path} - Retry successful!")
                    else:
                        st.error(f"❌ {file_path} - Retry also failed: {retry_reason}. Keeping original.")
                        results[file_path] = original_content
                
            except Exception as e:
                st.error(f"❌ Error processing {file_path}: {str(e)}")
                results[file_path] = original_content
        
        # Final summary
        if successful_changes == 0:
            st.error("❌ No files were successfully modified.")
            st.write("**Suggestions:**")
            st.write("  • Be more specific about what files to modify")
            st.write("  • Include file names in your request")
            st.write("  • Break down complex requests into smaller tasks")
        else:
            st.success(f"✅ Successfully modified {successful_changes}/{len(target_files)} file(s)")
        
        return results