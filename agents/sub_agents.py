from langchain_google_vertexai import VertexAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
import streamlit as st
from typing import Dict, List, Tuple, Optional
from utils.llm_normalizer import LLMOutputNormalizer, CodeExtractor
import re
import json
import os


class FileOperation(BaseModel):
    """Represents a file operation."""
    operation: str = Field(description="Operation type: 'create', 'modify', 'delete', 'rename'")
    file_path: str = Field(description="File path")
    new_path: Optional[str] = Field(default=None, description="New path for rename operations")


class FileAnalysis(BaseModel):
    """Structured analysis of the user's refactoring request."""
    file_operations: List[FileOperation] = Field(
        description="List of file operations to perform"
    )
    change_type: str = Field(
        description="Type of change: 'feature addition', 'refactoring', 'bug fix', 'documentation', or 'testing'"
    )
    implementation_points: List[str] = Field(
        description="Technical tasks to accomplish"
    )


def escape_template_braces(text: str) -> str:
    """Escape braces for template formatting."""
    return text.replace("{", "{{").replace("}", "}}")


class MultiFileRefactorAgent:
    """Comprehensive MultiFileRefactorAgent with proper file separation and import handling."""

    def __init__(self):
        self.model = VertexAI(
            model_name="gemini-2.5-pro",
            temperature=0.3,
            max_output_tokens=24000,
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

    def _get_file_info(self, file_path: str) -> Tuple[str, Optional[str]]:
        """Get language and extension for a file."""
        file_lower = file_path.lower()
        file_ext = next((ext for ext in self.extension_map if file_lower.endswith(ext)), None)
        lang = self.extension_map.get(file_ext, "plaintext")
        return lang, file_ext

    def _get_python_import_path(self, file_path: str, project_root: str = "") -> str:
        """
        Convert a file path to proper Python import path.
        Examples:
        - src/main/feedback.py → from src.main.feedback import
        - feedback_module.py → import feedback_module
        - utils/helpers.py → from utils.helpers import
        """
        # Remove .py extension
        path_without_ext = file_path.replace('.py', '')
        
        # Convert slashes to dots for Python imports
        import_path = path_without_ext.replace('/', '.')
        
        return import_path

    def _detect_project_structure(self, all_files: List[str]) -> Dict[str, any]:
        """Analyze project structure to understand layout."""
        structure = {
            'has_src': any('src/' in f for f in all_files),
            'has_pages': any('pages/' in f for f in all_files),
            'has_utils': any('utils/' in f for f in all_files),
            'root_files': [f for f in all_files if '/' not in f],
            'directories': set()
        }
        
        for file_path in all_files:
            if '/' in file_path:
                dir_path = '/'.join(file_path.split('/')[:-1])
                structure['directories'].add(dir_path)
        
        return structure

    def _extract_json_from_response(self, text: str) -> Optional[Dict]:
        """Extract JSON from various response formats."""
        try:
            return json.loads(text)
        except:
            pass
        
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)
        
        try:
            return json.loads(text.strip())
        except:
            pass
        
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.finditer(json_pattern, text, re.DOTALL)
        
        for match in matches:
            try:
                obj = json.loads(match.group())
                if 'file_operations' in obj:
                    return obj
            except:
                continue
        
        return None

    def _analyze_request(self, user_question: str, all_files: List[str]) -> Dict[str, any]:
        """Analyze request with comprehensive file operation detection."""
        files_list = "\n".join([f" - {f}" for f in all_files[:50]])
        more_files_indicator = f"... and {len(all_files) - 50} more files" if len(all_files) > 50 else ""

        # Detect project structure
        project_info = self._detect_project_structure(all_files)
        structure_hint = f"\nProject Structure: Root files in '/', modules in {list(project_info['directories'])}" if project_info['directories'] else ""

        analysis_template = """You are analyzing a code refactoring request. Return ONLY a JSON object.

User Request: {user_question}

Available Files:
{files_list}
{more_files_indicator}{structure_hint}

Analyze what file operations are needed and return this EXACT structure:
{{
  "file_operations": [
    {{"operation": "modify", "file_path": "existing_file.py"}},
    {{"operation": "create", "file_path": "new_module.py"}},
    {{"operation": "delete", "file_path": "obsolete.py"}},
    {{"operation": "rename", "file_path": "old.py", "new_path": "new.py"}}
  ],
  "change_type": "refactoring",
  "implementation_points": ["Task 1", "Task 2"]
}}

CRITICAL RULES FOR FILE PATHS:
1. For new files, MATCH THE EXISTING PROJECT STRUCTURE
   - If project has files in root (db.py, auth.py), create new files in root
   - If project uses pages/ folder, put page files there
   - If project uses utils/ or lib/, put utility files there
2. DO NOT invent new folder structures (src/main/, app/core/) unless they already exist
3. Keep file paths SIMPLE and consistent with existing project layout

Operation Types:
- "create": Make a NEW file (use same directory level as similar files)
- "modify": Change an EXISTING file (must be in Available Files)
- "delete": Remove a file that's no longer needed
- "rename": Move/rename a file

Examples:

Request: "Add feedback feature"
Existing: db.py, auth.py, pages/dashboard.py
Response:
{{
  "file_operations": [
    {{"operation": "create", "file_path": "feedback_module.py"}},
    {{"operation": "modify", "file_path": "db.py"}},
    {{"operation": "modify", "file_path": "pages/sales_forecasting.py"}}
  ],
  "change_type": "feature addition",
  "implementation_points": ["Create feedback module", "Add DB functions", "Integrate into page"]
}}

Request: "Replace BinarySearch.java with MergeSort.java"
Response:
{{
  "file_operations": [
    {{"operation": "delete", "file_path": "BinarySearch.java"}},
    {{"operation": "create", "file_path": "MergeSort.java"}}
  ],
  "change_type": "refactoring",
  "implementation_points": ["Remove binary search", "Implement merge sort"]
}}

Return ONLY the JSON object."""

        prompt_text = ChatPromptTemplate.from_template(analysis_template).format(
            user_question=user_question,
            files_list=files_list,
            more_files_indicator=more_files_indicator,
            structure_hint=structure_hint,
        )

        st.info("🧠 Analyzing file operations needed...")
        try:
            raw = self.model.invoke(prompt_text)
            raw_str = str(raw)
            
            parsed = self._extract_json_from_response(raw_str)
            
            if parsed:
                analysis = FileAnalysis(**parsed)
                return analysis.dict()
            else:
                raise ValueError("Could not extract valid JSON")
                
        except Exception as e:
            st.warning(f"⚠️ AI analysis failed: {e}. Using fallback detection.")
            return self._fallback_file_extraction("", user_question, all_files)

    def _fallback_file_extraction(self, response_text: str, user_question: str, all_files: List[str]) -> Dict:
        """Fallback file operation detection using heuristics."""
        file_operations = []
        user_lower = user_question.lower()

        # Detect operation keywords
        create_keywords = ['create', 'add', 'new', 'build', 'generate', 'make']
        delete_keywords = ['delete', 'remove', 'drop']
        rename_keywords = ['rename', 'move']
        replace_keywords = ['replace', 'swap', 'change from', 'convert']
        
        is_creating = any(kw in user_lower for kw in create_keywords)
        is_deleting = any(kw in user_lower for kw in delete_keywords)
        is_renaming = any(kw in user_lower for kw in rename_keywords)
        is_replacing = any(kw in user_lower for kw in replace_keywords)

        # Determine appropriate directory for new files
        project_info = self._detect_project_structure(all_files)
        default_dir = "" if project_info['root_files'] else "src/"

        # Extract filenames from question
        filename_pattern = r'\b([A-Z][a-zA-Z0-9_]*\.[a-z]+|[a-z][a-z0-9_]*\.[a-z]+)\b'
        mentioned_files = re.findall(filename_pattern, user_question)

        # Handle replacement
        if is_replacing:
            replace_pattern = r'replace\s+(\S+)\s+(?:with|by|using)\s+(\S+)'
            match = re.search(replace_pattern, user_lower)
            if match:
                old_file, new_file = match.groups()
                old_path = next((f for f in all_files if old_file in f.lower()), None)
                if old_path:
                    file_operations.append({"operation": "delete", "file_path": old_path})
                    # Place new file in same directory as old file
                    new_file_path = new_file if '/' in new_file else (os.path.dirname(old_path) + '/' + new_file if os.path.dirname(old_path) else new_file)
                    file_operations.append({"operation": "create", "file_path": new_file_path})

        # Handle deletions
        if is_deleting:
            for mentioned in mentioned_files:
                file_path = next((f for f in all_files if f.endswith(mentioned)), mentioned)
                if file_path in all_files:
                    file_operations.append({"operation": "delete", "file_path": file_path})

        # Handle renames
        if is_renaming:
            rename_pattern = r'rename\s+(\S+)\s+to\s+(\S+)'
            match = re.search(rename_pattern, user_lower)
            if match:
                old_name, new_name = match.groups()
                old_path = next((f for f in all_files if old_name in f.lower()), None)
                if old_path:
                    file_operations.append({
                        "operation": "rename",
                        "file_path": old_path,
                        "new_path": new_name
                    })

        # Handle creates
        if is_creating:
            for mentioned in mentioned_files:
                if not any(f.endswith(mentioned) for f in all_files):
                    # Keep file in root if that's where similar files are
                    file_operations.append({"operation": "create", "file_path": mentioned})

        # Find files to modify
        for file_path in all_files:
            basename = file_path.split('/')[-1]
            if basename.lower() in user_lower or file_path.lower() in user_lower:
                if not any(op['file_path'] == file_path for op in file_operations):
                    file_operations.append({"operation": "modify", "file_path": file_path})

        # If no operations found, assume modification
        if not file_operations:
            patterns = {
                r'(feedback|survey)': lambda f: any(x in f.lower() for x in ['feedback', 'survey', 'review']),
                r'(sort|sorting)': lambda f: 'sort' in f.lower(),
                r'(api|endpoint)': lambda f: any(x in f.lower() for x in ['api', 'route', 'controller']),
            }
            
            for pattern, matcher in patterns.items():
                if re.search(pattern, user_lower):
                    matched = [f for f in all_files if matcher(f)][:3]
                    for f in matched:
                        file_operations.append({"operation": "modify", "file_path": f})

        return {
            "file_operations": file_operations,
            "change_type": "refactoring",
            "implementation_points": [user_question],
        }

    def _build_creation_prompt(
        self,
        file_path: str,
        user_question: str,
        request_analysis: Dict,
        all_files: List[str],
        lang: str,
    ) -> str:
        """Build prompt for creating a new file with correct imports."""
        
        related_files = [f for f in all_files[:10]]
        files_context = "\n".join([f" - {f}" for f in related_files])

        implementation_points = request_analysis.get('implementation_points', [])
        impl_text = "\n".join([f" - {point}" for point in implementation_points])

        # Generate correct import statement if Python
        import_guidance = ""
        if lang == "python":
            import_path = self._get_python_import_path(file_path)
            import_guidance = f"""
CRITICAL IMPORT REQUIREMENTS:
- This file will be at: {file_path}
- Other files importing this should use: from {import_path} import ...
- If this file imports other project files, use their ACTUAL paths:
{chr(10).join([f"  - {f} → from {self._get_python_import_path(f)} import ..." for f in related_files[:5] if f.endswith('.py')])}
"""

        prompt = f"""You are an expert {lang} developer creating ONE SINGLE NEW FILE.

FILE TO CREATE: {file_path}
USER REQUEST: {user_question}
CHANGE TYPE: {request_analysis.get('change_type', 'feature addition')}

IMPLEMENTATION REQUIREMENTS:
{impl_text}

EXISTING PROJECT FILES:
{files_context}
{import_guidance}

CRITICAL INSTRUCTIONS:
1. ✅ CREATE ONLY THE FILE: {file_path}
2. ✅ DO NOT include other files in your response
3. ✅ USE CORRECT IMPORT PATHS based on file location
4. ✅ INCLUDE ALL NECESSARY IMPORTS
5. ✅ CREATE COMPLETE, PRODUCTION-READY CODE
6. ✅ ADD COMPREHENSIVE DOCUMENTATION
7. ✅ FOLLOW {lang.upper()} BEST PRACTICES
8. ✅ INCLUDE ERROR HANDLING

STRICTLY FORBIDDEN:
❌ Including multiple files in one response
❌ Using incorrect import paths (must match file structure)
❌ Placeholder code or TODOs
❌ Incomplete implementations
❌ Missing imports

OUTPUT FORMAT:
Return ONLY the complete content for {file_path}.
NO explanations, NO markdown fences, NO preamble, NO other files.
Start directly with the code for THIS FILE ONLY.

CREATE THE FILE {file_path}:"""

        return prompt

    def _build_refactoring_prompt(
        self,
        file_path: str,
        file_content: str,
        user_question: str,
        request_analysis: Dict,
        all_files: List[str],
        lang: str,
    ) -> str:
        """Build refactoring prompt with import correction."""
        
        related_files = [f for f in all_files if f != file_path][:5]
        files_context = "\n".join([f" - {f}" for f in related_files])

        implementation_points = request_analysis.get('implementation_points', [])
        impl_text = "\n".join([f" - {point}" for point in implementation_points])

        # Get info about other operations
        operations = request_analysis.get('file_operations', [])
        operations_context = ""
        import_updates = []
        
        if operations:
            ops_list = []
            for op in operations:
                if op['file_path'] != file_path:
                    op_type = op['operation']
                    if op_type == 'create':
                        new_file = op['file_path']
                        ops_list.append(f"Creating: {new_file}")
                        if lang == "python" and new_file.endswith('.py'):
                            import_path = self._get_python_import_path(new_file)
                            import_updates.append(f"New import: from {import_path} import ...")
                    elif op_type == 'delete':
                        ops_list.append(f"Deleting: {op['file_path']}")
                        import_updates.append(f"Remove imports from: {op['file_path']}")
                    elif op_type == 'rename':
                        old_path = op['file_path']
                        new_path = op.get('new_path')
                        ops_list.append(f"Renaming: {old_path} → {new_path}")
                        if lang == "python":
                            old_import = self._get_python_import_path(old_path)
                            new_import = self._get_python_import_path(new_path)
                            import_updates.append(f"Update import: from {old_import} → from {new_import}")
            
            if ops_list:
                operations_context = "\n\nOTHER CHANGES IN THIS REFACTORING:\n" + "\n".join(ops_list)
            
            if import_updates:
                operations_context += "\n\nIMPORT UPDATES REQUIRED:\n" + "\n".join(import_updates)

        prompt = f"""You are an expert {lang} developer. You MUST implement the requested changes.

FILE TO MODIFY: {file_path}
USER REQUEST: {user_question}
CHANGE TYPE: {request_analysis.get('change_type', 'refactoring')}

IMPLEMENTATION REQUIREMENTS:
{impl_text}

OTHER FILES IN PROJECT:
{files_context}{operations_context}

CRITICAL INSTRUCTIONS:
1. ✅ MODIFY ONLY THIS FILE: {file_path}
2. ✅ DO NOT include other files in your response
3. ✅ UPDATE IMPORTS to match new file locations/names
4. ✅ REMOVE REFERENCES to deleted files
5. ✅ IMPLEMENT THE FULL CHANGES - No placeholders
6. ✅ USE CORRECT IMPORT PATHS (file location matters!)
7. ✅ MAKE REAL CODE CHANGES - Not just comments
8. ✅ FOLLOW {lang.upper()} BEST PRACTICES

STRICTLY FORBIDDEN:
❌ Returning multiple files concatenated together
❌ Returning unchanged code
❌ Using incorrect import paths
❌ Adding TODO/placeholder comments without implementation
❌ Leaving references to deleted files
❌ Truncating or summarizing code

OUTPUT FORMAT:
Return ONLY the complete refactored code for {file_path}.
NO explanations, NO markdown fences, NO preamble, NO other files.
Start directly with the code.

ORIGINAL CODE FOR {file_path}:
{file_content}

REFACTORED CODE FOR {file_path}:"""

        return prompt

    def _validate_single_file_output(self, content: str, file_path: str) -> Tuple[bool, str]:
        """Check if output contains only one file (not multiple concatenated files)."""
        
        # Check for common multi-file markers
        multi_file_markers = [
            r'#\s*==+\s*\w+\.py\s*==+',  # Python file separators
            r'//\s*==+\s*\w+\.java\s*==+',  # Java file separators
            r'#\s+\w+\.py\s*$',  # File headers like "# db.py"
            r'//\s+\w+\.java\s*$',  # File headers like "// Main.java"
        ]
        
        for pattern in multi_file_markers:
            matches = re.findall(pattern, content, re.MULTILINE)
            if len(matches) > 1:
                return False, f"Output contains multiple files (found {len(matches)} file markers)"
        
        # Check for suspicious file count
        lines = content.split('\n')
        file_header_count = 0
        for line in lines[:50]:  # Check first 50 lines
            if re.match(r'^#\s*[a-z_]+\.py\s*$', line.strip(), re.IGNORECASE):
                file_header_count += 1
            elif re.match(r'^//\s*[a-zA-Z_]+\.java\s*$', line.strip()):
                file_header_count += 1
        
        if file_header_count > 1:
            return False, f"Found {file_header_count} file header comments - output may contain multiple files"
        
        return True, "Single file output confirmed"

    def _validate_changes(self, original: str, refactored: str, file_path: str, is_new_file: bool = False) -> Tuple[bool, str]:
        """Validate refactored code."""
        
        # First check: Single file output
        is_single, msg = self._validate_single_file_output(refactored, file_path)
        if not is_single:
            return False, msg
        
        if is_new_file:
            if not refactored or len(refactored.strip()) < 50:
                return False, "New file content is too short or empty"
            
            if refactored.upper().count('TODO') > 3:
                return False, "Too many TODOs in new file"
            
            lang, _ = self._get_file_info(file_path)
            if lang == "python":
                try:
                    import ast
                    ast.parse(refactored)
                except SyntaxError as e:
                    return False, f"Python syntax error: {e}"
            
            return True, "New file validation passed"
        
        def normalize(code):
            return re.sub(r'\s+', ' ', code).strip()
        
        orig_norm = normalize(original)
        refact_norm = normalize(refactored)

        if orig_norm == refact_norm:
            return False, "No changes detected (code is identical)"

        if len(refact_norm) < len(orig_norm) * 0.5:
            return False, f"Code too short ({len(refact_norm)} vs {len(orig_norm)} chars)"

        lang, _ = self._get_file_info(file_path)

        if lang == "python":
            try:
                import ast
                ast.parse(refactored)
            except SyntaxError as e:
                return False, f"Python syntax error: {e}"

        elif lang in ("java", "javascript", "typescript", "c", "cpp"):
            refact_open = refactored.count('{')
            refact_close = refactored.count('}')
            
            if refact_open != refact_close:
                return False, f"Unbalanced braces ({refact_open} open, {refact_close} close)"

        orig_todos = original.upper().count('TODO')
        refact_todos = refactored.upper().count('TODO')
        
        if refact_todos > orig_todos + 2:
            return False, f"Too many TODOs added ({refact_todos} vs {orig_todos})"

        return True, "Validation passed"

    def run(self, user_question: str, repo_data: Dict[str, str]) -> Dict[str, str]:
        """Main refactoring pipeline with full file operation support."""
        
        st.info("🔍 Analyzing file operations...")
        all_files = list(repo_data.keys())
        request_analysis = self._analyze_request(user_question, all_files)

        operations = request_analysis.get("file_operations", [])
        
        if not operations:
            st.error("❌ No file operations identified.")
            return {}

        # Categorize operations
        creates = [op for op in operations if op['operation'] == 'create']
        modifies = [op for op in operations if op['operation'] == 'modify']
        deletes = [op for op in operations if op['operation'] == 'delete']
        renames = [op for op in operations if op['operation'] == 'rename']

        # Display plan
        st.success(f"📋 **File Operation Plan**")
        
        if creates:
            st.write(f"✨ **Create** ({len(creates)} file(s)):")
            for op in creates:
                st.write(f"   • {op['file_path']}")
        
        if modifies:
            st.write(f"📝 **Modify** ({len(modifies)} file(s)):")
            for op in modifies:
                st.write(f"   • {op['file_path']}")
        
        if deletes:
            st.write(f"🗑️ **Delete** ({len(deletes)} file(s)):")
            for op in deletes:
                st.write(f"   • {op['file_path']}")
        
        if renames:
            st.write(f"📦 **Rename** ({len(renames)} file(s)):")
            for op in renames:
                st.write(f"   • {op['file_path']} → {op.get('new_path', '???')}")

        results = {}
        successful_operations = 0
        total_operations = len(creates) + len(modifies) + len(renames)

        # 1. Handle DELETIONS
        for op in deletes:
            file_path = op['file_path']
            if file_path in repo_data:
                st.info(f"🗑️ Marking for deletion: **{file_path}**")
                results[file_path] = None
                successful_operations += 1
                st.success(f"✅ {file_path} - Marked for deletion")
            else:
                st.warning(f"⚠️ {file_path} - File not found")

        # 2. Handle RENAMES
        for idx, op in enumerate(renames, 1):
            old_path = op['file_path']
            new_path = op.get('new_path')
            
            if not new_path:
                st.error(f"❌ Rename missing new_path for {old_path}")
                continue
                
            st.info(f"📦 Renaming ({idx}/{len(renames)}): **{old_path}** → **{new_path}**")
            
            if old_path in repo_data:
                results[new_path] = repo_data[old_path]
                results[old_path] = None
                successful_operations += 1
                st.success(f"✅ {old_path} → {new_path}")
            else:
                st.error(f"❌ {old_path} - File not found")

        # 3. Handle CREATES
        for idx, op in enumerate(creates, 1):
            file_path = op['file_path']
            st.info(f"✨ Creating ({idx}/{len(creates)}): **{file_path}**")

            lang, _ = self._get_file_info(file_path)

            prompt_text = self._build_creation_prompt(
                file_path=file_path,
                user_question=user_question,
                request_analysis=request_analysis,
                all_files=all_files,
                lang=lang,
            )

            try:
                raw_response = self.model.invoke(prompt_text)
                normalized = self.normalizer.normalize(raw_response)
                response_str = normalized.text

                cleaned_content = self.code_extractor.remove_markdown_fences(response_str)
                cleaned_content = self.code_extractor.remove_ai_preamble(cleaned_content)

                is_valid, reason = self._validate_changes("", cleaned_content, file_path, is_new_file=True)

                if is_valid:
                    results[file_path] = cleaned_content
                    successful_operations += 1
                    st.success(f"✅ {file_path} - Created")
                else:
                    st.warning(f"⚠️ {file_path} - Issue: {reason}. Retrying...")
                    
                    retry_prompt = f"""PREVIOUS ATTEMPT FAILED: {reason}

CRITICAL: Create ONLY the file {file_path}. Do NOT include any other files.

{prompt_text}"""

                    retry_response = self.model.invoke(retry_prompt)
                    retry_normalized = self.normalizer.normalize(retry_response)
                    retry_str = retry_normalized.text

                    retry_cleaned = self.code_extractor.remove_markdown_fences(retry_str)
                    retry_cleaned = self.code_extractor.remove_ai_preamble(retry_cleaned)

                    retry_valid, retry_reason = self._validate_changes("", retry_cleaned, file_path, is_new_file=True)

                    if retry_valid:
                        results[file_path] = retry_cleaned
                        successful_operations += 1
                        st.success(f"✅ {file_path} - Retry successful")
                    else:
                        st.error(f"❌ {file_path} - Retry failed: {retry_reason}")

            except Exception as e:
                st.error(f"❌ Error creating {file_path}: {str(e)}")

        # 4. Handle MODIFIES
        for idx, op in enumerate(modifies, 1):
            file_path = op['file_path']
            
            if file_path not in repo_data:
                st.warning(f"⚠️ {file_path} - File not found, skipping")
                continue
                
            st.info(f"🔄 Modifying ({idx}/{len(modifies)}): **{file_path}**")

            original_content = repo_data[file_path]
            lang, _ = self._get_file_info(file_path)

            prompt_text = self._build_refactoring_prompt(
                file_path=file_path,
                file_content=original_content,
                user_question=user_question,
                request_analysis=request_analysis,
                all_files=all_files,
                lang=lang,
            )

            try:
                raw_response = self.model.invoke(prompt_text)
                normalized = self.normalizer.normalize(raw_response)
                response_str = normalized.text

                cleaned_content = self.code_extractor.remove_markdown_fences(response_str)
                cleaned_content = self.code_extractor.remove_ai_preamble(cleaned_content)

                is_valid, reason = self._validate_changes(original_content, cleaned_content, file_path)

                if is_valid:
                    results[file_path] = cleaned_content
                    successful_operations += 1
                    st.success(f"✅ {file_path} - Modified")
                else:
                    st.warning(f"⚠️ {file_path} - Issue: {reason}. Retrying...")
                    
                    retry_prompt = f"""PREVIOUS ATTEMPT FAILED: {reason}

CRITICAL: Modify ONLY the file {file_path}. Do NOT include any other files.
Make substantial functional changes to THIS FILE ONLY.

{prompt_text}"""

                    retry_response = self.model.invoke(retry_prompt)
                    retry_normalized = self.normalizer.normalize(retry_response)
                    retry_str = retry_normalized.text

                    retry_cleaned = self.code_extractor.remove_markdown_fences(retry_str)
                    retry_cleaned = self.code_extractor.remove_ai_preamble(retry_cleaned)

                    retry_valid, retry_reason = self._validate_changes(original_content, retry_cleaned, file_path)

                    if retry_valid:
                        results[file_path] = retry_cleaned
                        successful_operations += 1
                        st.success(f"✅ {file_path} - Retry successful")
                    else:
                        st.error(f"❌ {file_path} - Retry failed: {retry_reason}")
                        results[file_path] = original_content

            except Exception as e:
                st.error(f"❌ Error modifying {file_path}: {str(e)}")
                results[file_path] = original_content

        # Summary
        total_expected = len(operations)
        if successful_operations == 0:
            st.error("❌ No file operations completed successfully.")
            st.info("💡 Try being more specific about the changes needed.")
        else:
            st.success(f"🎉 Successfully completed {successful_operations}/{total_expected} operation(s)")
            
            # Show import guidance for Python projects
            python_files_created = [op['file_path'] for op in creates if op['file_path'].endswith('.py')]
            if python_files_created:
                st.info("📦 **Import Guide for New Python Files:**")
                for file_path in python_files_created:
                    import_path = self._get_python_import_path(file_path)
                    st.code(f"from {import_path} import ...")

        return results