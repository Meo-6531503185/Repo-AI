from langchain_google_vertexai import VertexAI
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st
import re
from typing import Dict, List, Any, Tuple # <-- ADD THIS LINE
from utils.llm_normalizer import LLMOutputNormalizer, CodeExtractor


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
        self.model = VertexAI(
            model_name="gemini-2.5-pro",
            temperature=0.2,  # Lower temperature for more consistent refactoring
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
        file_lower = file_path.lower()
        file_ext = next((ext for ext in self.extension_map if file_lower.endswith(ext)), None)
        lang = self.extension_map.get(file_ext, "plaintext")
        return lang, file_ext

    def _build_dependency_context(self, target_file: str, all_files: List[str], repo_data: Dict[str, str]) -> str:
        """Build context about related files that might be affected by changes."""
        context_parts = []
        target_content = repo_data.get(target_file, "")
        
        # Extract imports/dependencies from target file
        if target_file.endswith('.py'):
            import_pattern = r'(?:from|import)\s+[\w.]+'
        elif target_file.endswith('.java'):
            import_pattern = r'import\s+[\w.]+;'
        elif target_file.endswith(('.js', '.ts')):
            import_pattern = r'(?:import|require)\s*\(["\'][\w./]+["\']\)|import.*from\s*["\'][\w./]+["\']'
        else:
            return ""
        
        import re
        imports = re.findall(import_pattern, target_content)
        
        # Find related files based on imports
        related_files = []
        for file_path in all_files:
            if file_path == target_file:
                continue
            file_name = file_path.split('/')[-1].replace('.py', '').replace('.java', '').replace('.js', '').replace('.ts', '')
            if any(file_name in imp for imp in imports):
                related_files.append(file_path)
        
        if related_files:
            context_parts.append("**Related files that may be affected:**")
            for rf in related_files[:5]:  # Limit to 5 most relevant
                # Include file signature (classes, functions) not full content
                related_content = repo_data.get(rf, "")
                if related_content:
                    signature = self._extract_signature(related_content, self._get_file_info(rf)[0])
                    context_parts.append(f"\n{rf}:\n{signature}\n")
        
        return "\n".join(context_parts)
    
    def _extract_signature(self, code: str, lang: str) -> str:
        """Extract function/class signatures from code."""
        signatures = []
        lines = code.split('\n')[:50]  # First 50 lines for context
        
        for line in lines:
            stripped = line.strip()
            if lang == "python":
                if stripped.startswith('class ') or stripped.startswith('def '):
                    signatures.append(line)
            elif lang == "java":
                if 'class ' in stripped or 'interface ' in stripped or 'public ' in stripped:
                    signatures.append(line)
            elif lang in ["javascript", "typescript"]:
                if stripped.startswith('class ') or stripped.startswith('function ') or 'const ' in stripped and '=>' in stripped:
                    signatures.append(line)
        
        return '\n'.join(signatures[:20]) if signatures else "// No clear signatures found"

    def _identify_target_files(self, user_question: str, all_files: List[str]) -> List[str]:
        """Identify files that need refactoring based on user question."""
        user_question_lower = user_question.lower()
        target_files = []
        
        # Direct file mentions
        for f in all_files:
            basename = f.split('/')[-1].lower()
            if f.lower() in user_question_lower or basename in user_question_lower:
                target_files.append(f)
        
        # Keyword-based detection
        if not target_files:
            keywords = {
                'test': lambda f: 'test' in f.lower(),
                'config': lambda f: any(x in f.lower() for x in ['config', 'settings', '.env']),
                'main': lambda f: 'main' in f.lower() or 'app' in f.lower(),
                'util': lambda f: 'util' in f.lower() or 'helper' in f.lower(),
            }
            
            for keyword, matcher in keywords.items():
                if keyword in user_question_lower:
                    target_files.extend([f for f in all_files if matcher(f)])
        
        return sorted(list(set(target_files)))

    def run(self, user_question: str, repo_data: Dict[str, str]) -> Dict[str, str]:
        """Main refactoring pipeline with enhanced context."""
        all_files = list(repo_data.keys())
        target_files = self._identify_target_files(user_question, all_files)
        
        # If no specific files identified, use heuristics or ask user
        if not target_files:
            st.warning("⚠️ No specific files identified. Analyzing repository structure...")
            # Apply to most likely candidates (exclude config/docs)
            target_files = [f for f in all_files 
                          if not any(x in f.lower() for x in ['readme', '.md', '.txt', '.json', '.yaml', 'license'])]
            target_files = target_files[:5]  # Limit to 5 files
        
        results = {}
        all_files_list_str = "\n".join(all_files)
        
        for idx, f in enumerate(target_files):
            st.info(f"🔄 Refactoring {idx+1}/{len(target_files)}: {f}")
            
            content = repo_data[f]
            lang, _ = self._get_file_info(f)
            
            # Build enhanced context
            dependency_context = self._build_dependency_context(f, all_files, repo_data)
            
            marker_char = "#" if lang in ["python", "ruby", "shell"] else "//"
            file_marker = f"{marker_char} === {f} ==="
            
            # ENHANCED SYSTEM PROMPT
            system_instruction = f"""You are an expert {lang} refactoring specialist.

**REFACTORING TASK:**
File: {f}
Goal: {user_question}

**REPOSITORY STRUCTURE:**
{all_files_list_str}

{dependency_context}

**REFACTORING REQUIREMENTS:**
1. Maintain all existing functionality - DO NOT remove features
2. Preserve API contracts (function signatures, class interfaces)
3. Keep all imports/dependencies unless explicitly asked to change
4. Follow {lang} best practices and conventions
5. Add comments for significant changes
6. Ensure backward compatibility

**CRITICAL OUTPUT FORMAT:**
- Start with: {file_marker}
- Follow with ONLY the complete refactored code
- NO markdown fences (```), explanations, or preamble
- Include ALL original code with modifications
- DO NOT truncate or summarize

**EXAMPLE OUTPUT:**
{file_marker}
[Your complete refactored code here]
"""
            
            escaped_content = content.replace("{", "{{").replace("}", "}}")
            
            prompt_messages = [
                ("system", system_instruction),
                ("user", f"""Refactor this code following the requirements above:

[CODE START]
{escaped_content}
[CODE END]

Remember: Output ONLY the marker followed by complete code. No explanations.""")
            ]
            
            prompt = ChatPromptTemplate.from_messages(prompt_messages)
            
            try:
                raw = self.model.predict(prompt)
                normalized = self.normalizer.normalize(raw)
                response_str = normalized.text
                
                # Clean and validate output
                if response_str.startswith(file_marker):
                    cleaned_content = response_str[len(file_marker):].strip()
                else:
                    cleaned_content = self.code_extractor.remove_markdown_fences(response_str)
                    cleaned_content = self.code_extractor.remove_ai_preamble(cleaned_content)
                
                # Validation: Ensure we got substantial code back
                if len(cleaned_content) < len(content) * 0.5:
                    st.warning(f"⚠️ Refactored {f} seems incomplete. Using original.")
                    results[f] = content
                else:
                    results[f] = cleaned_content
                    
            except Exception as e:
                st.error(f"❌ Error processing {f}: {str(e)}")
                results[f] = content
        
        return results