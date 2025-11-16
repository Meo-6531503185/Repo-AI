"""
Enhanced validation and diff visualization system.
Place this in: validators/code_validator.py (NEW FILE)
"""

import ast
import difflib
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import streamlit as st


@dataclass
class ValidationResult:
    """Result of code validation."""
    is_valid: bool
    file_path: str
    errors: List[str]
    warnings: List[str]
    language: str


class CodeValidator:
    """Validates refactored code before pushing to GitHub."""
    
    def __init__(self):
        self.validators = {
            'python': self._validate_python,
            'java': self._validate_java,
            'javascript': self._validate_javascript,
            'typescript': self._validate_typescript,
        }
    
    def validate_file(self, file_path: str, content: str, language: str) -> ValidationResult:
        """Validate a single file's content."""
        errors = []
        warnings = []
        
        if not content.strip():
            errors.append("File is empty")
            return ValidationResult(False, file_path, errors, warnings, language)
        
        validator = self.validators.get(language, self._validate_generic)
        lang_errors, lang_warnings = validator(content)
        errors.extend(lang_errors)
        warnings.extend(lang_warnings)
        
        if len(content) < 10:
            warnings.append("File content seems unusually short")
        
        if content.count('\n') < 3:
            warnings.append("File has very few lines - verify completeness")
        
        is_valid = len(errors) == 0
        return ValidationResult(is_valid, file_path, errors, warnings, language)
    
    def _validate_python(self, content: str) -> Tuple[List[str], List[str]]:
        """Python-specific validation using AST."""
        errors = []
        warnings = []
        
        try:
            ast.parse(content)
        except SyntaxError as e:
            errors.append(f"Syntax error at line {e.lineno}: {e.msg}")
        except Exception as e:
            errors.append(f"Parsing error: {str(e)}")
        
        if 'import *' in content:
            warnings.append("Wildcard imports detected")
        
        if content.count('TODO') > 0 or content.count('FIXME') > 0:
            warnings.append("TODO/FIXME comments found")
        
        return errors, warnings
    
    def _validate_java(self, content: str) -> Tuple[List[str], List[str]]:
        """Java-specific validation."""
        errors = []
        warnings = []
        
        open_braces = content.count('{')
        close_braces = content.count('}')
        if open_braces != close_braces:
            errors.append(f"Mismatched braces: {open_braces} opening, {close_braces} closing")
        
        open_parens = content.count('(')
        close_parens = content.count(')')
        if open_parens != close_parens:
            errors.append(f"Mismatched parentheses")
        
        return errors, warnings
    
    def _validate_javascript(self, content: str) -> Tuple[List[str], List[str]]:
        """JavaScript validation."""
        errors = []
        warnings = []
        
        open_braces = content.count('{')
        close_braces = content.count('}')
        if open_braces != close_braces:
            errors.append(f"Mismatched braces")
        
        if 'var ' in content:
            warnings.append("'var' keyword detected - consider 'let' or 'const'")
        
        return errors, warnings
    
    def _validate_typescript(self, content: str) -> Tuple[List[str], List[str]]:
        errors, warnings = self._validate_javascript(content)
        
        if ': any' in content:
            warnings.append("'any' type detected")
        
        return errors, warnings
    
    def _validate_generic(self, content: str) -> Tuple[List[str], List[str]]:
        errors = []
        warnings = []
        
        if content.count('{') != content.count('}'):
            warnings.append("Mismatched braces detected")
        
        return errors, warnings


class DiffVisualizer:
    """Creates visual diffs between original and refactored code."""
    
    @staticmethod
    def generate_diff(original: str, refactored: str, file_path: str) -> Dict:
        """Generate a structured diff."""
        differ = difflib.unified_diff(
            original.splitlines(keepends=True),
            refactored.splitlines(keepends=True),
            fromfile=f"original/{file_path}",
            tofile=f"refactored/{file_path}",
            lineterm=''
        )
        
        diff_lines = list(differ)
        
        additions = sum(1 for line in diff_lines if line.startswith('+') and not line.startswith('+++'))
        deletions = sum(1 for line in diff_lines if line.startswith('-') and not line.startswith('---'))
        
        return {
            'file_path': file_path,
            'diff': ''.join(diff_lines),
            'additions': additions,
            'deletions': deletions,
            'total_changes': additions + deletions
        }
    
    @staticmethod
    def render_diff_in_streamlit(diff_data: Dict):
        """Render diff in Streamlit UI."""
        st.markdown(f"### 📄 {diff_data['file_path']}")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Lines Added", diff_data['additions'], delta=diff_data['additions'])
        with col2:
            st.metric("Lines Deleted", diff_data['deletions'], delta=-diff_data['deletions'])
        with col3:
            st.metric("Total Changes", diff_data['total_changes'])
        
        if diff_data['diff']:
            st.code(diff_data['diff'], language='diff')
        else:
            st.info("No changes detected")


class ValidationPipeline:
    """Orchestrates validation and diff generation."""
    
    def __init__(self):
        self.validator = CodeValidator()
        self.diff_visualizer = DiffVisualizer()
    
    def validate_and_diff(
        self, 
        original_files: Dict[str, str], 
        refactored_files: Dict[str, str],
        language_detector
    ) -> Tuple[List[ValidationResult], List[Dict]]:
        """Validate all refactored files and generate diffs."""
        validation_results = []
        diff_data = []
        
        for file_path, refactored_content in refactored_files.items():
            language = language_detector(file_path)
            
            result = self.validator.validate_file(file_path, refactored_content, language)
            validation_results.append(result)
            
            if file_path in original_files:
                diff = self.diff_visualizer.generate_diff(
                    original_files[file_path],
                    refactored_content,
                    file_path
                )
                diff_data.append(diff)
        
        return validation_results, diff_data
    
    def display_validation_summary(self, validation_results: List[ValidationResult]) -> bool:
        """Display validation summary in Streamlit."""
        total_files = len(validation_results)
        valid_files = sum(1 for r in validation_results if r.is_valid)
        total_errors = sum(len(r.errors) for r in validation_results)
        total_warnings = sum(len(r.warnings) for r in validation_results)
        
        st.subheader("🔍 Validation Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Files", total_files)
        with col2:
            st.metric("Valid Files", valid_files)
        with col3:
            st.metric("Errors", total_errors)
        with col4:
            st.metric("Warnings", total_warnings)
        
        if total_errors > 0 or total_warnings > 0:
            st.markdown("---")
            for result in validation_results:
                if result.errors or result.warnings:
                    with st.expander(f"{'❌' if result.errors else '⚠️'} {result.file_path}"):
                        if result.errors:
                            st.error("**Errors:**")
                            for error in result.errors:
                                st.write(f"- {error}")
                        
                        if result.warnings:
                            st.warning("**Warnings:**")
                            for warning in result.warnings:
                                st.write(f"- {warning}")
        
        all_valid = total_errors == 0
        
        if all_valid:
            st.success("✅ All validations passed!")
        else:
            st.error("❌ Validation failed. Fix errors before pushing.")
        
        return all_valid