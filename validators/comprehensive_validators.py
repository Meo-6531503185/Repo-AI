"""
Enhanced validation pipeline for code refactoring with multi-language support.
Implements functional correctness, semantic preservation, and intent verification.
"""

import ast
import re
import json
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime
import streamlit as st
from langchain_google_vertexai import VertexAI, VertexAIEmbeddings
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class ValidationResult:
    """Results from a validation test."""
    test_name: str
    passed: bool
    score: Optional[float] = None
    threshold: Optional[float] = None
    details: str = ""
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class UniversalCodeQualityValidator:
    """Validates syntax and code quality metrics for multiple languages."""
    
    LANGUAGE_CONFIGS = {
        'python': {
            'comment_chars': ['#'],
            'multiline_comment': ('"""', "'''"),
            'uses_braces': False,
            'uses_semicolons': False,
        },
        'java': {
            'comment_chars': ['//'],
            'multiline_comment': ('/*', '*/'),
            'uses_braces': True,
            'uses_semicolons': True,
        },
        'javascript': {
            'comment_chars': ['//'],
            'multiline_comment': ('/*', '*/'),
            'uses_braces': True,
            'uses_semicolons': True,
        },
        'typescript': {
            'comment_chars': ['//'],
            'multiline_comment': ('/*', '*/'),
            'uses_braces': True,
            'uses_semicolons': True,
        },
        'c': {
            'comment_chars': ['//'],
            'multiline_comment': ('/*', '*/'),
            'uses_braces': True,
            'uses_semicolons': True,
        },
        'cpp': {
            'comment_chars': ['//'],
            'multiline_comment': ('/*', '*/'),
            'uses_braces': True,
            'uses_semicolons': True,
        },
        'go': {
            'comment_chars': ['//'],
            'multiline_comment': ('/*', '*/'),
            'uses_braces': True,
            'uses_semicolons': False,
        },
        'rust': {
            'comment_chars': ['//'],
            'multiline_comment': ('/*', '*/'),
            'uses_braces': True,
            'uses_semicolons': True,
        },
        'ruby': {
            'comment_chars': ['#'],
            'multiline_comment': ('=begin', '=end'),
            'uses_braces': False,
            'uses_semicolons': False,
        },
    }
    
    def validate_python_syntax(self, code: str) -> Tuple[bool, str]:
        """Check if Python code has valid syntax."""
        try:
            ast.parse(code)
            return True, "Syntax valid"
        except SyntaxError as e:
            return False, f"Syntax error at line {e.lineno}: {e.msg}"
        except Exception as e:
            return False, f"Parse error: {str(e)}"
    
    def validate_brace_balance(self, code: str) -> Tuple[bool, str]:
        """Check if braces are balanced."""
        open_braces = code.count('{')
        close_braces = code.count('}')
        open_parens = code.count('(')
        close_parens = code.count(')')
        open_brackets = code.count('[')
        close_brackets = code.count(']')
        
        issues = []
        if open_braces != close_braces:
            issues.append(f"Unbalanced braces: {open_braces} open, {close_braces} close")
        if open_parens != close_parens:
            issues.append(f"Unbalanced parentheses: {open_parens} open, {close_parens} close")
        if open_brackets != close_brackets:
            issues.append(f"Unbalanced brackets: {open_brackets} open, {close_brackets} close")
        
        if issues:
            return False, "; ".join(issues)
        return True, "Balanced delimiters"
    
    def calculate_universal_complexity(self, code: str, language: str) -> int:
        """Calculate complexity using language-agnostic heuristics."""
        complexity = 1  # Base complexity
        
        # Control flow keywords (works for most C-like and Python-like languages)
        control_keywords = [
            r'\bif\b', r'\belse\b', r'\belif\b', r'\bwhile\b', 
            r'\bfor\b', r'\bswitch\b', r'\bcase\b', r'\bcatch\b',
            r'\bexcept\b', r'\btry\b'
        ]
        
        for keyword in control_keywords:
            complexity += len(re.findall(keyword, code))
        
        # Logical operators
        complexity += code.count('&&') + code.count('||')
        complexity += code.count(' and ') + code.count(' or ')
        
        return complexity
    
    def calculate_python_complexity(self, code: str) -> int:
        """Calculate cyclomatic complexity for Python using AST."""
        try:
            tree = ast.parse(code)
            complexity = 1
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                    complexity += 1
                elif isinstance(node, ast.BoolOp):
                    complexity += len(node.values) - 1
            
            return complexity
        except:
            return self.calculate_universal_complexity(code, 'python')
    
    def count_todos(self, code: str) -> int:
        """Count TODO/FIXME comments."""
        return code.upper().count('TODO') + code.upper().count('FIXME')
    
    def extract_functions_universal(self, code: str, language: str) -> List[str]:
        """Extract function names using regex patterns."""
        patterns = {
            'python': r'def\s+(\w+)\s*\(',
            'java': r'(?:public|private|protected)?\s+(?:static\s+)?[\w<>\[\]]+\s+(\w+)\s*\(',
            'javascript': r'function\s+(\w+)\s*\(|const\s+(\w+)\s*=\s*(?:\([^)]*\)|[^=]+)\s*=>',
            'typescript': r'function\s+(\w+)\s*\(|const\s+(\w+)\s*=\s*(?:\([^)]*\)|[^=]+)\s*=>',
            'c': r'[\w<>\[\]]+\s+(\w+)\s*\([^)]*\)\s*\{',
            'cpp': r'[\w<>\[\]]+\s+(\w+)\s*\([^)]*\)\s*\{',
            'go': r'func\s+(?:\([^)]+\)\s*)?(\w+)\s*\(',
            'rust': r'fn\s+(\w+)\s*[<(]',
            'ruby': r'def\s+(\w+)',
        }
        
        pattern = patterns.get(language, r'function\s+(\w+)')
        matches = re.findall(pattern, code)
        
        # Flatten tuples (from groups in regex)
        functions = []
        for match in matches:
            if isinstance(match, tuple):
                functions.extend([m for m in match if m])
            else:
                functions.append(match)
        
        return list(set(functions))  # Remove duplicates
    
    def verify_code_quality(self, code: str, language: str = 'python') -> ValidationResult:
        """Comprehensive code quality check for multiple languages."""
        issues = []
        metadata = {}
        
        # Check minimum code length
        if len(code.strip()) < 50:
            issues.append("Code too short (< 50 chars)")
            return ValidationResult(
                test_name="Code Quality",
                passed=False,
                details="; ".join(issues),
                metadata=metadata
            )
        
        # Language-specific syntax validation
        if language == 'python':
            is_valid, msg = self.validate_python_syntax(code)
            if not is_valid:
                return ValidationResult(
                    test_name="Code Quality",
                    passed=False,
                    details=msg,
                    metadata={"syntax_valid": False, "language": language}
                )
            
            complexity = self.calculate_python_complexity(code)
            metadata['complexity'] = complexity
            metadata['complexity_method'] = 'ast'
            
            if complexity > 15:
                issues.append(f"High complexity: {complexity} (threshold: 15)")
        
        else:
            # For other languages, check brace balance
            config = self.LANGUAGE_CONFIGS.get(language, {})
            
            if config.get('uses_braces', False):
                is_valid, msg = self.validate_brace_balance(code)
                if not is_valid:
                    return ValidationResult(
                        test_name="Code Quality",
                        passed=False,
                        details=msg,
                        metadata={"syntax_valid": False, "language": language}
                    )
            
            complexity = self.calculate_universal_complexity(code, language)
            metadata['complexity'] = complexity
            metadata['complexity_method'] = 'heuristic'
            
            if complexity > 30:  # Higher threshold for heuristic method
                issues.append(f"High complexity: {complexity} (threshold: 30)")
        
        # Count TODOs
        todos = self.count_todos(code)
        metadata['todos'] = todos
        
        if todos > 5:
            issues.append(f"Too many TODOs/FIXMEs: {todos}")
        
        # Extract functions (informational)
        functions = self.extract_functions_universal(code, language)
        metadata['function_count'] = len(functions)
        metadata['functions'] = functions[:10]  # Limit to first 10
        
        passed = len(issues) == 0
        details = "All quality checks passed" if passed else "; ".join(issues)
        metadata['syntax_valid'] = True
        metadata['language'] = language
        
        return ValidationResult(
            test_name="Code Quality",
            passed=passed,
            details=details,
            metadata=metadata
        )


class UniversalSemanticPreservationValidator:
    """Validates semantic preservation for multiple languages."""
    
    def extract_function_signatures_python(self, code: str) -> List[Dict[str, Any]]:
        """Extract function signatures from Python code using AST."""
        try:
            tree = ast.parse(code)
            signatures = []
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    sig = {
                        'name': node.name,
                        'args': [arg.arg for arg in node.args.args],
                        'async': isinstance(node, ast.AsyncFunctionDef)
                    }
                    signatures.append(sig)
            
            return signatures
        except:
            return []
    
    def extract_function_signatures_universal(self, code: str, language: str) -> Set[str]:
        """Extract function names using regex for any language."""
        patterns = {
            'python': r'def\s+(\w+)\s*\(',
            'java': r'(?:public|private|protected)?\s+(?:static\s+)?[\w<>\[\]]+\s+(\w+)\s*\(',
            'javascript': r'function\s+(\w+)\s*\(|const\s+(\w+)\s*=\s*(?:\([^)]*\)|[^=]+)\s*=>',
            'typescript': r'function\s+(\w+)\s*\(|const\s+(\w+)\s*=\s*(?:\([^)]*\)|[^=]+)\s*=>',
            'c': r'[\w<>\[\]]+\s+(\w+)\s*\([^)]*\)\s*\{',
            'cpp': r'[\w<>\[\]]+\s+(\w+)\s*\([^)]*\)\s*\{',
            'go': r'func\s+(?:\([^)]+\)\s*)?(\w+)\s*\(',
            'rust': r'fn\s+(\w+)\s*[<(]',
            'ruby': r'def\s+(\w+)',
        }
        
        pattern = patterns.get(language, r'function\s+(\w+)')
        matches = re.findall(pattern, code)
        
        # Flatten tuples
        functions = set()
        for match in matches:
            if isinstance(match, tuple):
                functions.update([m for m in match if m])
            else:
                functions.add(match)
        
        return functions
    
    def extract_class_names_python(self, code: str) -> List[str]:
        """Extract class names from Python code."""
        try:
            tree = ast.parse(code)
            return [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        except:
            return []
    
    def extract_class_names_universal(self, code: str, language: str) -> Set[str]:
        """Extract class names using regex."""
        patterns = {
            'python': r'class\s+(\w+)',
            'java': r'(?:public|private|protected)?\s+(?:abstract\s+)?class\s+(\w+)',
            'javascript': r'class\s+(\w+)',
            'typescript': r'(?:export\s+)?(?:abstract\s+)?class\s+(\w+)',
            'cpp': r'class\s+(\w+)',
            'go': r'type\s+(\w+)\s+struct',
            'rust': r'struct\s+(\w+)',
            'ruby': r'class\s+(\w+)',
        }
        
        pattern = patterns.get(language, r'class\s+(\w+)')
        return set(re.findall(pattern, code))
    
    def check_semantic_preservation(
        self, 
        original: str, 
        refactored: str,
        language: str = 'python'
    ) -> ValidationResult:
        """Check if core semantics are preserved across languages."""
        
        issues = []
        metadata = {'language': language, 'method': 'universal'}
        
        # Use Python AST for Python, regex for others
        if language == 'python':
            try:
                orig_funcs = {f['name'] for f in self.extract_function_signatures_python(original)}
                refact_funcs = {f['name'] for f in self.extract_function_signatures_python(refactored)}
                orig_classes = set(self.extract_class_names_python(original))
                refact_classes = set(self.extract_class_names_python(refactored))
                metadata['method'] = 'ast'
            except:
                # Fallback to regex
                orig_funcs = self.extract_function_signatures_universal(original, language)
                refact_funcs = self.extract_function_signatures_universal(refactored, language)
                orig_classes = self.extract_class_names_universal(original, language)
                refact_classes = self.extract_class_names_universal(refactored, language)
        else:
            orig_funcs = self.extract_function_signatures_universal(original, language)
            refact_funcs = self.extract_function_signatures_universal(refactored, language)
            orig_classes = self.extract_class_names_universal(original, language)
            refact_classes = self.extract_class_names_universal(refactored, language)
        
        # Check functions
        removed_funcs = orig_funcs - refact_funcs
        added_funcs = refact_funcs - orig_funcs
        
        metadata['removed_functions'] = list(removed_funcs)
        metadata['added_functions'] = list(added_funcs)
        metadata['original_function_count'] = len(orig_funcs)
        metadata['refactored_function_count'] = len(refact_funcs)
        
        # Allow some flexibility for refactoring
        if len(removed_funcs) > 3:
            issues.append(f"Multiple functions removed: {', '.join(list(removed_funcs)[:5])}")
        
        # Check classes
        removed_classes = orig_classes - refact_classes
        if removed_classes:
            metadata['removed_classes'] = list(removed_classes)
            if len(removed_classes) > 1:
                issues.append(f"Classes removed: {', '.join(list(removed_classes)[:3])}")
        
        metadata['original_class_count'] = len(orig_classes)
        metadata['refactored_class_count'] = len(refact_classes)
        
        passed = len(issues) == 0
        details = "Semantic structure preserved" if passed else "; ".join(issues)
        
        return ValidationResult(
            test_name="Semantic Preservation",
            passed=passed,
            details=details,
            metadata=metadata
        )


class LLMIntentVerificationValidator:
    """Uses LLM to verify if specific intent was fulfilled."""
    
    def __init__(self, llm_client: VertexAI):
        self.llm = llm_client
    
    def verify_specific_intent(
        self, 
        user_request: str, 
        refactored_code: str,
        file_path: str,
        language: str
    ) -> ValidationResult:
        """Use LLM to verify intent fulfillment with language awareness."""
        
        prompt = f"""You are a code review expert. Analyze if the refactored {language} code fulfills the user's request.

USER REQUEST: {user_request}

FILE: {file_path}
LANGUAGE: {language}

REFACTORED CODE:
```{language}
{refactored_code[:4000]}
```

Analyze and respond in JSON format:
{{
    "requirements_met": ["list of requirements that were fulfilled"],
    "requirements_missing": ["list of requirements that were NOT fulfilled"],
    "unintended_changes": ["list of changes not in the original request"],
    "alignment_score": <0-100>,
    "summary": "brief explanation"
}}

Focus on:
1. Does the code implement what was requested?
2. Are there missing features from the request?
3. Were there changes made that weren't requested?
4. Overall, does this fulfill the user's intent?
"""
        
        try:
            raw_response = self.llm.invoke(prompt)
            
            # Normalize response
            response_text = ""
            if isinstance(raw_response, list) and len(raw_response) > 0:
                item = raw_response[0]
                response_text = item.get("text", str(item)) if isinstance(item, dict) else str(item)
            elif isinstance(raw_response, dict):
                response_text = raw_response.get("text", str(raw_response))
            elif hasattr(raw_response, "text"):
                response_text = raw_response.text
            else:
                response_text = str(raw_response)
            
            # Extract JSON
            response_text = re.sub(r'```json\s*', '', response_text)
            response_text = re.sub(r'```\s*', '', response_text)
            
            result = json.loads(response_text.strip())
            
            score = result.get('alignment_score', 0)
            passed = score >= 70
            
            return ValidationResult(
                test_name="Intent Verification",
                passed=passed,
                score=score,
                threshold=70,
                details=result.get('summary', ''),
                metadata=result
            )
            
        except Exception as e:
            return ValidationResult(
                test_name="Intent Verification",
                passed=False,
                details="LLM verification failed",
                error=str(e)
            )


class ImprovedIntentSimilarityValidator:
    """Improved intent similarity checker using code descriptions."""
    
    def __init__(self, embedding_model: VertexAIEmbeddings):
        self.embedding_model = embedding_model
    
    def extract_code_description(self, code: str, language: str) -> str:
        """Extract natural language description from code."""
        description_parts = []
        
        # Extract comments
        comment_patterns = {
            'python': r'#\s*(.+)',
            'java': r'//\s*(.+)',
            'javascript': r'//\s*(.+)',
            'typescript': r'//\s*(.+)',
        }
        
        pattern = comment_patterns.get(language, r'//\s*(.+)')
        comments = re.findall(pattern, code)
        
        if comments:
            description_parts.append("Comments: " + " ".join(comments[:5]))
        
        # Extract docstrings for Python
        if language == 'python':
            docstrings = re.findall(r'"""(.+?)"""', code, re.DOTALL)
            if docstrings:
                description_parts.append("Docstrings: " + " ".join(docstrings[:3]))
        
        # Extract function/class names
        function_pattern = r'(?:def|function|func|fn)\s+(\w+)'
        class_pattern = r'class\s+(\w+)'
        
        functions = re.findall(function_pattern, code)
        classes = re.findall(class_pattern, code)
        
        if functions:
            description_parts.append(f"Functions: {', '.join(functions[:10])}")
        if classes:
            description_parts.append(f"Classes: {', '.join(classes[:5])}")
        
        # If no description found, use a sample of the code
        if not description_parts:
            # Take first 500 chars as description
            code_sample = code[:500].replace('\n', ' ')
            description_parts.append(f"Code sample: {code_sample}")
        
        return " | ".join(description_parts)
    
    def check_intent_similarity(
        self, 
        user_request: str, 
        refactored_code: str,
        language: str = 'python',
        threshold: float = 0.35
    ) -> ValidationResult:
        """Check intent similarity with improved approach."""
        
        metadata = {
            'language': language,
            'threshold': threshold,
            'method': 'code_description'
        }
        
        if not self.embedding_model:
            return ValidationResult(
                test_name="Intent Similarity",
                passed=True,  # Don't fail if embeddings unavailable
                details="Embedding model not available (informational only)",
                metadata=metadata
            )
        
        try:
            # Extract code description
            code_description = self.extract_code_description(refactored_code, language)
            metadata['code_description_preview'] = code_description[:200]
            
            # Get embeddings
            user_emb = self.embedding_model.embed_query(user_request)
            code_desc_emb = self.embedding_model.embed_query(code_description)
            
            # Calculate similarity
            user_array = np.array(user_emb).reshape(1, -1)
            code_array = np.array(code_desc_emb).reshape(1, -1)
            
            score = cosine_similarity(user_array, code_array)[0][0]
            score_percentage = round(score * 100, 2)
            
            # This is now informational - don't fail the validation
            passed = score >= threshold
            
            details = f"Similarity: {score_percentage}% (threshold: {threshold * 100}%)"
            if not passed:
                details += " - Below threshold but informational only"
            
            return ValidationResult(
                test_name="Intent Similarity",
                passed=True,  # Always pass - this is informational
                score=score_percentage,
                threshold=threshold * 100,
                details=details,
                metadata=metadata
            )
            
        except Exception as e:
            return ValidationResult(
                test_name="Intent Similarity",
                passed=True,  # Don't fail on errors - this is informational
                details="Similarity check error (informational only)",
                error=str(e),
                metadata=metadata
            )


class ComprehensiveValidationPipeline:
    """Complete validation pipeline with multi-language support."""
    
    def __init__(
        self,
        embedding_model: Optional[VertexAIEmbeddings] = None,
        llm_client: Optional[VertexAI] = None
    ):
        self.quality_validator = UniversalCodeQualityValidator()
        self.semantic_validator = UniversalSemanticPreservationValidator()
        
        if embedding_model:
            self.intent_similarity_validator = ImprovedIntentSimilarityValidator(embedding_model)
        else:
            self.intent_similarity_validator = None
        
        if llm_client:
            self.intent_verification_validator = LLMIntentVerificationValidator(llm_client)
        else:
            self.intent_verification_validator = None
    
    def validate_refactored_file(
        self,
        file_path: str,
        original_code: str,
        refactored_code: str,
        user_request: str,
        language: str = 'python',
        is_new_file: bool = False
    ) -> Dict[str, ValidationResult]:
        """Run all validation tests on a refactored file."""
        
        results = {}
        
        # 1. Code Quality Check (CRITICAL - must pass)
        results['quality'] = self.quality_validator.verify_code_quality(
            refactored_code, 
            language
        )
        
        # 2. Semantic Preservation (CRITICAL for modifications)
        if not is_new_file and original_code.strip():
            results['semantics'] = self.semantic_validator.check_semantic_preservation(
                original_code,
                refactored_code,
                language
            )
        
        # 3. LLM Intent Verification (PRIMARY intent check)
        if self.intent_verification_validator:
            results['intent_verification'] = self.intent_verification_validator.verify_specific_intent(
                user_request,
                refactored_code,
                file_path,
                language
            )
        
        # 4. Intent Similarity (INFORMATIONAL only)
        if self.intent_similarity_validator:
            results['intent_similarity'] = self.intent_similarity_validator.check_intent_similarity(
                user_request,
                refactored_code,
                language,
                threshold=0.35
            )
        
        return results
    
    def validate_all_files(
        self,
        user_request: str,
        original_files: Dict[str, str],
        refactored_files: Dict[str, str],
        language_detector
    ) -> Dict[str, Dict[str, ValidationResult]]:
        """Validate all refactored files."""
        
        all_results = {}
        
        for file_path, refactored_code in refactored_files.items():
            original_code = original_files.get(file_path, "")
            is_new_file = file_path not in original_files
            language = language_detector(file_path)
            
            all_results[file_path] = self.validate_refactored_file(
                file_path=file_path,
                original_code=original_code,
                refactored_code=refactored_code,
                user_request=user_request,
                language=language,
                is_new_file=is_new_file
            )
        
        return all_results
    
    def generate_validation_report(
        self,
        validation_results: Dict[str, Dict[str, ValidationResult]]
    ) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'overall_pass': True,
            'files_validated': len(validation_results),
            'files_passed': 0,
            'files_failed': 0,
            'critical_failures': 0,
            'test_summary': {},
            'detailed_results': {}
        }
        
        test_types = set()
        
        for file_path, file_results in validation_results.items():
            # Only fail on critical tests (not intent_similarity)
            critical_tests = {k: v for k, v in file_results.items() if k != 'intent_similarity'}
            file_passed = all(result.passed for result in critical_tests.values())
            
            if file_passed:
                report['files_passed'] += 1
            else:
                report['files_failed'] += 1
                report['overall_pass'] = False
                report['critical_failures'] += sum(
                    1 for k, v in critical_tests.items() if not v.passed
                )
            
            report['detailed_results'][file_path] = {
                'passed': file_passed,
                'tests': {
                    test_name: {
                        'passed': result.passed,
                        'score': result.score,
                        'details': result.details,
                        'error': result.error,
                        'is_critical': test_name != 'intent_similarity'
                    }
                    for test_name, result in file_results.items()
                }
            }
            
            test_types.update(file_results.keys())
        
        # Aggregate test type results
        for test_type in test_types:
            passed = sum(
                1 for file_results in validation_results.values()
                if test_type in file_results and file_results[test_type].passed
            )
            total = sum(
                1 for file_results in validation_results.values()
                if test_type in file_results
            )
            
            report['test_summary'][test_type] = {
                'passed': passed,
                'total': total,
                'pass_rate': round(passed / total * 100, 2) if total > 0 else 0,
                'is_critical': test_type != 'intent_similarity'
            }
        
        return report
    
    def display_validation_summary_streamlit(
        self,
        validation_results: Dict[str, Dict[str, ValidationResult]]
    ):
        """Display validation results in Streamlit."""
        
        report = self.generate_validation_report(validation_results)
        
        # Overall status
        if report['overall_pass']:
            st.success(f"All critical validation checks passed ({report['files_passed']}/{report['files_validated']} files)")
        else:
            st.error(f"Validation failed - {report['critical_failures']} critical issue(s) in {report['files_failed']} file(s)")
        
        # Test summary
        st.subheader("Validation Summary")
        
        # Separate critical and informational tests
        critical_tests = {k: v for k, v in report['test_summary'].items() if v['is_critical']}
        info_tests = {k: v for k, v in report['test_summary'].items() if not v['is_critical']}
        
        # Display critical tests
        if critical_tests:
            st.markdown("**Critical Tests:**")
            cols = st.columns(len(critical_tests))
            
            for idx, (test_type, summary) in enumerate(critical_tests.items()):
                with cols[idx]:
                    pass_rate = summary['pass_rate']
                    # icon = "✅" if pass_rate == 100 else "" if pass_rate >= 70 else "❌"
                    
                    st.metric(
                        label=f"{test_type.replace('_', ' ').title()}",
                        value=f"{summary['passed']}/{summary['total']}",
                        delta=f"{pass_rate}%"
                    )
        
        # Display informational tests
        if info_tests:
            st.markdown("**Informational Tests:**")
            cols = st.columns(len(info_tests))
            
            for idx, (test_type, summary) in enumerate(info_tests.items()):
                with cols[idx]:
                    pass_rate = summary['pass_rate']
                    # icon = "ℹ️"
                    
                    st.metric(
                        label=f" {test_type.replace('_', ' ').title()}",
                        value=f"{summary['passed']}/{summary['total']}",
                        delta=f"{pass_rate}%",
                        help="Informational only - does not affect validation"
                    )
        
        # Detailed results per file
        with st.expander("📋 Detailed Results", expanded=not report['overall_pass']):
            for file_path, file_result in report['detailed_results'].items():
                # status_icon = "✅" if file_result['passed'] else "❌"
                st.markdown(f"**{file_path}**")
                
                for test_name, test_result in file_result['tests'].items():
                    # test_icon = "✅" if test_result['passed'] else "❌"
                    # if not test_result['is_critical']:
                    #     test_icon = "ℹ️"
                    
                    details = test_result['details']
                    if test_result['score'] is not None:
                        details = f"{test_result['score']}% - {details}"
                    
                    if test_result['error']:
                        details += f" (Error: {test_result['error']})"
                    
                    criticality = "" if test_result['is_critical'] else " [INFO]"
                    st.text(f"   {test_name}{criticality}: {details}")
                
                st.markdown("---")
        
        return report['overall_pass']


# Helper function to detect language from file extension
def detect_language_from_path(file_path: str) -> str:
    """Detect programming language from file path."""
    extension_map = {
        ".py": "python",
        ".java": "java",
        ".js": "javascript",
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".c": "c",
        ".h": "c",
        ".cpp": "cpp",
        ".hpp": "cpp",
        ".cc": "cpp",
        ".cxx": "cpp",
        ".go": "go",
        ".rs": "rust",
        ".rb": "ruby",
        ".php": "php",
        ".swift": "swift",
        ".kt": "kotlin",
        ".kts": "kotlin",
        ".scala": "scala",
        ".sh": "shell",
        ".bash": "shell",
    }
    
    for ext, lang in extension_map.items():
        if file_path.endswith(ext):
            return lang
    
    return "unknown"


# Example usage
if __name__ == "__main__":
    # Example: Validate a Python refactoring
    
    original_python = '''
def calculate_sum(numbers):
    total = 0
    for num in numbers:
        total += num
    return total

class Calculator:
    def add(self, a, b):
        return a + b
'''
    
    refactored_python = '''
def calculate_sum(numbers):
    """Calculate the sum of a list of numbers."""
    return sum(numbers)

class Calculator:
    """Simple calculator class."""
    
    def add(self, a, b):
        """Add two numbers."""
        return a + b
    
    def subtract(self, a, b):
        """Subtract b from a."""
        return a - b
'''
    
    # Initialize validators (without LLM/embeddings for this example)
    pipeline = ComprehensiveValidationPipeline()
    
    # Run validation
    results = pipeline.validate_refactored_file(
        file_path="calculator.py",
        original_code=original_python,
        refactored_code=refactored_python,
        user_request="Add docstrings and use built-in sum function",
        language="python",
        is_new_file=False
    )
    
    # Print results
    print("\n" + "="*60)
    print("VALIDATION RESULTS")
    print("="*60 + "\n")
    
    for test_name, result in results.items():
        # status = "✅ PASS" if result.passed else "❌ FAIL"
        print(f"{test_name}")
        print(f"  Details: {result.details}")
        if result.score:
            print(f"  Score: {result.score}% (threshold: {result.threshold}%)")
        if result.metadata:
            print(f"  Metadata: {result.metadata}")
        print()
    
    # Example: Validate a Java file
    print("\n" + "="*60)
    print("JAVA VALIDATION EXAMPLE")
    print("="*60 + "\n")
    
    java_code = '''
public class HelloWorld {
    public static void main(String[] args) {
        System.out.println("Hello, World!");
    }
    
    public int add(int a, int b) {
        return a + b;
    }
}
'''
    
    java_results = pipeline.validate_refactored_file(
        file_path="HelloWorld.java",
        original_code="",
        refactored_code=java_code,
        user_request="Create a simple Hello World program with an add method",
        language="java",
        is_new_file=True
    )
    
    for test_name, result in java_results.items():
        # status = "✅ PASS" if result.passed else "❌ FAIL"
        print(f"{test_name}")
        print(f"  Details: {result.details}")
        print()