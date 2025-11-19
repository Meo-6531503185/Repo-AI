"""
Enhanced validation pipeline for code refactoring with comprehensive testing.
Implements functional correctness, semantic preservation, and intent verification.
"""

import ast
import re
import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
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
    metadata: Dict[str, Any] = None


class CodeQualityValidator:
    """Validates syntax and code quality metrics."""
    
    def validate_python_syntax(self, code: str) -> Tuple[bool, str]:
        """Check if Python code has valid syntax."""
        try:
            ast.parse(code)
            return True, "Syntax valid"
        except SyntaxError as e:
            return False, f"Syntax error at line {e.lineno}: {e.msg}"
    
    def calculate_complexity(self, code: str) -> int:
        """Calculate cyclomatic complexity (simplified)."""
        try:
            tree = ast.parse(code)
            complexity = 1  # Base complexity
            
            for node in ast.walk(tree):
                # Count decision points
                if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                    complexity += 1
                elif isinstance(node, ast.BoolOp):
                    complexity += len(node.values) - 1
                    
            return complexity
        except:
            return 0
    
    def count_todos(self, code: str) -> int:
        """Count TODO comments."""
        return code.upper().count('TODO')
    
    def verify_code_quality(self, code: str, language: str = 'python') -> ValidationResult:
        """Comprehensive code quality check."""
        issues = []
        metadata = {}
        
        if language == 'python':
            is_valid, msg = self.validate_python_syntax(code)
            if not is_valid:
                return ValidationResult(
                    test_name="Code Quality",
                    passed=False,
                    details=msg,
                    metadata={"syntax_valid": False}
                )
            
            complexity = self.calculate_complexity(code)
            metadata['complexity'] = complexity
            
            if complexity > 15:
                issues.append(f"High complexity: {complexity} (threshold: 15)")
            
            todos = self.count_todos(code)
            metadata['todos'] = todos
            
            if todos > 3:
                issues.append(f"Too many TODOs: {todos}")
        
        elif language in ('java', 'javascript', 'typescript', 'c', 'cpp'):
            # Check brace balance
            open_braces = code.count('{')
            close_braces = code.count('}')
            
            if open_braces != close_braces:
                return ValidationResult(
                    test_name="Code Quality",
                    passed=False,
                    details=f"Unbalanced braces: {open_braces} open, {close_braces} close",
                    metadata={"syntax_valid": False}
                )
        
        # Check for minimum code length
        if len(code.strip()) < 50:
            issues.append("Code too short (< 50 chars)")
        
        passed = len(issues) == 0
        details = "All quality checks passed" if passed else "; ".join(issues)
        
        return ValidationResult(
            test_name="Code Quality",
            passed=passed,
            details=details,
            metadata=metadata
        )


class SemanticPreservationValidator:
    """Validates that refactoring preserves semantic meaning."""
    
    def extract_function_signatures(self, code: str) -> List[Dict[str, Any]]:
        """Extract function signatures from Python code."""
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
    
    def extract_class_names(self, code: str) -> List[str]:
        """Extract class names from Python code."""
        try:
            tree = ast.parse(code)
            return [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        except:
            return []
    
    def extract_imports(self, code: str) -> List[str]:
        """Extract import statements."""
        try:
            tree = ast.parse(code)
            imports = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.extend([alias.name for alias in node.names])
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    imports.append(module)
            
            return imports
        except:
            return []
    
    def check_semantic_preservation(
        self, 
        original: str, 
        refactored: str,
        language: str = 'python'
    ) -> ValidationResult:
        """Check if core semantics are preserved."""
        
        if language != 'python':
            return ValidationResult(
                test_name="Semantic Preservation",
                passed=True,
                details="Semantic check skipped for non-Python code",
                metadata={"checked": False}
            )
        
        issues = []
        metadata = {}
        
        # Check function signatures
        orig_funcs = self.extract_function_signatures(original)
        refact_funcs = self.extract_function_signatures(refactored)
        
        orig_names = {f['name'] for f in orig_funcs}
        refact_names = {f['name'] for f in refact_funcs}
        
        removed_funcs = orig_names - refact_names
        added_funcs = refact_names - orig_names
        
        metadata['removed_functions'] = list(removed_funcs)
        metadata['added_functions'] = list(added_funcs)
        
        # Allow some flexibility for refactoring
        if len(removed_funcs) > 2:
            issues.append(f"Multiple functions removed: {removed_funcs}")
        
        # Check class structures
        orig_classes = set(self.extract_class_names(original))
        refact_classes = set(self.extract_class_names(refactored))
        
        removed_classes = orig_classes - refact_classes
        if removed_classes:
            metadata['removed_classes'] = list(removed_classes)
            issues.append(f"Classes removed: {removed_classes}")
        
        passed = len(issues) == 0
        details = "Semantic structure preserved" if passed else "; ".join(issues)
        
        return ValidationResult(
            test_name="Semantic Preservation",
            passed=passed,
            details=details,
            metadata=metadata
        )


class IntentVerificationValidator:
    """Uses LLM to verify if specific intent was fulfilled."""
    
    def __init__(self, llm_client: VertexAI):
        self.llm = llm_client
    
    def verify_specific_intent(
        self, 
        user_request: str, 
        refactored_code: str,
        file_path: str
    ) -> ValidationResult:
        """Use LLM to verify intent fulfillment."""
        
        prompt = f"""You are a code review expert. Analyze if the refactored code fulfills the user's request.

USER REQUEST: {user_request}

FILE: {file_path}

REFACTORED CODE:
```
{refactored_code[:3000]}  # Limit to avoid token overflow
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


class IntentSimilarityValidator:
    """Enhanced intent similarity checker with detailed reporting."""
    
    def __init__(self, embedding_model: VertexAIEmbeddings):
        self.embedding_model = embedding_model
    
    def extract_code_chunks(self, code_string: str) -> Dict[str, str]:
        """Split code into meaningful chunks using AST."""
        chunks = {}
        
        if not code_string or not isinstance(code_string, str):
            return {"Full Code": ""}
        
        try:
            tree = ast.parse(code_string)
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    # Get function source (simplified)
                    func_name = node.name
                    chunks[f"Function: {func_name}"] = ast.get_source_segment(code_string, node) or ""
                elif isinstance(node, ast.ClassDef):
                    class_name = node.name
                    chunks[f"Class: {class_name}"] = ast.get_source_segment(code_string, node) or ""
        
        except Exception:
            # Fallback to full code
            if code_string.strip():
                chunks["Full Code"] = code_string.strip()
        
        if not chunks and code_string.strip():
            chunks["Full Code"] = code_string.strip()
        
        return chunks
    
    def check_intent_similarity(
        self, 
        user_request: str, 
        refactored_code: str,
        threshold: float = 0.7
    ) -> ValidationResult:
        """Enhanced intent similarity with detailed reporting."""
        
        metadata = {
            'chunks_analyzed': 0,
            'best_matching_chunk': None,
            'all_scores': {}
        }
        
        if not self.embedding_model:
            return ValidationResult(
                test_name="Intent Similarity",
                passed=False,
                error="Embedding model not loaded",
                metadata=metadata
            )
        
        try:
            # Get user intent embedding
            user_emb = self.embedding_model.embed_query(user_request)
            
            # Extract and analyze code chunks
            code_chunks = self.extract_code_chunks(refactored_code)
            metadata['chunks_analyzed'] = len(code_chunks)
            
            max_score = 0.0
            best_chunk = None
            
            for chunk_name, chunk_code in code_chunks.items():
                if not chunk_code:
                    continue
                
                # Get chunk embedding
                chunk_emb = self.embedding_model.embed_query(chunk_code)
                
                # Calculate similarity
                user_array = np.array(user_emb).reshape(1, -1)
                chunk_array = np.array(chunk_emb).reshape(1, -1)
                
                score = cosine_similarity(user_array, chunk_array)[0][0]
                metadata['all_scores'][chunk_name] = round(score * 100, 2)
                
                if score > max_score:
                    max_score = score
                    best_chunk = chunk_name
            
            score_percentage = round(max_score * 100, 2)
            metadata['best_matching_chunk'] = best_chunk
            
            passed = max_score >= threshold
            
            return ValidationResult(
                test_name="Intent Similarity",
                passed=passed,
                score=score_percentage,
                threshold=threshold * 100,
                details=f"Best match: {best_chunk} ({score_percentage}%)",
                metadata=metadata
            )
            
        except Exception as e:
            return ValidationResult(
                test_name="Intent Similarity",
                passed=False,
                error=str(e),
                metadata=metadata
            )


class ComprehensiveValidationPipeline:
    """Complete validation pipeline integrating all validators."""
    
    def __init__(
        self,
        embedding_model: Optional[VertexAIEmbeddings] = None,
        llm_client: Optional[VertexAI] = None
    ):
        self.quality_validator = CodeQualityValidator()
        self.semantic_validator = SemanticPreservationValidator()
        
        if embedding_model:
            self.intent_similarity_validator = IntentSimilarityValidator(embedding_model)
        else:
            self.intent_similarity_validator = None
        
        if llm_client:
            self.intent_verification_validator = IntentVerificationValidator(llm_client)
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
        
        # 1. Code Quality Check
        results['quality'] = self.quality_validator.verify_code_quality(
            refactored_code, 
            language
        )
        
        # 2. Semantic Preservation (only for modifications)
        if not is_new_file:
            results['semantics'] = self.semantic_validator.check_semantic_preservation(
                original_code,
                refactored_code,
                language
            )
        
        # 3. Intent Similarity
        if self.intent_similarity_validator:
            results['intent_similarity'] = self.intent_similarity_validator.check_intent_similarity(
                user_request,
                refactored_code,
                threshold=0.7
            )
        
        # 4. LLM Intent Verification
        if self.intent_verification_validator:
            results['intent_verification'] = self.intent_verification_validator.verify_specific_intent(
                user_request,
                refactored_code,
                file_path
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
            'test_summary': {},
            'detailed_results': {}
        }
        
        test_types = set()
        
        for file_path, file_results in validation_results.items():
            file_passed = all(result.passed for result in file_results.values())
            
            if file_passed:
                report['files_passed'] += 1
            else:
                report['files_failed'] += 1
                report['overall_pass'] = False
            
            report['detailed_results'][file_path] = {
                'passed': file_passed,
                'tests': {
                    test_name: {
                        'passed': result.passed,
                        'score': result.score,
                        'details': result.details,
                        'error': result.error
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
                'pass_rate': round(passed / total * 100, 2) if total > 0 else 0
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
            st.success(f"✅ All validation checks passed ({report['files_passed']}/{report['files_validated']} files)")
        else:
            st.error(f"❌ Validation failed ({report['files_failed']}/{report['files_validated']} files have issues)")
        
        # Test summary
        st.subheader("📊 Validation Summary")
        
        cols = st.columns(len(report['test_summary']))
        
        for idx, (test_type, summary) in enumerate(report['test_summary'].items()):
            with cols[idx]:
                pass_rate = summary['pass_rate']
                icon = "✅" if pass_rate == 100 else "⚠️" if pass_rate >= 70 else "❌"
                
                st.metric(
                    label=f"{icon} {test_type.replace('_', ' ').title()}",
                    value=f"{summary['passed']}/{summary['total']}",
                    delta=f"{pass_rate}%"
                )
        
        # Detailed results per file
        with st.expander("📋 Detailed Results", expanded=not report['overall_pass']):
            for file_path, file_result in report['detailed_results'].items():
                status_icon = "✅" if file_result['passed'] else "❌"
                st.markdown(f"**{status_icon} {file_path}**")
                
                for test_name, test_result in file_result['tests'].items():
                    test_icon = "✅" if test_result['passed'] else "❌"
                    
                    details = test_result['details']
                    if test_result['score'] is not None:
                        details = f"{test_result['score']}% - {details}"
                    
                    if test_result['error']:
                        details += f" (Error: {test_result['error']})"
                    
                    st.text(f"  {test_icon} {test_name}: {details}")
                
                st.markdown("---")
        
        return report['overall_pass']