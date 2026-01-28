"""
LLM output normalization utilities.
Create this file at: utils/llm_normalizer.py
"""

import re
from dataclasses import dataclass
from typing import Any, Optional, Dict


@dataclass
class NormalizedOutput:
    """Normalized LLM output with metadata."""
    text: str
    is_normalized: bool
    original_type: str
    metadata: Dict[str, Any]


class LLMOutputNormalizer:
    """Normalizes various LLM output formats to plain text."""
    
    def __init__(self, provider: str = "vertex_ai"):
        """
        Args:
            provider: LLM provider name (vertex_ai, openai, anthropic)
        """
        self.provider = provider
    
    def normalize(self, raw_output: Any) -> NormalizedOutput:
        """
        Normalize LLM output to plain text.
        
        Handles:
        - List responses
        - Dict responses
        - Object responses with .text attribute
        - Plain string responses
        """
        metadata = {"provider": self.provider}
        original_type = type(raw_output).__name__
        
        try:
            # Case 1: Already a string
            if isinstance(raw_output, str):
                return NormalizedOutput(
                    text=raw_output,
                    is_normalized=True,
                    original_type=original_type,
                    metadata=metadata
                )
            
            # Case 2: List (common with Vertex AI)
            if isinstance(raw_output, list):
                if len(raw_output) == 0:
                    return NormalizedOutput(
                        text="",
                        is_normalized=True,
                        original_type=original_type,
                        metadata={**metadata, "empty_list": True}
                    )
                
                # Try to extract text from first item
                first_item = raw_output[0]
                
                if isinstance(first_item, dict):
                    text = first_item.get("text") or first_item.get("output_text") or str(first_item)
                elif hasattr(first_item, "text"):
                    text = first_item.text
                else:
                    text = str(first_item)
                
                return NormalizedOutput(
                    text=text,
                    is_normalized=True,
                    original_type=original_type,
                    metadata={**metadata, "list_length": len(raw_output)}
                )
            
            # Case 3: Dict
            if isinstance(raw_output, dict):
                # Try common keys
                for key in ["text", "output_text", "content", "message"]:
                    if key in raw_output:
                        return NormalizedOutput(
                            text=raw_output[key],
                            is_normalized=True,
                            original_type=original_type,
                            metadata={**metadata, "extracted_key": key}
                        )
                
                # Fallback to string representation
                return NormalizedOutput(
                    text=str(raw_output),
                    is_normalized=False,
                    original_type=original_type,
                    metadata={**metadata, "warning": "Used dict string representation"}
                )
            
            # Case 4: Object with .text attribute
            if hasattr(raw_output, "text"):
                return NormalizedOutput(
                    text=raw_output.text,
                    is_normalized=True,
                    original_type=original_type,
                    metadata=metadata
                )
            
            # Case 5: Fallback
            return NormalizedOutput(
                text=str(raw_output),
                is_normalized=False,
                original_type=original_type,
                metadata={**metadata, "warning": "Used fallback string conversion"}
            )
            
        except Exception as e:
            return NormalizedOutput(
                text="",
                is_normalized=False,
                original_type=original_type,
                metadata={**metadata, "error": str(e)}
            )


class CodeExtractor:
    """Extract and clean code from LLM responses."""
    
    @staticmethod
    def remove_markdown_fences(text: str) -> str:
        """Remove markdown code fences (```language ... ```)."""
        # Remove opening fence with optional language
        text = re.sub(r'^```[\w]*\s*\n', '', text, flags=re.MULTILINE)
        # Remove closing fence
        text = re.sub(r'\n```\s*$', '', text, flags=re.MULTILINE)
        return text.strip()
    
    @staticmethod
    def remove_ai_preamble(text: str) -> str:
        """
        Remove common AI preambles like:
        - "Here's the refactored code:"
        - "I've made the following changes:"
        """
        preamble_patterns = [
            r'^(?:Here\'s|Here is|I\'ve|I have).*?:?\s*\n',
            r'^(?:The refactored|The updated|The modified).*?:?\s*\n',
            r'^(?:Below is|Following is).*?:?\s*\n',
        ]
        
        for pattern in preamble_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.MULTILINE)
        
        return text.strip()
    
    @staticmethod
    def extract_code_blocks(text: str) -> list:
        """Extract all code blocks from markdown-formatted text."""
        pattern = r'```[\w]*\s*\n(.*?)\n```'
        matches = re.findall(pattern, text, re.DOTALL)
        return [match.strip() for match in matches]
    
    @staticmethod
    def clean_for_github(code: str) -> str:
        """
        Clean code for GitHub commit.
        - Normalize line endings
        - Remove trailing whitespace
        - Ensure final newline
        """
        # Normalize line endings to \n
        code = code.replace('\r\n', '\n').replace('\r', '\n')
        
        # Remove trailing whitespace from each line
        lines = code.split('\n')
        lines = [line.rstrip() for line in lines]
        
        # Join and ensure final newline
        code = '\n'.join(lines)
        if not code.endswith('\n'):
            code += '\n'
        
        return code