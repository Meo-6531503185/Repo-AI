"""
Comprehensive error handling system.
Create this file at: utils/error_handlers.py
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, List
import streamlit as st
import traceback


class ErrorCategory(Enum):
    """Categories of errors for better UX."""
    GITHUB_API = "github_api"
    GITHUB_AUTH = "github_auth"
    RATE_LIMIT = "rate_limit"
    VALIDATION = "validation"
    LLM_ERROR = "llm_error"
    NETWORK = "network"
    UNKNOWN = "unknown"


@dataclass
class UserFriendlyError:
    """Structured error information for display."""
    category: ErrorCategory
    title: str
    message: str
    technical_details: str
    suggested_actions: List[str]
    can_retry: bool = False
    docs_link: Optional[str] = None


class ErrorHandler:
    """Centralized error handling with user-friendly messages."""
    
    def __init__(self):
        self.error_messages = {
            ErrorCategory.GITHUB_AUTH: {
                "title": "🔐 GitHub Authentication Failed",
                "default_message": "Unable to authenticate with GitHub",
                "actions": [
                    "Verify your GitHub App ID is correct",
                    "Ensure the private key file exists and is readable",
                    "Check that the app is installed on the repository",
                    "Confirm the app has the required permissions"
                ],
                "docs": "https://docs.github.com/en/apps/creating-github-apps/authenticating-with-a-github-app"
            },
            ErrorCategory.RATE_LIMIT: {
                "title": "⏱️ GitHub Rate Limit Exceeded",
                "default_message": "Too many requests to GitHub API",
                "actions": [
                    "Wait a few minutes before trying again",
                    "Check your rate limit status at https://api.github.com/rate_limit",
                    "Consider upgrading your GitHub plan for higher limits"
                ],
                "docs": "https://docs.github.com/en/rest/overview/rate-limits-for-the-rest-api"
            },
            ErrorCategory.GITHUB_API: {
                "title": "🐙 GitHub API Error",
                "default_message": "An error occurred while communicating with GitHub",
                "actions": [
                    "Check your internet connection",
                    "Verify the repository name is correct",
                    "Ensure you have the required permissions",
                    "Try again in a few moments"
                ],
                "docs": "https://docs.github.com/en/rest"
            },
            ErrorCategory.VALIDATION: {
                "title": "⚠️ Code Validation Failed",
                "default_message": "The refactored code has validation errors",
                "actions": [
                    "Review the validation errors above",
                    "Try refining your refactoring request",
                    "Check if the original code had similar issues"
                ],
                "docs": None
            },
            ErrorCategory.LLM_ERROR: {
                "title": "🤖 AI Model Error",
                "default_message": "The AI model encountered an error",
                "actions": [
                    "Try rephrasing your request",
                    "Break down complex requests into smaller steps",
                    "Check if the code is too large (try fewer files)"
                ],
                "docs": None
            }
        }
    
    def handle_github_auth_error(self, exception: Exception) -> UserFriendlyError:
        """Handle GitHub authentication errors."""
        error_info = self.error_messages[ErrorCategory.GITHUB_AUTH]
        
        technical = str(exception)
        message = error_info["default_message"]
        
        if "installation" in technical.lower():
            message = "GitHub App is not installed on the repository"
        elif "private key" in technical.lower():
            message = "Invalid or missing GitHub App private key"
        elif "app id" in technical.lower():
            message = "Invalid GitHub App ID"
        
        return UserFriendlyError(
            category=ErrorCategory.GITHUB_AUTH,
            title=error_info["title"],
            message=message,
            technical_details=technical,
            suggested_actions=error_info["actions"],
            can_retry=False,
            docs_link=error_info["docs"]
        )
    
    def handle_rate_limit_error(self, exception: Exception) -> UserFriendlyError:
        """Handle rate limit errors."""
        error_info = self.error_messages[ErrorCategory.RATE_LIMIT]
        
        return UserFriendlyError(
            category=ErrorCategory.RATE_LIMIT,
            title=error_info["title"],
            message=error_info["default_message"],
            technical_details=str(exception),
            suggested_actions=error_info["actions"],
            can_retry=True,
            docs_link=error_info["docs"]
        )
    
    def handle_validation_error(self, validation_results: list) -> UserFriendlyError:
        """Handle code validation errors."""
        error_info = self.error_messages[ErrorCategory.VALIDATION]
        
        total_errors = sum(len(r.errors) for r in validation_results)
        message = f"Found {total_errors} validation error(s) across {len(validation_results)} file(s)"
        
        return UserFriendlyError(
            category=ErrorCategory.VALIDATION,
            title=error_info["title"],
            message=message,
            technical_details="See validation details above",
            suggested_actions=error_info["actions"],
            can_retry=True,
            docs_link=error_info["docs"]
        )
    
    def handle_llm_error(self, exception: Exception) -> UserFriendlyError:
        """Handle LLM/AI model errors."""
        error_info = self.error_messages[ErrorCategory.LLM_ERROR]
        
        technical = str(exception)
        message = error_info["default_message"]
        
        if "quota" in technical.lower() or "limit" in technical.lower():
            message = "API quota exceeded for the AI model"
        elif "timeout" in technical.lower():
            message = "AI model request timed out"
        elif "context" in technical.lower() or "token" in technical.lower():
            message = "Input is too large for the AI model"
        
        return UserFriendlyError(
            category=ErrorCategory.LLM_ERROR,
            title=error_info["title"],
            message=message,
            technical_details=technical,
            suggested_actions=error_info["actions"],
            can_retry=True,
            docs_link=error_info["docs"]
        )
    
    def display_error(self, error: UserFriendlyError):
        """Display error in Streamlit UI with helpful formatting."""
        st.error(f"### {error.title}")
        st.write(error.message)
        
        if error.suggested_actions:
            st.write("**💡 Suggested Actions:**")
            for action in error.suggested_actions:
                st.write(f"- {action}")
        
        with st.expander("🔍 Technical Details"):
            st.code(error.technical_details, language="text")
            st.write("**Full Traceback:**")
            st.code(traceback.format_exc(), language="text")
        
        if error.docs_link:
            st.info(f"📚 [Read the documentation]({error.docs_link})")
        
        if error.can_retry:
            st.warning("You can try again after addressing the issue above.")
    
    def wrap_operation(self, operation_name: str, operation_func, *args, **kwargs):
        """
        Wrapper for operations that handles errors gracefully.
        
        Usage:
            result = error_handler.wrap_operation(
                "GitHub Push",
                push_agent.run,
                response, repo_name, commit_msg
            )
        """
        try:
            return operation_func(*args, **kwargs)
        except Exception as e:
            # Try to categorize the error
            error_str = str(e).lower()
            
            if "rate limit" in error_str:
                error = self.handle_rate_limit_error(e)
            elif any(x in error_str for x in ["auth", "permission", "token"]):
                error = self.handle_github_auth_error(e)
            elif "validation" in error_str:
                error = UserFriendlyError(
                    category=ErrorCategory.VALIDATION,
                    title="Validation Error",
                    message=str(e),
                    technical_details=traceback.format_exc(),
                    suggested_actions=["Check the validation results above"],
                    can_retry=True
                )
            else:
                # Generic error
                error = UserFriendlyError(
                    category=ErrorCategory.UNKNOWN,
                    title=f"❌ {operation_name} Failed",
                    message="An unexpected error occurred",
                    technical_details=str(e),
                    suggested_actions=[
                        "Check the technical details below",
                        "Try again with a simpler request",
                        "Report this issue if it persists"
                    ],
                    can_retry=True
                )
            
            self.display_error(error)
            return None