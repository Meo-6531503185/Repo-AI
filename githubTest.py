from __future__ import annotations
import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional
import requests
from langchain_core.utils import get_from_dict_or_env
from pydantic import BaseModel, ConfigDict, model_validator

if TYPE_CHECKING:
    from github.Issue import Issue
    from github.PullRequest import PullRequest

LANGSMITH_ENDPOINT = "https://api.smith.langchain.com"
LANGSMITH_PROJECT = "pr-puzzled-utilisation-86"

def _import_tiktoken() -> Any:
    """Import tiktoken."""
    try:
        import tiktoken
    except ImportError:
        raise ImportError(
            "tiktoken is not installed. "
            "Please install it with `pip install tiktoken`"
        )
    return tiktoken

class GitHubAPIWrapper(BaseModel):
    """Wrapper for GitHub API."""

    github: Any = None  #: :meta private:
    github_repo_instance: Any = None  #: :meta private:
    github_repository: Optional[str] = None
    github_app_id: Optional[str] = None
    github_app_private_key: Optional[str] = None
    active_branch: Optional[str] = None
    github_base_branch: Optional[str] = None

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Any:
        """Validate that api key and python package exists in environment."""
        github_repository = get_from_dict_or_env(values, "github_repository", "GITHUB_REPOSITORY")
        github_app_id = get_from_dict_or_env(values, "github_app_id", "GITHUB_APP_ID")
        github_app_private_key = get_from_dict_or_env(values, "github_app_private_key", "GITHUB_APP_PRIVATE_KEY")

        try:
            from github import Auth, GithubIntegration
        except ImportError:
            raise ImportError(
                "PyGithub is not installed. "
                "Please install it with `pip install PyGithub`"
            )

        try:
            # interpret the key as a file path, fallback to interpreting as the key itself
            with open(github_app_private_key, "r") as f:
                private_key = f.read()
        except Exception:
            private_key = github_app_private_key

        auth = Auth.AppAuth(github_app_id, private_key)
        gi = GithubIntegration(auth=auth)
        installation = gi.get_installations()

        if not installation:
            raise ValueError(
                f"Please make sure to install the created github app with id "
                f"{github_app_id} on the repo: {github_repository}"
                "More instructions can be found at "
                "https://docs.github.com/en/apps/using-"
                "github-apps/installing-your-own-github-app"
            )

        try:
            installation = installation[0]
        except ValueError as e:
            raise ValueError(
                "Please make sure to give correct github parameters "
                f"Error message: {e}"
            )

        # create a GitHub instance:
        g = installation.get_github_for_installation()
        repo = g.get_repo(github_repository)

        github_base_branch = get_from_dict_or_env(
            values, "github_base_branch", "GITHUB_BASE_BRANCH", default=repo.default_branch
        )

        active_branch = get_from_dict_or_env(
            values, "active_branch", "ACTIVE_BRANCH", default=repo.default_branch
        )

        values["github"] = g
        values["github_repo_instance"] = repo
        values["github_repository"] = github_repository
        values["github_app_id"] = github_app_id
        values["github_app_private_key"] = github_app_private_key
        values["active_branch"] = active_branch
        values["github_base_branch"] = github_base_branch

        return values

    def list_files_in_main_branch(self) -> str:
        """
        Fetches all files in the main branch of the repo.

        Returns:
            str: A plaintext report containing the paths and names of the files.
        """
        files: List[str] = []
        try:
            contents = self.github_repo_instance.get_contents("", ref=self.github_base_branch)
            for content in contents:
                if content.type == "dir":
                    files.extend(self._list_files(content.path))
                else:
                    files.append(content.path)

            if files:
                files_str = "\n".join(files)
                return f"Found {len(files)} files in the main branch:\n{files_str}"
            else:
                return "No files found in the main branch"
        except Exception as e:
            return f"Error occurred while listing files: {str(e)}"

    def _list_files(self, directory_path: str) -> List[str]:
        files: List[str] = []
        contents = self.github_repo_instance.get_contents(directory_path, ref=self.active_branch)
        for content in contents:
            if content.type == "dir":
                files.extend(self._list_files(content.path))
            else:
                files.append(content.path)
        return files

    def parse_issues(self, issues: List[Issue]) -> List[dict]:
        """Extracts title and number from each Issue and puts them in a dictionary."""
        parsed = []
        for issue in issues:
            title = issue.title
            number = issue.number
            opened_by = issue.user.login if issue.user else None
            issue_dict = {"title": title, "number": number}
            if opened_by is not None:
                issue_dict["opened_by"] = opened_by
            parsed.append(issue_dict)
        return parsed

    def parse_pull_requests(self, pull_requests: List[PullRequest]) -> List[dict]:
        """Extracts title and number from each Pull Request and puts them in a dictionary."""
        parsed = []
        for pr in pull_requests:
            parsed.append(
                {
                    "title": pr.title,
                    "number": pr.number,
                    "commits": str(pr.commits),
                    "comments": str(pr.comments),
                }
            )
        return parsed

    def create_branch(self, proposed_branch_name: str) -> str:
        """Create a new branch, and set it as the active bot branch."""
        from github import GithubException

        i = 0
        new_branch_name = proposed_branch_name
        base_branch = self.github_repo_instance.get_branch(self.github_repo_instance.default_branch)

        for i in range(1000):
            try:
                self.github_repo_instance.create_git_ref(ref=f"refs/heads/{new_branch_name}", sha=base_branch.commit.sha)
                self.active_branch = new_branch_name
                return f"Branch '{new_branch_name}' created successfully and set as current active branch."
            except GithubException as e:
                if e.status == 422 and "Reference already exists" in e.data["message"]:
                    i += 1
                    new_branch_name = f"{proposed_branch_name}_v{i}"
                else:
                    raise Exception(f"Unable to create branch name from proposed branch name: {proposed_branch_name}")

        return f"Unable to create branch after 1000 attempts with derived names from {proposed_branch_name}"

    def create_file(self, file_query: str) -> str:
        """Create a new file on the repo."""
        if self.active_branch == self.github_base_branch:
            return (
                f"You're attempting to commit directly to the {self.github_base_branch} branch, which is protected. "
                "Please create a new branch and try again."
            )

        file_path = file_query.split("\n")[0]
        file_contents = file_query[len(file_path) + 2 :]

        try:
            try:
                file = self.github_repo_instance.get_contents(file_path, ref=self.active_branch)
                if file:
                    return f"File already exists at `{file_path}` on branch `{self.active_branch}`. Use `update_file` to modify it."
            except Exception:
                pass  # expected behavior, file shouldn't exist yet

            self.github_repo_instance.create_file(
                path=file_path,
                message="Create " + file_path,
                content=file_contents,
                branch=self.active_branch,
            )
            return f"Created file {file_path}"
        except Exception as e:
            return f"Unable to create file due to error: {str(e)}"

    def read_file(self, file_path: str) -> str:
        """Read a file from this agent's branch."""
        try:
            file = self.github_repo_instance.get_contents(file_path, ref=self.active_branch)
            return file.decoded_content.decode("utf-8")
        except Exception as e:
            return f"File not found `{file_path}` on branch `{self.active_branch}`. Error: {str(e)}"

    def update_file(self, file_query: str) -> str:
        """Updates a file with new content."""
        if self.active_branch == self.github_base_branch:
            return (
                f"You're attempting to commit directly to the {self.github_base_branch} branch, which is protected. "
                "Please create a new branch and try again."
            )

        try:
            file_path = file_query.split("\n")[0]
            old_file_contents = file_query.split("OLD <<<<")[1].split(">>>> OLD")[0].strip()
            new_file_contents = file_query.split("NEW <<<<")[1].split(">>>> NEW")[0].strip()

            file_content = self.read_file(file_path)
            updated_file_content = file_content.replace(old_file_contents, new_file_contents)

            if file_content == updated_file_content:
                return "File content was not updated because old content was not found."

            self.github_repo_instance.update_file(
                path=file_path,
                message="Update " + str(file_path),
                content=updated_file_content,
                branch=self.active_branch,
                sha=self.github_repo_instance.get_contents(file_path, ref=self.active_branch).sha,
            )
            return f"Updated file {file_path}"
        except Exception as e:
            return f"Unable to update file due to error: {str(e)}"

    def delete_file(self, file_path: str) -> str:
        """Deletes a file from the repo."""
        if self.active_branch == self.github_base_branch:
            return (
                f"You're attempting to commit directly to the {self.github_base_branch} branch, which is protected. "
                "Please create a new branch and try again."
            )

        try:
            self.github_repo_instance.delete_file(
                path=file_path,
                message="Delete " + file_path,
                branch=self.active_branch,
                sha=self.github_repo_instance.get_contents(file_path, ref=self.active_branch).sha,
            )
            return f"Deleted file {file_path}"
        except Exception as e:
            return f"Unable to delete file due to error: {str(e)}"
import os
from github import Github
from github.GithubException import UnknownObjectException
# Load environment variables
GITHUB_APP_ID = os.getenv("GITHUB_APP_ID")
GITHUB_PRIVATE_KEY_PATH = os.getenv("GITHUB_PRIVATE_KEY_PATH")

# Read the private key from file
def load_private_key():
    try:
        with open(GITHUB_PRIVATE_KEY_PATH, "r") as key_file:
            return key_file.read()
    except FileNotFoundError:
        print(f"Error: Private key file not found at {GITHUB_PRIVATE_KEY_PATH}")
        return None
    except Exception as e:
        print(f"Error loading private key: {str(e)}")
        return None

# Initialize the GitHubAPIWrapper with the necessary information
def initialize_github_connection(github_repository):
    private_key = load_private_key()
    if not private_key:
        raise Exception("Private key not loaded correctly.")

    # Pass the user-provided repository URL dynamically
    wrapper = GitHubAPIWrapper(
        github_repository=github_repository,
        github_app_id=GITHUB_APP_ID,
        github_app_private_key=private_key
    )

    return wrapper

# Fetch repository information
def fetch_repo_info(wrapper):
    try:
        # Dynamically fetch repository based on the user-provided name
        repo = wrapper.github.get_repo(wrapper.github_repository)
        print("Repository:", repo.name)

        issues = repo.get_issues(state="open")
        print("Open Issues:", wrapper.parse_issues(issues))

        pull_requests = repo.get_pulls()
        print("Pull Requests:", wrapper.parse_pull_requests(pull_requests))

        files_in_main_branch = wrapper.list_files_in_main_branch()
        print("Files in Main Branch:", files_in_main_branch)

    except UnknownObjectException as e:
        print(f"Error: Repository or object not found. {e}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

# Main execution flow
if __name__ == "__main__":
    # Replace this with the repository URL that the user provides
    user_repo_input = input("Enter your GitHub repository (e.g., owner/repo): ")
    
    try:
        wrapper = initialize_github_connection(user_repo_input)
        print("GitHub connection established successfully!")
        fetch_repo_info(wrapper)
    except Exception as e:
        print(f"Error initializing GitHub connection: {str(e)}")

