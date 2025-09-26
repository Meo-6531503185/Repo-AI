import requests
import streamlit as st
from processing import *


def fetch_repo_contents(api_url, headers):
   """Fetch the contents of the entire repository."""
   response = requests.get(api_url, headers=headers)
   if response.status_code != 200:
       st.error(f"Failed to retrieve files from GitHub. Status code: {response.status_code}")
       return []
   return response.json()


def fetch_all_files_from_repo(api_url, headers):
   """Fetch all files from a GitHub repository, including files inside subfolders."""
   all_files = []
   repo_contents = fetch_repo_contents(api_url, headers)
   for file_info in repo_contents:
       if file_info["type"] == "file":
           all_files.append({
               "name": file_info["name"],
               "download_url": file_info["download_url"],  # Using download_url directly
               "path": file_info["path"]
           })
       elif file_info["type"] == "dir":
           # If it's a directory, recurse into it
           subfolder_url = file_info["url"]
           all_files += fetch_all_files_from_repo(subfolder_url, headers)
   return all_files


def fetch_file_content(file_url):
   """Fetch the content of a single file."""
   response = requests.get(file_url)
   if response.status_code == 200:
       content = response.text
       return content
   else:
       print(f"Failed to retrieve file: {file_url}, Status Code: {response.status_code}")
       return ""
   
def read_all_repo_files(github_url):
   """Read all files in a repository and store them for later queries."""
   # Extract the owner and repo from the URL
   parts = github_url.rstrip("/").split("/")
   owner, repo = parts[-2], parts[-1]
   # GitHub API URL for repository contents
   api_url = f"https://api.github.com/repos/{owner}/{repo}/contents"
   # Load GitHub App private key
  
   installation_token = get_installation_token()
   headers = {
           "Authorization": f"token {installation_token}",
           "Accept": "application/vnd.github+json"
       }
   # Fetch all files from the repository
   all_files = fetch_all_files_from_repo(api_url, headers)
   # Prepare data and chunking
   repo_data = {}
   all_file_chunks = []
   for file in all_files:
       file_content = fetch_file_content(file.get("download_url"))
      
       if not isinstance(file_content, str):
           st.write(f"ERROR: Expected string, got {type(file_content)} for file {file['path']}")
           continue  # Skip this file if it's not a string
       chunks = get_text_chunks(file_content)  # Assuming get_text_chunks exists
       all_file_chunks.extend(chunks)
       repo_data[file["path"]] = file_content
   # Generate vector store from chunks
   vector_store = get_vector_store(all_file_chunks)
   return repo_data, vector_store


  