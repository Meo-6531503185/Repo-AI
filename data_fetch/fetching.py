import requests
import streamlit as st


def fetch_file_content(file_url):
    """Fetch the content of a single file."""
    file_response = requests.get(file_url)
    if file_response.status_code == 200:
        return file_response.text
    else:
        st.warning(f"Failed to retrieve file: {file_url}")
        return ""
    

def fetch_repo_contents(api_url, headers):
    """Fetch the contents of the entire repository."""
    response = requests.get(api_url, headers=headers)
    
    if response.status_code != 200:
        st.error(f"Failed to retrieve files from GitHub. Status code: {response.status_code}")
        return []

    return response.json()
    

def fetch_all_files_from_repo(api_url, headers):
    """Fetch all files from a GitHub repository, including files inside subfolders."""
    all_files = []  # This will hold the paths of all files in the repository
    repo_contents = fetch_repo_contents(api_url, headers)

    for file_info in repo_contents:
        if file_info["type"] == "file":
            # For files, add them to the list
            all_files.append({
                "name": file_info["name"],
                "url": file_info["download_url"],
                "path": file_info["path"]
            })
        elif file_info["type"] == "dir":
            # If it's a directory, recurse into it
            subfolder_url = file_info["url"]
            all_files += fetch_all_files_from_repo(subfolder_url, headers)

    return all_files
