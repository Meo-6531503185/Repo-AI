# test_github_permissions.py
from githubTest import GitHubAPIWrapper
import os
from dotenv import load_dotenv

load_dotenv()

wrapper = GitHubAPIWrapper(
    github_repository="moepyaePK/BinarySearch",
    github_app_id=os.getenv("GITHUB_APP_ID"),
    github_app_private_key=os.getenv("GITHUB_PRIVATE_KEY")
)

print("Testing GitHub App Permissions...")
print(f"Repository: {wrapper.github_repository}")
print(f"Base branch: {wrapper.github_base_branch}")

# Test 1: Read access
try:
    files = wrapper.list_files_in_main_branch()
    print("✅ Read access: OK")
except Exception as e:
    print(f"❌ Read access failed: {e}")

# Test 2: Branch creation
try:
    result = wrapper.create_branch("test-permissions-branch")
    if "Error" not in result and "Unable" not in result:
        print("✅ Branch creation: OK")
        # Clean up
        wrapper.delete_branch("test-permissions-branch")
        print("✅ Branch deletion: OK")
    else:
        print(f"❌ Branch creation failed: {result}")
except Exception as e:
    print(f"❌ Branch creation failed: {e}")
    if "403" in str(e):
        print("\n⚠️  PERMISSIONS ERROR - Follow the guide above to fix!")

print("\nDone!")