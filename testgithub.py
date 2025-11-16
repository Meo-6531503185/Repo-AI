"""
Comprehensive GitHub App Debugger
This will show you EXACTLY what's wrong with your GitHub App setup.
Run with: python detailed_debug.py
"""

import os
import sys
import jwt
import time
import requests
from dotenv import load_dotenv

load_dotenv()

print("=" * 70)
print("COMPREHENSIVE GITHUB APP DEBUGGER")
print("=" * 70)

# Step 1: Check environment variables
print("\n📋 STEP 1: Environment Variables")
print("-" * 70)

github_app_id = os.getenv("GITHUB_APP_ID")
github_private_key_path = os.getenv("GITHUB_PRIVATE_KEY")
github_repo = os.getenv("GITHUB_REPOSITORY", "moepyaePK/dormbooking")

if not github_app_id:
    print("❌ GITHUB_APP_ID not found in .env file")
    sys.exit(1)
print(f"✅ App ID: {github_app_id}")

if not github_private_key_path:
    print("❌ GITHUB_PRIVATE_KEY not found in .env file")
    sys.exit(1)
print(f"✅ Private Key Path: {github_private_key_path}")

if not os.path.exists(github_private_key_path):
    print(f"❌ Private key file not found at: {github_private_key_path}")
    sys.exit(1)
print(f"✅ Private key file exists")

print(f"ℹ️  Target Repository: {github_repo}")

# Step 2: Read and validate private key
print("\n🔑 STEP 2: Private Key Validation")
print("-" * 70)

try:
    with open(github_private_key_path, 'r') as f:
        private_key = f.read()
    
    if not private_key.strip():
        print("❌ Private key file is empty")
        sys.exit(1)
    
    if "BEGIN RSA PRIVATE KEY" not in private_key and "BEGIN PRIVATE KEY" not in private_key:
        print("❌ Private key format is invalid")
        print("   It should start with '-----BEGIN RSA PRIVATE KEY-----' or '-----BEGIN PRIVATE KEY-----'")
        sys.exit(1)
    
    print("✅ Private key format looks valid")
    print(f"   Length: {len(private_key)} characters")
    
except Exception as e:
    print(f"❌ Error reading private key: {e}")
    sys.exit(1)

# Step 3: Generate JWT
print("\n🎫 STEP 3: JWT Generation")
print("-" * 70)

try:
    now = int(time.time())
    payload = {
        "iat": now,
        "exp": now + 600,
        "iss": github_app_id
    }
    
    jwt_token = jwt.encode(payload, private_key, algorithm="RS256")
    print("✅ JWT generated successfully")
    print(f"   Token preview: {jwt_token[:50]}...")
    
except Exception as e:
    print(f"❌ Failed to generate JWT: {e}")
    print("\n   Common causes:")
    print("   - Private key doesn't match the GitHub App")
    print("   - Private key format is incorrect")
    print("   - App ID is wrong")
    sys.exit(1)

# Step 4: Get installation info
print("\n🔌 STEP 4: GitHub App Installations")
print("-" * 70)

headers = {
    "Authorization": f"Bearer {jwt_token}",
    "Accept": "application/vnd.github+json"
}

try:
    url = "https://api.github.com/app/installations"
    response = requests.get(url, headers=headers)
    
    print(f"Response Status: {response.status_code}")
    
    if response.status_code == 401:
        print("❌ Authentication failed - JWT is invalid")
        print("   Possible reasons:")
        print("   - App ID is incorrect")
        print("   - Private key doesn't match this app")
        print("\n   Verify at: https://github.com/settings/apps")
        sys.exit(1)
    
    if response.status_code == 404:
        print("❌ App not found")
        print(f"   App ID {github_app_id} doesn't exist")
        print("   Verify at: https://github.com/settings/apps")
        sys.exit(1)
    
    if response.status_code != 200:
        print(f"❌ Unexpected status code: {response.status_code}")
        print(f"   Response: {response.text[:500]}")
        sys.exit(1)
    
    installations = response.json()
    
    if not installations:
        print("❌ No installations found")
        print("\n   TO FIX:")
        print("   1. Go to: https://github.com/apps/[YOUR_APP_NAME]")
        print("   2. Click 'Install App' or 'Configure'")
        print(f"   3. Select the repository: {github_repo}")
        print("   4. Complete the installation")
        sys.exit(1)
    
    print(f"✅ Found {len(installations)} installation(s)")
    
    for idx, installation in enumerate(installations):
        print(f"\n   Installation {idx + 1}:")
        print(f"   - ID: {installation['id']}")
        print(f"   - Account: {installation['account']['login']}")
        print(f"   - Type: {installation['account']['type']}")
        print(f"   - Target: {installation.get('target_type', 'N/A')}")
        
        # Get detailed installation info
        installation_id = installation['id']
        
except Exception as e:
    print(f"❌ Error getting installations: {e}")
    sys.exit(1)

# Step 5: Get installation token
print("\n🎟️  STEP 5: Installation Access Token")
print("-" * 70)

try:
    installation_id = installations[0]['id']
    token_url = f"https://api.github.com/app/installations/{installation_id}/access_tokens"
    
    response = requests.post(token_url, headers=headers)
    
    if response.status_code != 201:
        print(f"❌ Failed to get installation token: {response.status_code}")
        print(f"   Response: {response.text[:500]}")
        sys.exit(1)
    
    token_data = response.json()
    installation_token = token_data['token']
    
    print("✅ Installation token generated")
    print(f"   Token preview: {installation_token[:20]}...")
    print(f"   Expires: {token_data.get('expires_at', 'N/A')}")
    
    # Check permissions
    if 'permissions' in token_data:
        print("\n   📜 Current Permissions:")
        for key, value in token_data['permissions'].items():
            emoji = "✅" if value == "write" else "⚠️" if value == "read" else "❌"
            print(f"   {emoji} {key}: {value}")
        
        # Check critical permissions
        contents_perm = token_data['permissions'].get('contents', 'none')
        pr_perm = token_data['permissions'].get('pull_requests', 'none')
        
        if contents_perm != 'write':
            print(f"\n   ⚠️  WARNING: 'contents' permission is '{contents_perm}', needs 'write'")
        if pr_perm != 'write':
            print(f"   ⚠️  WARNING: 'pull_requests' permission is '{pr_perm}', needs 'write'")
        
        if contents_perm != 'write' or pr_perm != 'write':
            print("\n   🚨 PERMISSION ISSUE FOUND!")
            print("\n   TO FIX:")
            print("   1. Go to: https://github.com/settings/apps")
            print(f"   2. Click on your app (ID: {github_app_id})")
            print("   3. Go to 'Permissions & events'")
            print("   4. Under 'Repository permissions':")
            print("      - Set 'Contents' to 'Read and write'")
            print("      - Set 'Pull requests' to 'Read and write'")
            print("   5. Scroll down and click 'Save changes'")
            print("   6. Go to: https://github.com/settings/installations")
            print("   7. Click 'Configure' on your app")
            print("   8. You'll see 'New permissions requested'")
            print("   9. Click 'Accept new permissions'")
            print("\n   Then run this script again!")
            
except Exception as e:
    print(f"❌ Error getting installation token: {e}")
    sys.exit(1)

# Step 6: Test repository access
print("\n📂 STEP 6: Repository Access Test")
print("-" * 70)

try:
    repo_url = f"https://api.github.com/repos/{github_repo}"
    headers_with_token = {
        "Authorization": f"token {installation_token}",
        "Accept": "application/vnd.github+json"
    }
    
    response = requests.get(repo_url, headers=headers_with_token)
    
    if response.status_code == 404:
        print(f"❌ Repository not found: {github_repo}")
        print("\n   Possible reasons:")
        print("   - Repository name is incorrect")
        print("   - App is not installed on this repository")
        print("   - Repository is private and app doesn't have access")
        sys.exit(1)
    
    if response.status_code != 200:
        print(f"❌ Cannot access repository: {response.status_code}")
        print(f"   Response: {response.text[:500]}")
        sys.exit(1)
    
    repo_data = response.json()
    print(f"✅ Can access repository: {repo_data['full_name']}")
    print(f"   Default branch: {repo_data['default_branch']}")
    print(f"   Private: {repo_data['private']}")
    
except Exception as e:
    print(f"❌ Error accessing repository: {e}")
    sys.exit(1)

# Step 7: Test branch creation (the actual problem)
print("\n🌿 STEP 7: Branch Creation Test")
print("-" * 70)

try:
    # Get default branch SHA
    branch_url = f"https://api.github.com/repos/{github_repo}/git/refs/heads/{repo_data['default_branch']}"
    response = requests.get(branch_url, headers=headers_with_token)
    
    if response.status_code != 200:
        print(f"❌ Cannot get default branch: {response.status_code}")
        sys.exit(1)
    
    default_sha = response.json()['object']['sha']
    print(f"✅ Got default branch SHA: {default_sha[:10]}...")
    
    # Try to create a test branch
    test_branch = f"permission-test-{int(time.time())}"
    create_branch_url = f"https://api.github.com/repos/{github_repo}/git/refs"
    
    payload = {
        "ref": f"refs/heads/{test_branch}",
        "sha": default_sha
    }
    
    print(f"\n   Attempting to create test branch: {test_branch}")
    response = requests.post(create_branch_url, json=payload, headers=headers_with_token)
    
    print(f"   Response Status: {response.status_code}")
    
    if response.status_code == 201:
        print("   ✅ SUCCESS! Branch created successfully")
        print("\n   🎉 Your GitHub App has WRITE permissions!")
        print("   You can now use the REPO AI Refactorer")
        
        # Clean up test branch
        delete_url = f"https://api.github.com/repos/{github_repo}/git/refs/heads/{test_branch}"
        requests.delete(delete_url, headers=headers_with_token)
        print(f"   ✅ Test branch deleted")
        
    elif response.status_code == 403:
        print("   ❌ PERMISSION DENIED (403)")
        error_data = response.json()
        print(f"   Error: {error_data.get('message', 'Unknown error')}")
        
        print("\n   🚨 THIS IS THE PROBLEM!")
        print("\n   Your GitHub App DOES NOT have write permissions.")
        print("\n   SOLUTION:")
        print("   1. Visit: https://github.com/settings/apps")
        print(f"   2. Select your app (App ID: {github_app_id})")
        print("   3. Click 'Permissions & events' in sidebar")
        print("   4. Find 'Repository permissions' section")
        print("   5. Change these to 'Read and write':")
        print("      • Contents")
        print("      • Pull requests")
        print("   6. Scroll to bottom → Click 'Save changes'")
        print("   7. Visit: https://github.com/settings/installations")
        print("   8. Find your app → Click 'Configure'")
        print("   9. Accept the new permissions")
        print("\n   WAIT 1-2 MINUTES after accepting permissions")
        print("   Then run this script again to verify!")
        
    elif response.status_code == 422:
        print("   ⚠️  Branch might already exist (422)")
        print("   This actually means write permissions might work!")
        
    else:
        print(f"   ❌ Unexpected error: {response.status_code}")
        print(f"   Response: {response.text[:500]}")
    
except Exception as e:
    print(f"❌ Error testing branch creation: {e}")
    import traceback
    traceback.print_exc()

# Final summary
print("\n" + "=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)

print("\n📊 Summary:")
print("   ✅ App authentication: Working")
print("   ✅ Installation found: Working")
print("   ✅ Installation token: Working")
print("   ✅ Repository access: Working")

if response.status_code == 201:
    print("   ✅ Branch creation: Working")
    print("\n🎉 Everything is configured correctly!")
    print("   You can now run: streamlit run app.py")
else:
    print("   ❌ Branch creation: FAILED")
    print("\n⚠️  Follow the SOLUTION steps above to fix permissions")

print("\n" + "=" * 70)