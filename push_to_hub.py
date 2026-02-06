
from huggingface_hub import HfApi, create_repo, upload_file
import os

# 1. Define the local directory (current working directory)
local_dir = "."

# 2. Define the Hugging Face Space ID
repo_id = "Dattaluri/TourismPackagePrediction-Space" # Using the one from the previous step

# 3. Initialize Hugging Face API
api = HfApi()

# 4. Create the Space on Hugging Face Hub if it doesn't exist
try:
    print(f"Ensuring Space '{repo_id}' exists on Hugging Face Hub...")
    create_repo(repo_id=repo_id, repo_type="space", private=False, exist_ok=True, space_sdk="docker")
    print(f"Space '{repo_id}' created or already exists.")
except Exception as e:
    print(f"Error creating/checking Space '{repo_id}': {e}")
    exit(1) # Exit if repository creation fails

# List of files to be deployed from the current working directory
deployment_files = [
    "Dockerfile",
    "app.py",
    "requirements.txt",
    ".gitattributes",
    "README.md",
    "model_loader.py",
    "prediction_pipeline.py",
]

# Upload each file to the Hugging Face Space
print(f"Uploading deployment files to Hugging Face Space '{repo_id}'...")
for file_name in deployment_files:
    src_path = os.path.join(local_dir, file_name)
    path_in_repo = file_name # Upload to the root of the Space

    if os.path.exists(src_path):
        try:
            upload_file(
                repo_id=repo_id,
                path_or_fileobj=src_path,
                path_in_repo=path_in_repo,
                repo_type="space",
                commit_message=f"Update {file_name}",
                token=os.environ.get("HF_TOKEN") # Use token from environment
            )
            print(f"  Successfully uploaded/updated: {file_name}")
        except Exception as e:
            print(f"  Error uploading {file_name}: {e}")
    else:
        print(f"  Warning: {file_name} not found in '{local_dir}', skipping.")

print(f"Deployment process complete for Space: {repo_id}")

# Diagnostic: Verify current user (if logged in)
try:
    user_info = api.whoami()
    print(f"\nHugging Face user logged in: {user_info['name']}")
except Exception as e:
    print(f"Could not retrieve Hugging Face user info: {e}")
