
import joblib
import os
from huggingface_hub import HfApi, upload_file, create_repo

def run_model_registration():
    # Define paths
    # Use a dedicated repository for models
    model_repo_id = "Dattaluri/TourismPackagePrediction-Model"
    trained_models_dir = "trained_models"
    model_filename = "best_model.joblib"
    local_model_path = os.path.join(trained_models_dir, model_filename)

    # Ensure the trained_models directory exists (it would be created by model_training.py)
    os.makedirs(trained_models_dir, exist_ok=True)

    # Simulate loading the model (in a real GitHub Actions run, this file would already exist
    # from the previous model_training.py step). For local testing, ensure a dummy model exists.
    if not os.path.exists(local_model_path):
        print(f"Warning: '{local_model_path}' not found. Creating a dummy model for registration.")
        # Create a dummy model for local testing if it doesn't exist
        from sklearn.ensemble import GradientBoostingClassifier
        dummy_model = GradientBoostingClassifier(random_state=1)
        # Save a dummy model locally for this script to pick up
        joblib.dump(dummy_model, local_model_path)

    print(f"Attempting to upload model from: {local_model_path}")

    # Initialize Hugging Face API
    api = HfApi()

    try:
        # Create the model repository if it doesn't exist
        create_repo(repo_id=model_repo_id, repo_type="model", private=False, exist_ok=True)
        print(f"Model repository '{model_repo_id}' created or already exists.")

        upload_file(
            repo_id=model_repo_id,
            path_or_fileobj=local_model_path,
            path_in_repo=model_filename, # Upload directly to the root of the model repo
            repo_type="model",
            commit_message=f"Upload best trained model ({model_filename})",
            token=os.environ.get("HF_TOKEN") # Use token from environment
        )
        print(f"Successfully uploaded '{model_filename}' to Hugging Face Model Hub repository '{model_repo_id}'.")
    except Exception as e:
        print(f"Error uploading model to repository: {e}")

if __name__ == "__main__":
    run_model_registration()
