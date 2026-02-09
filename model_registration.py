
import joblib
import os
from huggingface_hub import HfApi # Corrected import

# --- Constants for Hugging Face and local paths ---
REPO_ID = "Dattaluri/TourismPackagePrediction"  # Your Hugging Face repository ID

# Local path where the best model is saved by model_training.py
MODEL_LOCAL_PATH = "trained_models/gradient_boosting_model.joblib"

# Path within the Hugging Face repository where the model should be uploaded
MODEL_HF_PATH = "trained_models/gradient_boosting_model.joblib"

def register_model_on_hf(
    repo_id: str,
    model_local_path: str,
    model_hf_path: str
):
    """
    Uploads a model file to the Hugging Face Model Hub using HfApi.

    Args:
        repo_id (str): The Hugging Face repository ID (e.g., 'your-username/your-repo-name').
        model_local_path (str): The local path to the model file to be uploaded.
        model_hf_path (str): The path within the Hugging Face repository where the model will be stored.
    """
    api = HfApi()
    try:
        print(f"Uploading model from '{model_local_path}' to '{repo_id}/{model_hf_path}'...")
        api.upload_file(
            repo_id=repo_id,
            path_or_fileobj=model_local_path,
            path_in_repo=model_hf_path,
            commit_message=f"Registering {os.path.basename(model_local_path)}"
        )
        print("Model successfully uploaded to Hugging Face Model Hub.")
    except Exception as e:
        print(f"Error uploading model to Hugging Face: {e}")

if __name__ == '__main__':
    print("Starting model registration process...")
    register_model_on_hf(REPO_ID, MODEL_LOCAL_PATH, MODEL_HF_PATH)
    print("Model registration process completed.")
