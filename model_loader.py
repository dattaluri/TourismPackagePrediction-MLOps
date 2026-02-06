
import joblib
from huggingface_hub import hf_hub_download
import os

def load_model_from_hf(repo_id, filename):
    """
    Downloads a model from Hugging Face Hub and loads it using joblib.

    Args:
        repo_id (str): The Hugging Face repository ID.
        filename (str): The path to the model file within the repository.

    Returns:
        object: The loaded model object.
    """
    try:
        # Download the file from Hugging Face
        local_file_path = hf_hub_download(repo_id=repo_id, filename=filename)
        print(f"Model file downloaded to: {local_file_path}")

        # Load the downloaded model
        loaded_model = joblib.load(local_file_path)
        print("Model loaded successfully.")
        return loaded_model
    except Exception as e:
        print(f"Error downloading or loading model from Hugging Face: {e}")
        return None

if __name__ == "__main__":
    # Define the Hugging Face repository ID and the filename within the repository
    repo_id = "Dattaluri/TourismPackagePrediction-Model"
    filename = "best_model.joblib"

    # Load the model
    model = load_model_from_hf(repo_id, filename)

    if model:
        print(f"Type of loaded model: {type(model)}")
        # You can add further testing of the model here if needed
        # For example, print a snippet of its configuration or make a dummy prediction
