
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download
import os

# --- Global variables for model and columns ---
# These will be loaded once when the application starts
MODEL = None
X_TRAIN_ENCODED_COLUMNS = None

REPO_ID = "Dattaluri/TourismPackagePrediction-Model"
MODEL_FILENAME = "best_model.joblib"
COLUMNS_FILENAME = "processed_data/X_train_encoded_columns.joblib"

def load_artifacts():
    """
    Loads the trained model and X_train_encoded column names from Hugging Face Hub.
    This function should be called once at application startup.
    """
    global MODEL, X_TRAIN_ENCODED_COLUMNS

    if MODEL is None:
        try:
            model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILENAME)
            MODEL = joblib.load(model_path)
            print(f"Model '{MODEL_FILENAME}' loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")

    if X_TRAIN_ENCODED_COLUMNS is None:
        try:
            # Assuming 'processed_data' is in the main data repo, not the model repo
            columns_repo_id = "Dattaluri/TourismPackagePrediction"
            columns_path = hf_hub_download(repo_id=columns_repo_id, filename=COLUMNS_FILENAME)
            X_TRAIN_ENCODED_COLUMNS = joblib.load(columns_path)
            print(f"Columns '{COLUMNS_FILENAME}' loaded successfully.")
        except Exception as e:
            print(f"Error loading columns: {e}")

def preprocess_input(input_data: dict) -> pd.DataFrame:
    """
    Receives raw input data (e.g., from a web request), structures it into a pandas DataFrame,
    applies one-hot encoding, and ensures its columns match the training data format.

    Args:
        input_data (dict): A dictionary representing a single customer's features.

    Returns:
        pd.DataFrame: A prepared DataFrame with encoded features, ready for prediction.
    """
    if X_TRAIN_ENCODED_COLUMNS is None:
        raise RuntimeError("X_TRAIN_ENCODED_COLUMNS not loaded. Call load_artifacts() first.")

    # Convert the input data to a DataFrame
    # Ensure it's a DataFrame with one row to handle categorical encoding correctly
    input_df = pd.DataFrame([input_data])

    # Apply one-hot encoding
    input_df_encoded = pd.get_dummies(input_df, drop_first=True)

    # Reindex columns to match X_train_encoded
    # Add missing columns with fill_value=0 and align order
    final_input_df = input_df_encoded.reindex(columns=X_TRAIN_ENCODED_COLUMNS, fill_value=0)

    return final_input_df

if __name__ == "__main__":
    # Load artifacts when the script is executed directly (e.g., for testing)
    load_artifacts()

    if MODEL and X_TRAIN_ENCODED_COLUMNS:
        print("\nReady for predictions. Testing preprocess_input function...")

        # Example raw input data (should reflect original df_cleaned columns)
        sample_input = {
            'Age': 35.0,
            'TypeofContact': 'Self Enquiry',
            'CityTier': 1,
            'DurationOfPitch': 10.0,
            'Occupation': 'Salaried',
            'Gender': 'Male',
            'NumberOfPersonVisiting': 2,
            'NumberOfFollowups': 3.0,
            'ProductPitched': 'Basic',
            'PreferredPropertyStar': 3.0,
            'MaritalStatus': 'Married',
            'NumberOfTrips': 2.0,
            'Passport': 0,
            'PitchSatisfactionScore': 4,
            'OwnCar': 1,
            'NumberOfChildrenVisiting': 1.0,
            'Designation': 'Executive',
            'MonthlyIncome': 25000.0
        }

        preprocessed_data = preprocess_input(sample_input)
        print("\nPreprocessed Data Head:")
        print(preprocessed_data.head())
        print("\nPreprocessed Data Shape:", preprocessed_data.shape)
        print("Preprocessed Data Columns == X_TRAIN_ENCODED_COLUMNS:",
              list(preprocessed_data.columns) == list(X_TRAIN_ENCODED_COLUMNS))

        # Make a prediction with the loaded model
        if MODEL:
            prediction = MODEL.predict(preprocessed_data)[0]
            prediction_proba = MODEL.predict_proba(preprocessed_data)[0].tolist()
            print(f"\nPrediction for sample input: {prediction}")
            print(f"Prediction probability for sample input: {prediction_proba}")
