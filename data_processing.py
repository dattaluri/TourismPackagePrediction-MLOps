
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download
import os
from sklearn.model_selection import train_test_split

# Define global variables for the Hugging Face repository ID and filenames
REPO_ID = "Dattaluri/TourismPackagePrediction"
PREPROCESSED_DATA_FILENAME = "data/data/tourism_preprocessed.csv"
PROCESSED_DATA_DIR = "processed_data"

def load_and_preprocess_data():
    """
    Downloads the preprocessed data, cleans it, splits it into training and testing sets,
    and saves the processed data locally.

    Returns:
        tuple: X_train, X_test, y_train, y_test DataFrames/Series.
    """
    try:
        # Download the preprocessed CSV file from Hugging Face
        local_file_path = hf_hub_download(repo_id=REPO_ID, filename=PREPROCESSED_DATA_FILENAME)
        print(f"File '{PREPROCESSED_DATA_FILENAME}' downloaded to: {local_file_path}")

        # Load the downloaded CSV file into a pandas DataFrame
        df = pd.read_csv(local_file_path)
        print("DataFrame loaded successfully.")

        # Drop irrelevant columns
        df_cleaned = df.drop(columns=['Unnamed: 0', 'CustomerID'])
        print("Dropped 'Unnamed: 0' and 'CustomerID' columns.")

        # Split the data into features (X) and target (y)
        X = df_cleaned.drop('ProdTaken', axis=1)
        y = df_cleaned['ProdTaken']
        print("Data split into features (X) and target (y).")

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print("Performed stratified train-test split.")

        # Create a directory for processed data if it doesn't exist
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
        print(f"Directory '{PROCESSED_DATA_DIR}' created or already exists.")

        # Save the training and testing sets as CSV files
        X_train.to_csv(os.path.join(PROCESSED_DATA_DIR, 'X_train.csv'), index=False)
        X_test.to_csv(os.path.join(PROCESSED_DATA_DIR, 'X_test.csv'), index=False)
        y_train.to_csv(os.path.join(PROCESSED_DATA_DIR, 'y_train.csv'), index=False)
        y_test.to_csv(os.path.join(PROCESSED_DATA_DIR, 'y_test.csv'), index=False)
        print(f"Processed data saved to '{PROCESSED_DATA_DIR}' directory.")

        return X_train, X_test, y_train, y_test

    except Exception as e:
        print(f"Error during data loading or processing: {e}")
        return None, None, None, None

if __name__ == "__main__":
    print("Starting data processing...")
    X_train, X_test, y_train, y_test = load_and_preprocess_data()

    if X_train is not None:
        print(f"\nX_train shape: {X_train.shape}")
        print(f"X_test shape: {X_test.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"y_test shape: {y_test.shape}")
        print("Data processing completed and verified.")
    else:
        print("Data processing failed.")
