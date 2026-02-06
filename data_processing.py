
import pandas as pd
from sklearn.model_selection import train_test_split
from huggingface_hub import hf_hub_download
import os
import joblib

def run_data_processing():
    # Define paths
    repo_id = "Dattaluri/TourismPackagePrediction"
    raw_data_filename = "data/data/tourism_preprocessed.csv"
    processed_data_dir = "processed_data"

    # Create processed_data directory if it doesn't exist
    os.makedirs(processed_data_dir, exist_ok=True)

    print("Downloading raw data from Hugging Face...")
    try:
        # Download the raw data file
        local_raw_data_path = hf_hub_download(repo_id=repo_id, filename=raw_data_filename)
        print(f"Raw data downloaded to: {local_raw_data_path}")

        # Load the raw CSV file into a pandas DataFrame
        df = pd.read_csv(local_raw_data_path)
        print("Raw DataFrame loaded successfully.")

        # Remove irrelevant columns
        df_cleaned = df.drop(columns=['Unnamed: 0', 'CustomerID'])
        print("Dropped 'Unnamed: 0' and 'CustomerID' columns.")

        # Split data into features (X) and target (y)
        X = df_cleaned.drop('ProdTaken', axis=1)
        y = df_cleaned['ProdTaken']
        print("Split data into features (X) and target (y).")

        # Perform a stratified train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print("Performed stratified train-test split.")

        # Save X_train, X_test, y_train, y_test as CSV files
        X_train.to_csv(os.path.join(processed_data_dir, 'X_train.csv'), index=False)
        X_test.to_csv(os.path.join(processed_data_dir, 'X_test.csv'), index=False)
        y_train.to_csv(os.path.join(processed_data_dir, 'y_train.csv'), index=False)
        y_test.to_csv(os.path.join(processed_data_dir, 'y_test.csv'), index=False)
        print("Saved X_train, X_test, y_train, y_test to processed_data directory.")

        # Apply one-hot encoding to X_train for column alignment reference
        X_train_encoded = pd.get_dummies(X_train, drop_first=True)

        # Save X_train_encoded column names
        column_names_path = os.path.join(processed_data_dir, 'X_train_encoded_columns.joblib')
        joblib.dump(X_train_encoded.columns.tolist(), column_names_path)
        print(f"Saved X_train_encoded column names to '{column_names_path}'.")

        print("Data processing complete.")

    except Exception as e:
        print(f"An error occurred during data processing: {e}")

if __name__ == "__main__":
    run_data_processing()
