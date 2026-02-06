
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np

from prediction_pipeline import load_artifacts, preprocess_input, MODEL, X_TRAIN_ENCODED_COLUMNS

# Define the input data model using Pydantic BaseModel
class PredictionRequest(BaseModel):
    Age: float
    TypeofContact: str
    CityTier: int
    DurationOfPitch: float
    Occupation: str
    Gender: str
    NumberOfPersonVisiting: int
    NumberOfFollowups: float
    ProductPitched: str
    PreferredPropertyStar: float
    MaritalStatus: str
    NumberOfTrips: float
    Passport: int
    PitchSatisfactionScore: int
    OwnCar: int
    NumberOfChildrenVisiting: float
    Designation: str
    MonthlyIncome: float

# Initialize FastAPI app
app = FastAPI()

# Load model and column names when the application starts
# This ensures artifacts are loaded only once
@app.on_event("startup")
async def startup_event():
    print("Loading model and column names on startup...")
    load_artifacts()
    if MODEL is None or X_TRAIN_ENCODED_COLUMNS is None:
        raise RuntimeError("Failed to load model or column names during startup.")
    print("Model and column names loaded successfully.")

@app.get("/")
async def root():
    return {"message": "Welcome to the Tourism Package Prediction API!"}

@app.post("/predict")
async def predict(request: PredictionRequest):
    # Convert incoming request data to a dictionary
    input_data_dict = request.model_dump()

    # Preprocess the input data
    preprocessed_df = preprocess_input(input_data_dict)

    # Make prediction
    prediction = MODEL.predict(preprocessed_df)[0]
    prediction_proba = MODEL.predict_proba(preprocessed_df)[0].tolist()

    # Map prediction to a human-readable label if desired
    prediction_label = "Purchased" if prediction == 1 else "Not Purchased"

    return {
        "prediction": int(prediction),
        "prediction_label": prediction_label,
        "probability_not_purchased": prediction_proba[0],
        "probability_purchased": prediction_proba[1]
    }
