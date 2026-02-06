---
title: Tourism Package Prediction API
emoji: "✈✈"
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 8000
---

# Tourism Package Prediction API

This Hugging Face Space hosts a machine learning model that predicts whether a customer will purchase a Wellness Tourism Package. The model is served via a FastAPI application within a Docker container.

## Model Details

The best performing model, a Gradient Boosting Classifier, was trained on customer demographic and interaction data. It was chosen for its superior F1-score on the test set.

## API Usage

The API exposes a `/predict` endpoint to receive customer data and return a prediction (purchased/not purchased) along with probabilities.

### Endpoint: `/predict` (POST)

**Request Body Example:**

```json
{
    "Age": 35.0,
    "TypeofContact": "Self Enquiry",
    "CityTier": 1,
    "DurationOfPitch": 10.0,
    "Occupation": "Salaried",
    "Gender": "Male",
    "NumberOfPersonVisiting": 2,
    "NumberOfFollowups": 3.0,
    "ProductPitched": "Basic",
    "PreferredPropertyStar": 3.0,
    "MaritalStatus": "Married",
    "NumberOfTrips": 2.0,
    "Passport": 0,
    "PitchSatisfactionScore": 4,
    "OwnCar": 1,
    "NumberOfChildrenVisiting": 1.0,
    "Designation": "Executive",
    "MonthlyIncome": 25000.0
}
```

**Response Body Example:**

```json
{
    "prediction": 0,
    "prediction_label": "Not Purchased",
    "probability_not_purchased": 0.9999989682705609,
    "probability_purchased": 1.031655848574163e-06
}
```

## Deployment

This Space is deployed using a Docker SDK, with the FastAPI application running on port 8000.
The model and preprocessing artifacts are loaded directly from the `Dattaluri/TourismPackagePrediction` dataset on Hugging Face Hub.
