# src/app/fastapi_app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
import numpy as np

app = FastAPI()

# Load the model
model_path = "models/random_forest_model.pkl"
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    raise FileNotFoundError(f"Model file not found at {model_path}")

# Define the request schema
class CustomerData(BaseModel):
    flight_class: int
    seat_comfort: float
    inflight_services: float
    departure_delay: float
    # Add more fields as necessary

@app.post("/predict/")
def predict_satisfaction(data: CustomerData):
    """
    Predict customer satisfaction based on input features.
    """
    try:
        # Prepare the feature vector for prediction
        feature_vector = np.array([[
            data.flight_class,
            data.seat_comfort,
            data.inflight_services,
            data.departure_delay
            # Add more features in the same order as the training data
        ]])

        # Make prediction
        prediction = model.predict(feature_vector)
        return {"satisfaction_prediction": bool(prediction[0])}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
