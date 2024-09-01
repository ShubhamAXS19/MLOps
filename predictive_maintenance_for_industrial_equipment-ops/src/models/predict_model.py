# src/app.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np

# Load the trained model and scaler
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

app = FastAPI()

class SensorData(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    # Add all relevant features here

@app.post("/predict")
def predict(data: SensorData):
    try:
        # Convert input data to DataFrame
        data_dict = data.dict()
        df = pd.DataFrame([data_dict])

        # Scale the input data
        df_scaled = scaler.transform(df)

        # Make a prediction
        prediction = model.predict(df_scaled)
        
        # Convert prediction to list for JSON serialization
        prediction = prediction.tolist()

        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
