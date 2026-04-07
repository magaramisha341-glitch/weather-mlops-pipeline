from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import logging

logging.basicConfig(
    filename="logs.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)

app = FastAPI(title="Weather Rain Prediction API")

model = joblib.load("models/rain_model.pkl")

class WeatherInput(BaseModel):
    temp_max: float
    temp_min: float
    precipitation: float

@app.get("/")
def home():
    return {"message": "Weather Rain Prediction API is running"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: WeatherInput):
    input_df = pd.DataFrame([{
        "temp_max": data.temp_max,
        "temp_min": data.temp_min,
        "precipitation": data.precipitation
    }])

    prediction = model.predict(input_df)[0]
    result = "Rain tomorrow" if prediction == 1 else "No rain tomorrow"

    logging.info(f"Input: {data.dict()} -> Prediction: {result}")

    return {
        "prediction": int(prediction),
        "result": result
    }
