from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import logging
import json
import time
from datetime import datetime
import os

# Model version
MODEL_VERSION = "v1"

# Monitoring
request_count = 0
success_count = 0
error_count = 0
last_prediction_time = None

# Logging
logging.basicConfig(
    filename="logs.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)

# App
app = FastAPI(title="Rain Prediction API")

# Load model
model_path = f"models/rain_model_{MODEL_VERSION}.pkl"
model = joblib.load(model_path)
logging.info(f"Model version {MODEL_VERSION} loaded successfully")


class WeatherInput(BaseModel):
    temp_max: float
    temp_min: float
    precipitation: float


@app.get("/")
def home():
    return {"message": "API running"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_version": MODEL_VERSION,
        "model_loaded": os.path.exists(model_path)
    }


@app.get("/metrics")
def get_metrics():
    try:
        with open("metrics.json", "r") as f:
            saved_metrics = json.load(f)

        return {
            "status": "success",
            "model_version": MODEL_VERSION,
            "runtime_metrics": {
                "total_requests": request_count,
                "success": success_count,
                "errors": error_count,
                "last_prediction_time": last_prediction_time
            },
            "training_metrics": saved_metrics
        }
    except Exception as e:
        logging.error(f"Metrics error | Version: {MODEL_VERSION} | {str(e)}")
        return {
            "status": "error",
            "message": "Could not load metrics"
        }


@app.post("/predict")
def predict(data: WeatherInput):
    global request_count, success_count, error_count, last_prediction_time

    start_time = time.time()
    request_count += 1

    try:
        if data.temp_max < data.temp_min:
            error_count += 1
            logging.error(
                f"Version: {MODEL_VERSION} | Invalid input: temp_max < temp_min | Input: {data.model_dump()}"
            )
            return {
                "status": "error",
                "message": "temp_max cannot be less than temp_min"
            }

        if data.precipitation < 0:
            error_count += 1
            logging.error(
                f"Version: {MODEL_VERSION} | Invalid input: negative precipitation | Input: {data.model_dump()}"
            )
            return {
                "status": "error",
                "message": "precipitation cannot be negative"
            }

        # Feature engineering must match training
        temp_range = data.temp_max - data.temp_min
        precipitation_lag1 = data.precipitation

        input_df = pd.DataFrame([{
            "temp_max": data.temp_max,
            "temp_min": data.temp_min,
            "precipitation": data.precipitation,
            "temp_range": temp_range,
            "precipitation_lag1": precipitation_lag1
        }])

        prediction = model.predict(input_df)[0]
        result = "Rain tomorrow" if prediction == 1 else "No rain tomorrow"

        success_count += 1
        last_prediction_time = datetime.now().isoformat()
        duration = round(time.time() - start_time, 4)

        log_entry = {
            "timestamp": last_prediction_time,
            "model_version": MODEL_VERSION,
            "input": data.model_dump(),
            "prediction": int(prediction),
            "result": result,
            "request_number": request_count,
            "time_taken_seconds": duration
        }

        logging.info(json.dumps(log_entry))

        return {
            "status": "success",
            "prediction": int(prediction),
            "result": result,
            "model_version": MODEL_VERSION,
            "request_number": request_count,
            "time_taken_seconds": duration
        }

    except Exception as e:
        error_count += 1
        logging.error(f"Version: {MODEL_VERSION} | Error: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }
