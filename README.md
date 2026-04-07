# Weather Rain Prediction MLOps Pipeline

## Overview

This project implements a complete MLOps pipeline for predicting whether it will rain tomorrow based on weather data.

The system includes data ingestion, preprocessing, model training, API deployment, monitoring, and containerization.

The pipeline automates data collection, processing, model training, and deployment for real-time predictions.

---

## Features

- Data ingestion from Open-Meteo API
- Data preprocessing and feature engineering
- Logistic Regression model for prediction
- FastAPI-based REST API
- Logging for monitoring
- Docker containerization

---

## Tech Stack

- Python
- FastAPI
- Scikit-learn
- Docker
- Open-Meteo API

---

## Project Structure

```text
weather-mlops-pipeline/
├── app/                 # API code
├── data/                # raw and processed data
├── models/              # trained model
├── src/                 # pipeline scripts
├── notebooks/           # experiments
├── Dockerfile           # container setup
├── requirements.txt     # dependencies
├── report.md            # project report
```

---

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run pipeline
```bash
python src/ingest.py
python src/preprocess.py
python src/train.py
```

### 3. Run API
```bash
uvicorn app.main:app --reload
```

Open:
```
http://127.0.0.1:8000/docs
```

---

## Docker

### Build image
```bash
docker build -t weather-mlops .
```

### Run container
```bash
docker run -p 8000:8000 weather-mlops
```

---

## API Endpoint

### POST /predict

Example input:
```json
{
  "temp_max": 20,
  "temp_min": 10,
  "precipitation": 2
}
```

Example output:
```json
{
  "prediction": 0,
  "result": "No rain tomorrow"
}
```

---

## Monitoring

All predictions are logged in `logs.log`.
