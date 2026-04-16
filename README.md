# 🌧️ Weather Rain Prediction MLOps Pipeline

## 📌 Overview

This project builds an end-to-end MLOps pipeline to predict whether it will rain tomorrow using weather data.

The system demonstrates a complete machine learning lifecycle, including data ingestion, preprocessing, model training, deployment, monitoring, and reproducibility.

---

## ⚙️ Features

- Data ingestion (Open-Meteo API)  
- Data preprocessing and feature engineering  
- Logistic Regression model  
- Model evaluation (accuracy, precision, recall, F1-score)  
- FastAPI deployment  
- Monitoring using logs and runtime metrics  
- Artifact storage for reproducibility  
- MLflow experiment tracking  
- Streamlit interactive dashboard  
- Docker containerization  

---

## 🧱 Pipeline Overview

The pipeline follows this flow:

1. Data Ingestion → Fetch weather data from API  
2. Data Preprocessing → Clean data and create features  
3. Feature Engineering → Create `temp_range` and `precipitation_lag1`  
4. Model Training → Train Logistic Regression model  
5. Evaluation → Calculate performance metrics  
6. Deployment → Serve model via FastAPI  
7. Monitoring → Track logs and runtime metrics  

---

## 📂 Project Structure

```bash
weather-mlops-pipeline/
├── app/                  # FastAPI application
├── src/                  # Pipeline scripts
├── data/                 # Raw and processed data
├── models/               # Trained models
├── mlruns/               # MLflow tracking
├── dashboard.py          # Streamlit dashboard
├── metrics.json          # Model performance
├── model_history.json    # Model history
├── artifacts.json        # Artifact tracking
├── logs.log              # Logs
├── Dockerfile            # Container setup
├── requirements.txt      # Dependencies
└── README.md             # Documentation
```

## 🚀 How to Run

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

Open API docs:

http://127.0.0.1:8000/docs

---

## 🔌 API Endpoints
- /predict → Predict rain
- /health → Check system status
- /metrics → View runtime + training metrics


### Example Request
```json
{
  "temp_max": 20,
  "temp_min": 10,
  "precipitation": 2
}
```

---

## 📊 Model Performance
- Accuracy: 0.56
- Precision: 0.43
- Recall: 0.43
- F1-score: 0.43

Note: The focus is on building a complete MLOps pipeline rather than achieving perfect model accuracy.

---

## 📈 Monitoring

The system tracks:

- Total requests
- Successful predictions
- Errors
- Last prediction time

Logs are stored in `logs.log`.

---

## 📦 Artifacts

Artifacts are stored for reproducibility:

- Raw data → data/raw/weather.csv
- Processed data → data/processed/weather_processed.csv
- Model → models/rain_model_v1.pkl
- Metrics → metrics.json
- Model history → model_history.json
- Logs → logs.log

---

## 🔁 Reproducibility

The project ensures reproducibility by:

- Saving datasets and models
- Tracking artifacts
- Using a fixed random state
- Containerizing with Docker

---

## 🧪 MLflow Tracking

MLflow is used for:

- Experiment tracking  
- Logging parameters  
- Logging metrics  
- Saving trained models  

Runs are stored in:

mlruns/

---

## 🖥️ Frontend (Streamlit)

Run dashboard:

```bash
streamlit run dashboard.py
```

The dashboard allows users to:

- Input weather values
- Get predictions
- View model performance

---

## 🐳 Docker

### Build image
```bash
docker build -t weather-mlops .
```

### Run container
```bash
docker run -p 8000:8000 weather-mlops
```

---

## 🎯 Goal

The goal of this project is to demonstrate a working, reproducible, and deployable MLOps system.

The emphasis is on technical implementation and operationalization rather than perfect model performance.
