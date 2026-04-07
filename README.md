The system includes data ingestion, preprocessing, model training, API deployment, monitoring, and containerization.

## Features

- Data ingestion from Open-Meteo API
- Data preprocessing and feature engineering
- Logistic Regression model for prediction
- FastAPI-based REST API
- Logging for monitoring
- Docker containerization

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
├── README.md            # project documentation
How to Run
1. Install dependencies
pip install -r requirements.txt
2. Run pipeline
python src/ingest.py
python src/preprocess.py
python src/train.py
3. Run API
uvicorn app.main:app --reload

Open:

http://127.0.0.1:8000/docs
Docker
Build image
docker build -t weather-mlops.
Run container
docker run -p 8000:8000 weather-mlops
API Endpoint
POST /predict

Example input:

{
  "temp_max": 20,
  "temp_min": 10,
  "precipitation": 2
}

Example output:

{
  "prediction": 0,
  "result": "No rain tomorrow"
}
Monitoring

Prediction activity is logged in logs.log.



├── notebooks/           
