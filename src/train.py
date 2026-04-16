import os
import json
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
)

# Model version
MODEL_VERSION = "v1"
MODEL_NAME = "Logistic Regression"

# MLflow settings
EXPERIMENT_NAME = "Weather Rain Prediction"
TRACKING_URI = "file:./mlruns"

# Input file
PROCESSED_DATA_PATH = "data/processed/weather_processed.csv"

# Features used in both training and API
FEATURE_COLUMNS = [
    "temp_max",
    "temp_min",
    "precipitation",
    "temp_range",
    "precipitation_lag1",
]

TARGET_COLUMN = "rain_tomorrow"


def train_model():
    # Set MLflow tracking
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Check processed data exists
    if not os.path.exists(PROCESSED_DATA_PATH):
        raise FileNotFoundError(
            f"Processed data not found at: {PROCESSED_DATA_PATH}"
        )

    # Load processed data
    df = pd.read_csv(PROCESSED_DATA_PATH)

    # Select features and target
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Parameters to log in MLflow
    params = {
        "model_type": "LogisticRegression",
        "max_iter": 1000,
        "test_size": 0.2,
        "random_state": 42,
        "class_weight": "balanced",
        "features": ",".join(FEATURE_COLUMNS),
        "model_version": MODEL_VERSION,
    }

    with mlflow.start_run(run_name=f"rain_model_{MODEL_VERSION}"):
        # Log params
        mlflow.log_params(params)

        # Train model
        model = LogisticRegression(max_iter=1000, class_weight="balanced")
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        report_text = classification_report(y_test, y_pred)

        metrics = {
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4)
        }

        print("✅ Model trained successfully")
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:\n")
        print(report_text)

        # Save model
        os.makedirs("models", exist_ok=True)
        model_path = f"models/rain_model_{MODEL_VERSION}.pkl"
        joblib.dump(model, model_path)
        print(f"✅ Model saved to {model_path}")

        # Save metrics
        with open("metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)
        print("✅ Metrics saved to metrics.json")

        # Save model history
        history_file = "model_history.json"
        new_entry = {
            "version": MODEL_VERSION,
            "model_name": MODEL_NAME,
            "accuracy": round(accuracy, 2),
            "precision": round(precision, 2),
            "recall": round(recall, 2),
            "f1_score": round(f1, 2)
        }

        if os.path.exists(history_file):
            try:
                with open(history_file, "r") as f:
                    history = json.load(f)
                if not isinstance(history, list):
                    history = []
            except json.JSONDecodeError:
                history = []
        else:
            history = []

        # Remove old entry with same version to avoid duplicates
        history = [entry for entry in history if entry.get("version") != MODEL_VERSION]
        history.append(new_entry)

        with open(history_file, "w") as f:
            json.dump(history, f, indent=4)
        print("✅ Model history updated")

        # Save artifacts metadata
        artifacts = {
            "raw_data": "data/raw/weather.csv",
            "processed_data": PROCESSED_DATA_PATH,
            "model": model_path,
            "metrics": "metrics.json",
            "model_history": history_file,
            "logs": "logs.log"
        }

        with open("artifacts.json", "w") as f:
            json.dump(artifacts, f, indent=4)
        print("✅ Artifacts saved to artifacts.json")

        # Log metrics to MLflow
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # Log artifacts to MLflow
        mlflow.log_artifact("metrics.json")
        mlflow.log_artifact(history_file)
        mlflow.log_artifact("artifacts.json")

        # Log sklearn model
        mlflow.sklearn.log_model(model, artifact_path="model")

        # Tags
        mlflow.set_tag("project", "Weather Rain Prediction MLOps Pipeline")
        mlflow.set_tag("stage", "training")
        mlflow.set_tag("version", MODEL_VERSION)

        print("✅ MLflow run logged successfully")


if __name__ == "__main__":
    train_model()
