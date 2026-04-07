import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def train_model():
    df = pd.read_csv("data/processed/weather_processed.csv")

    X = df[["temp_max", "temp_min", "precipitation"]]
    y = df["rain_tomorrow"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("✅ Model trained successfully")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/rain_model.pkl")

    print("✅ Model saved to models/rain_model.pkl")

if __name__ == "__main__":
    train_model()
