import pandas as pd
import os

def preprocess_data():
    # Load raw data
    df = pd.read_csv("data/raw/weather.csv")

    # Feature engineering
    df["temp_range"] = df["temp_max"] - df["temp_min"]
    df["precipitation_lag1"] = df["precipitation"].shift(1)

    # Create target variable: will it rain tomorrow?
    df["rain_tomorrow"] = df["precipitation"].shift(-1)
    df["rain_tomorrow"] = df["rain_tomorrow"].apply(lambda x: 1 if x > 0 else 0)

    # Remove missing values created by shifting
    df = df.dropna()

    # Keep final columns used in training
    df = df[[
        "temp_max",
        "temp_min",
        "precipitation",
        "temp_range",
        "precipitation_lag1",
        "rain_tomorrow"
    ]]

    # Create output folder
    os.makedirs("data/processed", exist_ok=True)

    # Save processed data
    df.to_csv("data/processed/weather_processed.csv", index=False)

    print("✅ Processed data saved to data/processed/weather_processed.csv")


if __name__ == "__main__":
    preprocess_data()
