import pandas as pd
import os

def preprocess_data():
    df = pd.read_csv("data/raw/weather.csv")

    df["rain_tomorrow"] = df["precipitation"].shift(-1)
    df["rain_tomorrow"] = df["rain_tomorrow"].apply(lambda x: 1 if x > 0 else 0)

    df = df.dropna()

    print(df["rain_tomorrow"].value_counts())

    return df

if __name__ == "__main__":
    df = preprocess_data()

    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/weather_processed.csv", index=False)

    print("✅ Processed data saved to data/processed/weather_processed.csv")
