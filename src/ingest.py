import requests
import pandas as pd
import os

def fetch_weather_data():
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        "?latitude=57.0488&longitude=9.9217"
        "&start_date=2025-01-01&end_date=2025-03-31"
        "&daily=temperature_2m_max,temperature_2m_min,precipitation_sum"
        "&timezone=auto"
    )

    response = requests.get(url)
    data = response.json()

    df = pd.DataFrame({
        "date": data["daily"]["time"],
        "temp_max": data["daily"]["temperature_2m_max"],
        "temp_min": data["daily"]["temperature_2m_min"],
        "precipitation": data["daily"]["precipitation_sum"]
    })

    return df

if __name__ == "__main__":
    df = fetch_weather_data()

    os.makedirs("data/raw", exist_ok=True)
    df.to_csv("data/raw/weather.csv", index=False)

    print("✅ Data saved to data/raw/weather.csv")
