import streamlit as st
import requests

st.title("🌧️ Rain Prediction Dashboard")

# Inputs
temp_max = st.number_input("Max Temperature")
temp_min = st.number_input("Min Temperature")
precipitation = st.number_input("Precipitation")

if st.button("Predict"):
    data = {
        "temp_max": temp_max,
        "temp_min": temp_min,
        "precipitation": precipitation
    }

    response = requests.post("http://127.0.0.1:8000/predict", json=data)

    if response.status_code == 200:
        result = response.json()
        st.success(f"{result['result']}")
    else:
        st.error("Error connecting to API")
