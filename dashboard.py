import streamlit as st
import requests
import json
import os

st.set_page_config(page_title="Rain Prediction Dashboard", page_icon="🌧️", layout="wide")


def load_metrics():
    try:
        if os.path.exists("metrics.json"):
            with open("metrics.json", "r") as f:
                return json.load(f)
    except Exception:
        return None
    return None


st.title("🌧️ Weather Rain Prediction Dashboard")
st.write("Enter weather values to predict whether it will rain tomorrow.")

# -----------------------------
# Model performance section
# -----------------------------
st.subheader("📊 Model Performance Metrics")

metrics = load_metrics()

if metrics:
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", metrics.get("accuracy", "N/A"))
    col2.metric("Precision", metrics.get("precision", "N/A"))
    col3.metric("Recall", metrics.get("recall", "N/A"))
else:
    st.warning("Metrics file not found. Please train the model first to generate metrics.json.")

st.divider()

# -----------------------------
# Prediction section
# -----------------------------
st.subheader("🔮 Make a Prediction")

col1, col2, col3 = st.columns(3)

with col1:
    temp_max = st.number_input(
        "Max Temperature",
        min_value=-20.0,
        max_value=60.0,
        value=20.0,
        step=1.0
    )

with col2:
    temp_min = st.number_input(
        "Min Temperature",
        min_value=-30.0,
        max_value=50.0,
        value=10.0,
        step=1.0
    )

with col3:
    precipitation = st.number_input(
        "Precipitation",
        min_value=0.0,
        max_value=500.0,
        value=2.0,
        step=1.0
    )

if st.button("Predict"):
    data = {
        "temp_max": float(temp_max),
        "temp_min": float(temp_min),
        "precipitation": float(precipitation)
    }

    try:
        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json=data,
            timeout=10
        )

        result = response.json()

        if response.status_code == 200:
            if result.get("status") == "success":
                st.success(result["result"])

                res_col1, res_col2, res_col3 = st.columns(3)
                res_col1.metric("Prediction", result["prediction"])
                res_col2.metric("Model Version", result["model_version"])
                res_col3.metric("Status", result["status"])

            else:
                st.error(result.get("message", "Something went wrong."))

        else:
            st.error(f"API error: {result}")

    except Exception as e:
        st.error(f"Could not connect to API: {e}")
