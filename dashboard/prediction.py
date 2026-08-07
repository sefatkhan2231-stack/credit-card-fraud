import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ==========================================
# Model Path
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "saved_models" / "random_forest.pkl"


# ==========================================
# Load Model
# ==========================================

@st.cache_resource(show_spinner=False)
def load_model():
    """
    Load the trained Random Forest pipeline.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    return model


# ==========================================
# Validate Dataset
# ==========================================

def validate_features(df: pd.DataFrame):
    """
    Validate uploaded dataset before prediction.
    """

    if df.empty:
        raise ValueError("Uploaded dataset is empty.")

    if df.isnull().sum().sum() > 0:
        raise ValueError(
            "Dataset contains missing values. Please clean the data before prediction."
        )


# ==========================================
# Prediction
# ==========================================

def make_prediction(model, df):
    """
    Predict fraud and return prediction dataframe.
    """

    validate_features(df)

    prediction = model.predict(df)

    probability = model.predict_proba(df)[:, 1]

    result = df.copy()

    result["Prediction"] = prediction

    result["Fraud Probability"] = probability

    result["Prediction"] = result["Prediction"].map({
        0: "Legitimate",
        1: "Fraud"
    })

    result["Fraud Probability"] = (
        result["Fraud Probability"] * 100
    ).round(2)

    return result


# ==========================================
# Prediction Summary
# ==========================================

def prediction_summary(prediction_df):
    """
    Generate summary statistics.
    """

    total = len(prediction_df)

    fraud = (prediction_df["Prediction"] == "Fraud").sum()

    legitimate = (
        prediction_df["Prediction"] == "Legitimate"
    ).sum()

    fraud_rate = round(
        fraud / total * 100,
        2
    )

    return {
        "Total Transactions": total,
        "Fraud": fraud,
        "Legitimate": legitimate,
        "Fraud Rate (%)": fraud_rate
    }