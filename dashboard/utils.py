import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path


# ==============================
# Paths
# ==============================

BASE_DIR = Path(__file__).resolve().parent.parent

REPORT_DIR = BASE_DIR / "report"


# ==============================
# Load Uploaded CSV
# ==============================

@st.cache_data(show_spinner=False)
def load_csv(uploaded_file):
    """
    Load uploaded CSV file.
    """

    df = pd.read_csv(uploaded_file)

    return df


# ==============================
# Load Evaluation Reports
# ==============================

@st.cache_data(show_spinner=False)
def load_reports():
    """
    Load all saved reports generated during model training.
    """

    feature_report = pd.read_csv(
        REPORT_DIR / "feature_importance_report.csv"
    )

    result_report = pd.read_csv(
        REPORT_DIR / "result_report.csv"
    )

    best_params = pd.read_csv(
        REPORT_DIR / "best_params.csv"
    )

    y_test = np.load(
        REPORT_DIR / "y_test.npy"
    )

    y_prob = np.load(
        REPORT_DIR / "y_prob.npy"
    )

    confusion_matrix = pd.read_csv(
        REPORT_DIR / "confusion_matrix.csv",
        index_col=0
    )

    return (
        feature_report,
        result_report,
        best_params,
        y_test,
        y_prob,
        confusion_matrix
    )


# ==============================
# Dataset Information
# ==============================

def dataset_summary(df):
    """
    Return dataset summary.
    """

    return {
        "Rows": df.shape[0],
        "Columns": df.shape[1],
        "Missing Values": int(df.isna().sum().sum()),
        "Duplicate Rows": int(df.duplicated().sum())
    }


# ==============================
# Validate Dataset
# ==============================

def validate_dataset(df):
    """
    Check whether uploaded dataset is valid.
    """

    if df.empty:
        return False, "Dataset is empty."

    if df.isnull().all().all():
        return False, "Dataset contains only missing values."

    return True, "Dataset is valid."


# ==============================
# Convert DataFrame to CSV
# ==============================

@st.cache_data(show_spinner=False)
def convert_to_csv(df):
    """
    Convert dataframe to downloadable CSV.
    """

    return df.to_csv(index=False).encode("utf-8")