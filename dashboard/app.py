import streamlit as st
import pandas as pd

from config import page_config
from utils import (
    load_csv,
    load_reports,
    dataset_summary,
    validate_dataset,
    convert_to_csv
)

from prediction import (
    load_model,
    make_prediction,
    prediction_summary
)

from charts import (
    model_comparison,
    feature_importance,
    fraud_distribution,
    probability_distribution,
    roc_chart,
    pr_chart,
    confusion_matrix_chart
)


##########################################################
# Page Configuration
##########################################################

page_config()


##########################################################
# Custom CSS
##########################################################

def load_css():

    with open("./assets/style.css") as f:

        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )

load_css()


##########################################################
# Title
##########################################################

st.title("💳 Credit Card Fraud Detection Dashboard")

st.caption(
    "Detect fraudulent credit card transactions using a Machine Learning model."
)

st.markdown("---")


##########################################################
# Sidebar
##########################################################

st.sidebar.title("⚙ Dashboard")

st.sidebar.markdown("---")

uploaded_file = st.sidebar.file_uploader(

    "Upload Transaction CSV",

    type=["csv"]

)

st.sidebar.markdown("---")

st.sidebar.subheader("Model")

st.sidebar.success("Random Forest")

st.sidebar.write("SMOTE")

st.sidebar.write("ROC-AUC : 0.97")

st.sidebar.write("Average Precision : 0.87")

st.sidebar.markdown("---")


##########################################################
# No Dataset
##########################################################

if uploaded_file is None:

    st.info("👈 Upload a CSV file from the sidebar.")

    st.stop()


##########################################################
# Load Dataset
##########################################################

df = load_csv(uploaded_file)


##########################################################
# Validate Dataset
##########################################################

valid, message = validate_dataset(df)

if not valid:

    st.error(message)

    st.stop()


##########################################################
# Dataset Statistics
##########################################################

stats = dataset_summary(df)

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Rows",
    f"{stats['Rows']:,}"
)

col2.metric(
    "Columns",
    stats["Columns"]
)

col3.metric(
    "Missing Values",
    stats["Missing Values"]
)

col4.metric(
    "Duplicate Rows",
    stats["Duplicate Rows"]
)


##########################################################
# Dataset Preview
##########################################################

with st.expander("Dataset Preview", expanded=False):

    st.dataframe(

        df.head(20),

        use_container_width=True

    )


st.markdown("---")


# ==========================================
# Session State
# ==========================================
# NOTE: everything the dashboard needs to keep across reruns lives
# here. Streamlit reruns the *entire* script top-to-bottom on every
# widget interaction (clicking a tab, changing the filter, clicking
# download, etc). A plain local variable created inside
# `if st.sidebar.button(...):` only exists during the single rerun
# where the button was actually clicked — on every later rerun that
# `if` is False again and the variable is simply gone. Anything that
# needs to survive to later reruns must be stored in session_state.

if "prediction_df" not in st.session_state:
    st.session_state.prediction_df = None

if "feature_report" not in st.session_state:
    st.session_state.feature_report = None

if "result_report" not in st.session_state:
    st.session_state.result_report = None

if "best_params" not in st.session_state:
    st.session_state.best_params = None

if "y_test" not in st.session_state:
    st.session_state.y_test = None

if "y_prob" not in st.session_state:
    st.session_state.y_prob = None

if "cm" not in st.session_state:
    st.session_state.cm = None


##########################################################
# Prediction
##########################################################
# This block's only job is to run the model and stash the results in
# session_state. It does NOT render the filter/summary/tabs — that
# happens below, unconditionally, so it keeps working on every rerun
# (filtering, tab switches, downloads) rather than only the instant
# after the click.

if st.sidebar.button(

    "🚀 Run Prediction",

    use_container_width=True

):

    with st.spinner("Running prediction..."):

        model = load_model()

        st.session_state.prediction_df = make_prediction(

            model,

            df

        )

    (
        st.session_state.feature_report,
        st.session_state.result_report,
        st.session_state.best_params,
        st.session_state.y_test,
        st.session_state.y_prob,
        st.session_state.cm
    ) = load_reports()

    st.success("Prediction completed successfully!")


##########################################################
# Nothing predicted yet
##########################################################

if st.session_state.prediction_df is None:

    st.info("👈 Click **Run Prediction** in the sidebar to get started.")

    st.stop()


##########################################################
# Filter
##########################################################

filter_option = st.selectbox(

    "Prediction Filter",

    [

        "All",

        "Fraud",

        "Legitimate"

    ]

)

if filter_option != "All":

    prediction_df = st.session_state.prediction_df[

        st.session_state.prediction_df["Prediction"] == filter_option

    ]

else:

    prediction_df = st.session_state.prediction_df


##########################################################
# Sort
##########################################################

prediction_df = prediction_df.sort_values(

    by="Fraud Probability",

    ascending=False

)


##########################################################
# Summary
##########################################################

summary = prediction_summary(

    prediction_df

)


##########################################################
# KPI Cards
##########################################################

c1, c2, c3, c4 = st.columns(4)

c1.metric(

    "💳 Total",

    summary["Total Transactions"]

)

c2.metric(

    "🚨 Fraud",

    summary["Fraud"]

)

c3.metric(

    "✅ Legitimate",

    summary["Legitimate"]

)

c4.metric(

    "📊 Fraud Rate",

    f"{summary['Fraud Rate (%)']}%"

)

st.progress(

    min(summary["Fraud Rate (%)"] / 100, 1.0)

)

st.markdown("---")


##########################################################
# Dashboard Tabs
##########################################################

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📋 Predictions",
        "📊 Model Evaluation",
        "📈 Feature Analysis",
        "📄 Summary"
    ]
)

##########################################################
# TAB 1
# Predictions
##########################################################

with tab1:

    st.subheader("Prediction Results")

    st.dataframe(
        prediction_df,
        use_container_width=True,
        height=500
    )

    csv = convert_to_csv(prediction_df)

    st.download_button(
        label="⬇ Download Predictions",
        data=csv,
        file_name="fraud_predictions.csv",
        mime="text/csv",
        use_container_width=True
    )

##########################################################
# TAB 2
# Model Evaluation
##########################################################

with tab2:

    st.subheader("Model Comparison")

    st.plotly_chart(
        model_comparison(st.session_state.result_report),
        use_container_width=True
    )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        st.plotly_chart(
            roc_chart(
                st.session_state.y_test,
                st.session_state.y_prob
            ),
            use_container_width=True
        )

    with col2:

        st.plotly_chart(
            pr_chart(
                st.session_state.y_test,
                st.session_state.y_prob
            ),
            use_container_width=True
        )

    st.markdown("---")

    st.subheader("Confusion Matrix")

    st.plotly_chart(
        confusion_matrix_chart(st.session_state.cm),
        use_container_width=True
    )

##########################################################
# TAB 3
# Feature Analysis
##########################################################

with tab3:

    st.subheader("Feature Importance")

    st.plotly_chart(
        feature_importance(
            st.session_state.feature_report
        ),
        use_container_width=True
    )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        st.plotly_chart(
            fraud_distribution(
                prediction_df
            ),
            use_container_width=True
        )

    with col2:

        st.plotly_chart(
            probability_distribution(
                prediction_df
            ),
            use_container_width=True
        )


##########################################################
# TAB 4
# Summary & Reports
##########################################################

with tab4:

    st.subheader("📄 Prediction Summary")

    summary_df = pd.DataFrame(
        {
            "Metric": [
                "Total Transactions",
                "Fraud Detected",
                "Legitimate Transactions",
                "Fraud Rate (%)"
            ],
            "Value": [
                summary["Total Transactions"],
                summary["Fraud"],
                summary["Legitimate"],
                summary["Fraud Rate (%)"]
            ]
        }
    )

    st.dataframe(
        summary_df,
        hide_index=True,
        use_container_width=True
    )

    st.markdown("---")

    st.subheader("⚙ Best Hyperparameters")

    st.dataframe(
        st.session_state.best_params,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")

    st.subheader("📊 Model Performance")

    st.dataframe(
        st.session_state.result_report,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")

    st.subheader("📈 Top 10 High-Risk Transactions")

    top10 = prediction_df.sort_values(
        by="Fraud Probability",
        ascending=False
    ).head(10)

    st.dataframe(
        top10,
        use_container_width=True
    )

    st.markdown("---")

    st.subheader("ℹ About This Project")

    st.info(
        """
### Credit Card Fraud Detection Dashboard

This dashboard predicts fraudulent credit card transactions using a
**Random Forest Classifier** trained on an imbalanced dataset.

### Machine Learning Pipeline

- Data Cleaning
- Train/Test Split
- SMOTE (Synthetic Minority Over-sampling Technique)
- Random Forest Classifier
- Hyperparameter Tuning (RandomizedSearchCV)

### Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Precision-Recall Curve
- Confusion Matrix

### Dashboard Features

- Upload CSV
- Fraud Prediction
- Fraud Probability
- Interactive Charts
- Feature Importance
- Download Predictions
"""
    )

##########################################################
# Footer
##########################################################

st.markdown("---")

col1, col2 = st.columns([3, 1])

with col1:

    st.caption(
        "Developed by Md. Sefat Khan"
    )

with col2:

    st.caption(
        "Version 1.0"
    )