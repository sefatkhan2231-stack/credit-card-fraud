import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    auc
)


# ==========================================
# Common Layout
# ==========================================

def update_layout(fig):

    fig.update_layout(
        template="plotly_white",
        title_x=0.5,
        margin=dict(l=30, r=30, t=60, b=30),
        legend_title_text=""
    )

    return fig


# ==========================================
# Model Comparison
# ==========================================

def model_comparison(result_df):

    fig = px.bar(
        result_df,
        x="Model",
        y="ROC AUC",
        color="Model",
        text="ROC AUC",
        title="Model Comparison (ROC-AUC)"
    )

    fig.update_traces(texttemplate="%{text:.3f}")

    return update_layout(fig)


# ==========================================
# Feature Importance
# ==========================================

def feature_importance(feature_df):

    top = (
        feature_df
        .sort_values(
            by="Importance",
            ascending=False
        )
        .head(15)
    )

    fig = px.bar(
        top,
        x="Importance",
        y="Feature",
        orientation="h",
        text="Importance",
        title="Top 15 Feature Importance"
    )

    fig.update_layout(
        yaxis=dict(autorange="reversed")
    )

    fig.update_traces(
        texttemplate="%{text:.3f}"
    )

    return update_layout(fig)


# ==========================================
# Fraud Distribution
# ==========================================

def fraud_distribution(prediction_df):

    count = (
        prediction_df["Prediction"]
        .value_counts()
        .reset_index()
    )

    count.columns = ["Prediction", "Count"]

    fig = px.pie(
        count,
        names="Prediction",
        values="Count",
        hole=0.45,
        title="Fraud Prediction Distribution"
    )

    return update_layout(fig)


# ==========================================
# Probability Distribution
# ==========================================

def probability_distribution(prediction_df):

    fig = px.histogram(
        prediction_df,
        x="Fraud Probability",
        nbins=40,
        title="Fraud Probability Distribution"
    )

    fig.update_layout(
        bargap=0.05
    )

    return update_layout(fig)


# ==========================================
# ROC Curve
# ==========================================

def roc_chart(y_true, y_prob):

    fpr, tpr, _ = roc_curve(
        y_true,
        y_prob
    )

    roc_auc = auc(
        fpr,
        tpr
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name=f"AUC = {roc_auc:.3f}"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random",
            line=dict(dash="dash")
        )
    )

    fig.update_layout(
        title="ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate"
    )

    return update_layout(fig)


# ==========================================
# Precision Recall Curve
# ==========================================

def pr_chart(y_true, y_prob):

    precision, recall, _ = precision_recall_curve(
        y_true,
        y_prob
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=recall,
            y=precision,
            mode="lines",
            name="Precision-Recall"
        )
    )

    fig.update_layout(
        title="Precision-Recall Curve",
        xaxis_title="Recall",
        yaxis_title="Precision"
    )

    return update_layout(fig)


# ==========================================
# Confusion Matrix
# ==========================================

def confusion_matrix_chart(cm):

    fig = px.imshow(
        cm,
        text_auto=True,
        color_continuous_scale="Blues",
        title="Confusion Matrix"
    )

    fig.update_layout(
        xaxis_title="Predicted Label",
        yaxis_title="Actual Label"
    )

    return update_layout(fig)