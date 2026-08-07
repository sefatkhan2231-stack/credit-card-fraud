import streamlit as st


def page_config():
    """
    Configure the Streamlit page.
    """

    st.set_page_config(
        page_title="Credit Card Fraud Detection Dashboard",
        page_icon="💳",
        layout="wide",
        initial_sidebar_state="expanded"
    )