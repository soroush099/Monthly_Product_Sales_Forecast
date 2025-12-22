"""
Streamlit App Entry Point.
Run with: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config.base_config import (
    RAW_DATA_PATH,
    MODEL_SAVE_PATH,
    FORECAST_RESULTS_PATH
)
from src.data.data_loader import load_and_clean_data
from src.features.feature_engineering import build_features
from src.models.model_training import load_model
from src.models.forecasting import forecast_next_month
from src.visualization.dashboard import run_dashboard


@st.cache_data
def load_all_data():
    """Load and cache all data."""
    # Load raw data
    df = load_and_clean_data(RAW_DATA_PATH)

    # Build features
    monthly, feature_cols = build_features(df)

    # Calculate split date
    split_date = monthly['date'].max() - pd.DateOffset(months=2)

    return monthly, feature_cols, split_date


@st.cache_resource
def load_cached_model():
    """Load and cache model."""
    return load_model(MODEL_SAVE_PATH)


def main():
    # Load data
    with st.spinner("Loading data..."):
        monthly, feature_cols, split_date = load_all_data()
        model = load_cached_model()

    # Generate forecast
    forecast_df = forecast_next_month(model, monthly, feature_cols)

    # Run dashboard
    run_dashboard(monthly, feature_cols, forecast_df, split_date)


if __name__ == "__main__":
    main()
