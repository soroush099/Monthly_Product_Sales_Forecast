"""
Streamlit App Entry Point.
Run with: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import sys
import os
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
# نکته مهم: train_model را اضافه کردیم تا بتوانیم مدل بسازیم
from src.models.model_training import load_model, train_model
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


def get_model_smart(monthly_data, feature_cols):
    """
    Load model if exists and matches features.
    If not exists OR mismatch -> Train new model.
    """
    model = None
    retrain_needed = False

    # 1. Check if model file exists
    if not os.path.exists(MODEL_SAVE_PATH):
        print("⚠️ Model file not found. Starting initial training...")
        retrain_needed = True
    else:
        # 2. Try to load and check feature compatibility
        try:
            model = load_model(MODEL_SAVE_PATH)
            booster_features = model.get_booster().feature_names

            if booster_features != feature_cols:
                print("⚠️ Feature mismatch detected (Code changed). Retraining...")
                retrain_needed = True
            else:
                print("✅ Model loaded successfully.")

        except Exception as e:
            print(f"⚠️ Error loading model: {e}. Retraining...")
            retrain_needed = True

    # 3. Train if needed
    if retrain_needed:
        with st.spinner("⚙️ Training optimized XGBoost model... (This runs once)"):
            # Prepare data: Drop rows with NaNs (caused by lags) for training
            train_df = monthly_data.dropna(subset=feature_cols)

            X = train_df[feature_cols]
            y = train_df['MainQty']

            # Train and Save
            model = train_model(X, y)
            st.success("✅ Model trained and saved!")

    return model


def main():
    st.set_page_config(page_title="Sales Forecast", layout="wide")
    st.title("📊 Sales Forecasting System")

    # Force Retrain Button
    if st.sidebar.button("♻️ Force Retrain Model"):
        if os.path.exists(MODEL_SAVE_PATH):
            os.remove(MODEL_SAVE_PATH)
        st.cache_data.clear()
        st.rerun()

    # Load data
    with st.spinner("Loading data..."):
        monthly, feature_cols, split_date = load_all_data()

        # Smart Model Loading
        model = get_model_smart(monthly, feature_cols)

    # Generate forecast
    forecast_df = forecast_next_month(model, monthly, feature_cols)

    # Run dashboard
    run_dashboard(monthly, feature_cols, forecast_df, split_date)


if __name__ == "__main__":
    main()