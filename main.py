"""
Main entry point for the Monthly Product Sales Forecast pipeline.
"""
import warnings
import os
import pandas as pd

warnings.filterwarnings('ignore')

from src.config.base_config import (
    RAW_DATA_PATH,
    MODEL_SAVE_PATH,
    FIGURES_DIR,
    FORECAST_RESULTS_PATH
)
from src.data.data_loader import load_and_clean_data
from src.features.feature_engineering import build_features
from src.models.model_training import (
    train_model,
    save_model,
    load_model
)
from src.models.evaluation import evaluate_model, get_predictions
from src.models.forecasting import (
    forecast_next_month,
    save_forecast_results
)
from src.visualization.plots import (
    plot_historical_and_predictions,
    plot_feature_importance,
    plot_actual_vs_predicted,
    plot_residuals
)
from src.utils.helpers import (
    set_seed,
    ensure_dir,
    sample_goods_id,
    print_section
)


def train_test_split_temporal(monthly, feature_cols, test_months=2):
    """
    Split data temporally for time series.

    Parameters
    ----------
    monthly : DataFrame
        Monthly aggregated data
    feature_cols : list
        Feature column names
    test_months : int
        Number of months to use for testing

    Returns
    -------
    tuple : X_train, X_test, y_train, y_test, split_date
    """
    # Calculate split date
    split_date = monthly['date'].max() - pd.DateOffset(months=test_months)

    # Split data
    train_df = monthly[monthly['date'] <= split_date].dropna(subset=feature_cols)
    test_df = monthly[monthly['date'] > split_date].dropna(subset=feature_cols)

    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]
    y_train = train_df['MainQty']
    y_test = test_df['MainQty']

    print(f"Split Date: {split_date.strftime('%Y-%m')}")
    print(f"Train: {len(X_train):,} records | Test: {len(X_test):,} records")

    return X_train, X_test, y_train, y_test, split_date


def get_model_smart(X_train, y_train, feature_cols):
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

            if booster_features != list(feature_cols):
                print("⚠️ Feature mismatch detected (Code changed). Retraining...")
                retrain_needed = True
            else:
                print("✅ Model loaded successfully.")

        except Exception as e:
            print(f"⚠️ Error loading model: {e}. Retraining...")
            retrain_needed = True

    # 3. Train if needed
    if retrain_needed:
        print("⚙️ Training optimized XGBoost model...")
        model = train_model(X_train, y_train)
        save_model(model, MODEL_SAVE_PATH)
        print("✅ Model trained and saved!")

    return model


def main(
        data_path: str = None,
        save_plots: bool = True
):
    """
    Run the complete sales forecasting pipeline.

    Parameters
    ----------
    data_path : str, optional
        Path to raw data CSV file.
    save_plots : bool
        Whether to save generated plots.
    """
    # Setup
    set_seed(42)
    ensure_dir(FIGURES_DIR)

    # ─────────────────────────────────────────────────────────
    print_section("1. LOADING AND CLEANING DATA")
    # ─────────────────────────────────────────────────────────
    df = load_and_clean_data(data_path or RAW_DATA_PATH)

    # ─────────────────────────────────────────────────────────
    print_section("2. FEATURE ENGINEERING")
    # ─────────────────────────────────────────────────────────
    monthly, feature_cols = build_features(df)

    # ─────────────────────────────────────────────────────────
    print_section("3. TRAIN/TEST SPLIT")
    # ─────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test, split_date = train_test_split_temporal(
        monthly, feature_cols
    )

    # ─────────────────────────────────────────────────────────
    print_section("4. MODEL TRAINING")
    # ─────────────────────────────────────────────────────────
    model = get_model_smart(X_train, y_train, feature_cols)

    # ─────────────────────────────────────────────────────────
    print_section("5. MODEL EVALUATION")
    # ─────────────────────────────────────────────────────────
    metrics = evaluate_model(model, X_test, y_test)

    # Predictions for visualization
    y_pred = get_predictions(model, X_test)

    # ─────────────────────────────────────────────────────────
    print_section("6. FORECASTING")
    # ─────────────────────────────────────────────────────────
    forecast_df = forecast_next_month(model, monthly, feature_cols)
    save_forecast_results(forecast_df)

    # ─────────────────────────────────────────────────────────
    print_section("7. VISUALIZATION")
    # ─────────────────────────────────────────────────────────
    # Plot feature importance
    plot_feature_importance(
        model, feature_cols, top_n=20,
        save_path=FIGURES_DIR / "feature_importance.png" if save_plots else None
    )

    # Plot actual vs predicted
    plot_actual_vs_predicted(
        y_test, y_pred,
        save_path=FIGURES_DIR / "actual_vs_predicted.png" if save_plots else None
    )

    # Plot residuals
    plot_residuals(
        y_test, y_pred,
        save_path=FIGURES_DIR / "residuals.png" if save_plots else None
    )

    # Sample product visualization
    sample_id = sample_goods_id(monthly, min_records=12)
    print(f"\nSample GoodsID for visualization: {sample_id}")

    plot_historical_and_predictions(
        model, monthly, feature_cols, sample_id,
        forecast_df=forecast_df,
        split_date=split_date,
        save_path=FIGURES_DIR / f"forecast_goods_{sample_id}.png" if save_plots else None
    )

    # ─────────────────────────────────────────────────────────
    print_section("PIPELINE COMPLETE", char="✓")
    # ─────────────────────────────────────────────────────────

    print(f"""
    Summary:
    --------
    • Products: {monthly['GoodsID'].nunique():,}
    • Date Range: {monthly['date'].min().strftime('%Y-%m')} to {monthly['date'].max().strftime('%Y-%m')}
    • Model Performance (Test Set):
      - MAE:  {metrics['MAE']:.2f}
      - RMSE: {metrics['RMSE']:.2f}
      - R²:   {metrics['R2']:.3f}
      - MAPE: {metrics['MAPE']:.2f}%
    • Forecast saved to: {FORECAST_RESULTS_PATH}
    • Model saved to: {MODEL_SAVE_PATH}
    """)

    return model, monthly, feature_cols, forecast_df


if __name__ == "__main__":
    model, monthly, feature_cols, forecast_df = main(
        save_plots=True
    )