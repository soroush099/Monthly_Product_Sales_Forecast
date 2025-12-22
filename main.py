"""
Main entry point for the Monthly Product Sales Forecast pipeline.
"""
import warnings

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
    train_test_split_temporal,
    train_with_grid_search,
    train_with_best_params,
    save_model,
    load_model
)
from src.models.evaluation import evaluate_model, get_predictions
from src.models.forecasting import (
    forecast_next_month,
    save_forecast_results,
    get_historical_with_predictions
)
from src.visualization.plots import (
    plot_historical_and_predictions,
    plot_feature_importance,
    plot_actual_vs_predicted,
    plot_residuals,
    plot_sample_time_series
)
from src.utils.helpers import (
    set_seed,
    ensure_dir,
    sample_goods_id,
    print_section
)


def main(
        data_path: str = None,
        run_grid_search: bool = False,
        save_plots: bool = True
):
    """
    Run the complete sales forecasting pipeline.

    Parameters
    ----------
    data_path : str, optional
        Path to raw data CSV file.
    run_grid_search : bool
        Whether to run full grid search (slow) or use best params.
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
    if run_grid_search:
        model, best_params = train_with_grid_search(X_train, y_train, verbose=2)
        print(f"Best parameters: {best_params}")
    else:
        model = train_with_best_params(X_train, y_train)

    # Save model
    save_model(model, MODEL_SAVE_PATH)

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
    • Forecast saved to: {FORECAST_RESULTS_PATH}
    • Model saved to: {MODEL_SAVE_PATH}
    """)

    return model, monthly, feature_cols, forecast_df


if __name__ == "__main__":
    model, monthly, feature_cols, forecast_df = main(
        run_grid_search=False,  # Set True for full hyperparameter search
        save_plots=True
    )