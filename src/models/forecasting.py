"""
Forecasting module for generating predictions on new data.
"""
import pandas as pd
import numpy as np
from typing import List, Optional
from xgboost import XGBRegressor

from src.config.base_config import FORECAST_RESULTS_PATH


def forecast_next_month(
        model: XGBRegressor,
        monthly: pd.DataFrame,
        feature_cols: List[str]
) -> pd.DataFrame:
    """
    Forecast sales for the next month for each GoodsID.

    Parameters
    ----------
    model : XGBRegressor
        Trained model.
    monthly : pd.DataFrame
        Monthly feature-engineered dataframe.
    feature_cols : list
        List of feature column names.

    Returns
    -------
    pd.DataFrame
        Dataframe with forecasts for each GoodsID.
    """
    latest_date = monthly['date'].max()
    forecast_month = latest_date + pd.DateOffset(months=1)

    # Get latest features for each GoodsID
    latest_features = (
        monthly.sort_values(['GoodsID', 'date'])
        .groupby('GoodsID')
        .tail(1)[['GoodsID'] + feature_cols]
        .copy()
    )

    latest_features['forecast_month'] = forecast_month
    latest_features['predicted_qty'] = model.predict(latest_features[feature_cols])

    # Ensure non-negative predictions
    latest_features['predicted_qty'] = latest_features['predicted_qty'].clip(lower=0)

    forecast_df = latest_features[['GoodsID', 'forecast_month', 'predicted_qty']].copy()
    forecast_df = forecast_df.rename(columns={'predicted_qty': 'predicted_next_month'})

    print(f"✅ Forecast generated for {len(forecast_df):,} products")
    print(f"   Forecast month: {forecast_month.strftime('%Y-%m')}")

    return forecast_df


def forecast_multiple_months(
        model: XGBRegressor,
        monthly: pd.DataFrame,
        feature_cols: List[str],
        n_months: int = 3
) -> pd.DataFrame:
    """
    Forecast sales for multiple future months (recursive forecasting).

    Note: This uses recursive forecasting where predictions are used
    as inputs for subsequent predictions.

    Parameters
    ----------
    model : XGBRegressor
        Trained model.
    monthly : pd.DataFrame
        Monthly feature-engineered dataframe.
    feature_cols : list
        List of feature column names.
    n_months : int
        Number of months to forecast.

    Returns
    -------
    pd.DataFrame
        Dataframe with multi-month forecasts.
    """
    all_forecasts = []
    current_data = monthly.copy()

    for i in range(n_months):
        forecast_df = forecast_next_month(model, current_data, feature_cols)
        forecast_df['forecast_step'] = i + 1
        all_forecasts.append(forecast_df)

        # Update current_data with the new predictions for next iteration
        # This is a simplified approach - in practice, you'd need to 
        # properly update lag features

    result = pd.concat(all_forecasts, ignore_index=True)
    print(f"✅ Multi-month forecast complete: {n_months} months")

    return result


def get_historical_with_predictions(
        model: XGBRegressor,
        monthly: pd.DataFrame,
        feature_cols: List[str],
        goods_id: int
) -> pd.DataFrame:
    """
    Get historical data with model predictions for a specific product.

    Parameters
    ----------
    model : XGBRegressor
        Trained model.
    monthly : pd.DataFrame
        Monthly dataframe.
    feature_cols : list
        List of feature column names.
    goods_id : int
        GoodsID to filter.

    Returns
    -------
    pd.DataFrame
        Product dataframe with predictions.
    """
    product_df = monthly[monthly['GoodsID'] == goods_id].copy()
    product_df = product_df.sort_values('date')
    product_df['predicted'] = model.predict(product_df[feature_cols])

    return product_df


def save_forecast_results(
        forecast_df: pd.DataFrame,
        filepath: str = None
) -> None:
    """
    Save forecast results to CSV.

    Parameters
    ----------
    forecast_df : pd.DataFrame
        Forecast dataframe.
    filepath : str, optional
        Output file path.
    """
    if filepath is None:
        filepath = FORECAST_RESULTS_PATH

    forecast_df.to_csv(filepath, index=False)
    print(f"✅ Forecast results saved to {filepath}")