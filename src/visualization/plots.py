"""
Visualization module for sales forecasting.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Tuple
from pathlib import Path
from xgboost import XGBRegressor

from src.config.base_config import FIGURES_DIR


def setup_plot_style():
    """Set up matplotlib style for consistent plots."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 11


def plot_historical_and_predictions(
        model: XGBRegressor,
        monthly: pd.DataFrame,
        feature_cols: List[str],
        goods_id: int,
        forecast_df: pd.DataFrame = None,
        split_date: pd.Timestamp = None,
        save_path: str = None,
        show: bool = True
) -> None:
    """
    Plot historical sales, model predictions, and forecast.

    Parameters
    ----------
    model : XGBRegressor
        Trained model.
    monthly : pd.DataFrame
        Monthly dataframe.
    feature_cols : list
        List of feature column names.
    goods_id : int
        GoodsID to plot.
    forecast_df : pd.DataFrame, optional
        Forecast dataframe with future predictions.
    split_date : pd.Timestamp, optional
        Train/test split date for vertical line.
    save_path : str, optional
        Path to save the figure.
    show : bool
        Whether to display the plot.
    """
    # Prepare data
    product_df = monthly[monthly['GoodsID'] == goods_id].copy()
    product_df = product_df.sort_values('date')
    product_df['predicted'] = model.predict(product_df[feature_cols])

    # Plot
    plt.figure(figsize=(12, 6))

    # Actual sales
    plt.plot(
        product_df['date'],
        product_df['MainQty'],
        label='Actual Sales',
        marker='o',
        color='steelblue',
        linewidth=2
    )

    # Model predictions
    plt.plot(
        product_df['date'],
        product_df['predicted'],
        label='Model Prediction',
        linestyle='--',
        marker='.',
        color='orange',
        linewidth=2
    )

    # Add forecast for next month
    if forecast_df is not None:
        forecast_row = forecast_df[forecast_df['GoodsID'] == goods_id]
        if not forecast_row.empty:
            plt.scatter(
                forecast_row['forecast_month'],
                forecast_row['predicted_next_month'],
                color='red',
                s=150,
                marker='*',
                label='Next Month Forecast',
                zorder=5
            )

    # Split date marker
    if split_date is not None:
        plt.axvline(
            split_date,
            color='gray',
            linestyle=':',
            linewidth=2,
            label='Train/Test Split'
        )

    plt.title(f"GoodsID {goods_id} — Historical, Predicted, and Forecasted Sales")
    plt.xlabel("Date")
    plt.ylabel("Sales Quantity")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Figure saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_sample_time_series(
        monthly: pd.DataFrame,
        goods_id: int = None,
        save_path: str = None,
        show: bool = True
) -> None:
    """
    Plot raw time series for a sample product.

    Parameters
    ----------
    monthly : pd.DataFrame
        Monthly dataframe.
    goods_id : int, optional
        GoodsID to plot. If None, selects randomly.
    save_path : str, optional
        Path to save the figure.
    show : bool
        Whether to display the plot.
    """
    if goods_id is None:
        goods_id = monthly['GoodsID'].sample(1).iloc[0]

    product_df = monthly[monthly['GoodsID'] == goods_id].copy()
    product_df = product_df.sort_values('date')

    plt.figure(figsize=(12, 6))
    plt.plot(
        product_df['date'],
        product_df['MainQty'],
        label='Actual Sales',
        marker='o',
        color='steelblue',
        linewidth=2
    )

    plt.title(f"Sales Time Series — GoodsID {goods_id}")
    plt.xlabel("Date")
    plt.ylabel("Sales Quantity")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()


def plot_feature_importance(
        model: XGBRegressor,
        feature_cols: List[str],
        top_n: int = 20,
        save_path: str = None,
        show: bool = True
) -> None:
    """
    Plot feature importance from the trained model.

    Parameters
    ----------
    model : XGBRegressor
        Trained model.
    feature_cols : list
        List of feature column names.
    top_n : int
        Number of top features to show.
    save_path : str, optional
        Path to save the figure.
    show : bool
        Whether to display the plot.
    """
    importance = model.feature_importances_
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': importance
    }).sort_values('importance', ascending=True).tail(top_n)

    plt.figure(figsize=(10, 8))
    plt.barh(importance_df['feature'], importance_df['importance'], color='steelblue')
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title(f"Top {top_n} Feature Importance")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()


def plot_actual_vs_predicted(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_path: str = None,
        show: bool = True
) -> None:
    """
    Create actual vs predicted scatter plot.

    Parameters
    ----------
    y_true : array-like
        True values.
    y_pred : array-like
        Predicted values.
    save_path : str, optional
        Path to save the figure.
    show : bool
        Whether to display the plot.
    """
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.5, color='steelblue')

    # Perfect prediction line
    max_val = max(max(y_true), max(y_pred))
    plt.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect Prediction')

    plt.xlabel("Actual Sales")
    plt.ylabel("Predicted Sales")
    plt.title("Actual vs Predicted Sales")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()


def plot_residuals(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_path: str = None,
        show: bool = True
) -> None:
    """
    Plot residuals distribution and vs predicted values.

    Parameters
    ----------
    y_true : array-like
        True values.
    y_pred : array-like
        Predicted values.
    save_path : str, optional
        Path to save the figure.
    show : bool
        Whether to display the plot.
    """
    residuals = np.array(y_true) - np.array(y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    axes[0].hist(residuals, bins=50, color='steelblue', edgecolor='white')
    axes[0].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[0].set_xlabel("Residual")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Residuals Distribution")

    # Residuals vs Predicted
    axes[1].scatter(y_pred, residuals, alpha=0.5, color='steelblue')
    axes[1].axhline(0, color='red', linestyle='--', linewidth=2)
    axes[1].set_xlabel("Predicted Values")
    axes[1].set_ylabel("Residuals")
    axes[1].set_title("Residuals vs Predicted Values")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()