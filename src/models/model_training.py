"""
Model training module for XGBoost regressor.
"""
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any, Optional
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

from src.config.base_config import (
    PARAM_GRID,
    BEST_PARAMS,
    RANDOM_STATE,
    CV_SPLITS,
    TEST_MONTHS,
    MODEL_SAVE_PATH
)


def train_test_split_temporal(
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = 'MainQty',
        test_months: int = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Timestamp]:
    """
    Split data into train and test sets chronologically.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered dataframe with 'date' column.
    feature_cols : list
        List of feature column names.
    target_col : str
        Name of target column.
    test_months : int, optional
        Number of months to hold out for testing.

    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test, split_date)
    """
    if test_months is None:
        test_months = TEST_MONTHS

    X = df[feature_cols]
    y = df[target_col]

    split_date = df['date'].max() - pd.DateOffset(months=test_months)

    train_idx = df['date'] <= split_date
    test_idx = df['date'] > split_date

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    print(f"✅ Train/Test Split:")
    print(f"   Split date: {split_date.strftime('%Y-%m-%d')}")
    print(f"   Train size: {len(X_train):,}")
    print(f"   Test size: {len(X_test):,}")

    return X_train, X_test, y_train, y_test, split_date


def train_with_grid_search(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        param_grid: Dict[str, Any] = None,
        cv_splits: int = None,
        verbose: int = 1
) -> Tuple[XGBRegressor, Dict[str, Any]]:
    """
    Train XGBoost model with GridSearchCV.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target.
    param_grid : dict, optional
        Hyperparameter grid.
    cv_splits : int, optional
        Number of cross-validation folds.
    verbose : int
        Verbosity level.

    Returns
    -------
    tuple
        (best_model, best_params)
    """
    if param_grid is None:
        param_grid = PARAM_GRID
    if cv_splits is None:
        cv_splits = CV_SPLITS

    tscv = TimeSeriesSplit(n_splits=cv_splits)

    xgb_model = XGBRegressor(
        objective='reg:squarederror',
        random_state=RANDOM_STATE
    )

    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        cv=tscv,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        verbose=verbose
    )

    print("🔄 Starting GridSearchCV...")
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    print(f"✅ Best parameters: {best_params}")

    return best_model, best_params


def train_with_best_params(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        params: Dict[str, Any] = None
) -> XGBRegressor:
    """
    Train XGBoost model with pre-defined best parameters.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target.
    params : dict, optional
        Model parameters.

    Returns
    -------
    XGBRegressor
        Trained model.
    """
    if params is None:
        params = BEST_PARAMS

    model = XGBRegressor(
        objective='reg:squarederror',
        random_state=RANDOM_STATE,
        **params
    )

    print("🔄 Training model with best parameters...")
    model.fit(X_train, y_train)
    print("✅ Model training complete")

    return model


def save_model(model: XGBRegressor, filepath: str = None) -> None:
    """
    Save trained model to file.

    Parameters
    ----------
    model : XGBRegressor
        Trained model.
    filepath : str, optional
        Path to save the model.
    """
    if filepath is None:
        filepath = MODEL_SAVE_PATH

    model.save_model(str(filepath))
    print(f"✅ Model saved to {filepath}")


def load_model(filepath: str = None) -> XGBRegressor:
    """
    Load trained model from file.

    Parameters
    ----------
    filepath : str, optional
        Path to the model file.

    Returns
    -------
    XGBRegressor
        Loaded model.
    """
    if filepath is None:
        filepath = MODEL_SAVE_PATH

    model = XGBRegressor()
    model.load_model(str(filepath))
    print(f"✅ Model loaded from {filepath}")

    return model