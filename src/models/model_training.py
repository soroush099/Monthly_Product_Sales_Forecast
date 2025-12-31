import pandas as pd
import numpy as np
import os
import optuna
import xgboost as xgb
from typing import Dict, Any
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

from src.config.base_config import (
    MODEL_SAVE_PATH,
    RANDOM_STATE,
    STATIC_PARAMS
)


def optimize_hyperparameters(x: pd.DataFrame, y: pd.Series, n_trials: int = 20) -> Dict[str, Any]:
    print(f"🧠 Starting Hyperparameter Optimization ({n_trials} trials)...")

    # === SAFETY CHECK: CLEAN DATA BEFORE OPTIMIZATION ===
    # تبدیل تمام بی‌نهایت‌ها به NaN و سپس پر کردن با صفر
    x = x.replace([np.inf, -np.inf], np.nan).fillna(0)

    # ====================================================

    def objective(trial):
        param_suggestions = {
            'tweedie_variance_power': trial.suggest_float('tweedie_variance_power', 1.1, 1.9),
            'n_estimators': trial.suggest_int('n_estimators', 500, 2000, step=100),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        }

        params = {**STATIC_PARAMS, **param_suggestions}
        tscv = TimeSeriesSplit(n_splits=3)
        scores = []

        for train_idx, val_idx in tscv.split(x):
            x_tr, x_val = x.iloc[train_idx], x.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model = xgb.XGBRegressor(**params)
            model.fit(x_tr, y_tr, verbose=False)

            preds = model.predict(x_val)
            rmse = np.sqrt(mean_squared_error(y_val, preds))
            scores.append(rmse)

        return np.mean(scores)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)

    return study.best_params


def train_model(x: pd.DataFrame, y: pd.Series) -> xgb.XGBRegressor:
    # === SAFETY CHECK ===
    x = x.replace([np.inf, -np.inf], np.nan).fillna(0)
    # ====================

    # مرحله 1: هایپر تیونینگ
    best_tuned_params = optimize_hyperparameters(x, y, n_trials=20)

    final_params = {**STATIC_PARAMS, **best_tuned_params}

    print("🔄 Training Final Model with Optimized Parameters...")

    # مرحله 2: ساخت مدل نهایی
    model = xgb.XGBRegressor(**final_params)
    model.fit(x, y)

    save_model(model)
    return model


def save_model(model: xgb.XGBRegressor, filepath: str = None) -> None:
    if filepath is None:
        filepath = MODEL_SAVE_PATH

    os.makedirs(os.path.dirname(str(filepath)), exist_ok=True)
    model.save_model(str(filepath))
    print(f"✅ Model saved to {filepath}")


def load_model(filepath: str = None) -> xgb.XGBRegressor:
    if filepath is None:
        filepath = MODEL_SAVE_PATH

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found at {filepath}")

    model = xgb.XGBRegressor()
    model.load_model(str(filepath))
    print(f"✅ Model loaded from {filepath}")

    return model
