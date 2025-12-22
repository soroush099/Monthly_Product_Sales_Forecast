"""
Configuration settings for the sales forecasting project.
"""
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Raw data file
RAW_DATA_PATH = DATA_DIR / "ModelAllData2.csv"
# Alternative path (if needed)
# RAW_DATA_PATH = Path(r"D:\BI_NI_CI\ModelAllData2.csv")

# Processed data
PROCESSED_DATA_PATH = DATA_DIR / "processed_monthly.csv"

# Model output
MODEL_SAVE_PATH = MODELS_DIR / "xgb_model.json"

# Results
FORECAST_RESULTS_PATH = REPORTS_DIR / "seasonal_forecast_results.csv"

# ─────────────────────────────────────────────────────────────
# Feature Engineering Config
# ─────────────────────────────────────────────────────────────
LAGS = list(range(1, 25))  # 24 months of lags
ROLLING_WINDOWS = [3, 6, 9, 12]  # Rolling window sizes

# ─────────────────────────────────────────────────────────────
# Model Training Config
# ─────────────────────────────────────────────────────────────
TEST_MONTHS = 2  # Number of months to hold out for testing
RANDOM_STATE = 42
CV_SPLITS = 3  # TimeSeriesSplit folds

# Hyperparameter grid for XGBoost
PARAM_GRID = {
    'n_estimators': [100, 300, 500],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.03, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# Best parameters (after tuning, can be used directly)
BEST_PARAMS = {
    'colsample_bytree': 1.0,
    'learning_rate': 0.1,
    'max_depth': 5,
    'n_estimators': 100,
    'subsample': 0.8
}