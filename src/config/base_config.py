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
RAW_DATA_PATH = DATA_DIR / "ModelAllData3.csv"
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
LAGS = list(range(1, 24))  # 24 months of lags
ROLLING_WINDOWS = [3, 6, 9, 12]  # Rolling window sizes

# ─────────────────────────────────────────────────────────────
# Model Training Config
# ─────────────────────────────────────────────────────────────
TEST_MONTHS = 2  # Number of months to hold out for testing
RANDOM_STATE = 42
CV_SPLITS = 3  # TimeSeriesSplit folds

# Hyperparameter grid for XGBoost
# PARAM_GRID = {
#     'n_estimators': [100, 300, 500],
#     'max_depth': [3, 5, 7],
#     'learning_rate': [0.03, 0.1, 0.2],
#     'subsample': [0.8, 1.0],
#     'colsample_bytree': [0.8, 1.0]
# }
PARAM_GRID = {
    'n_estimators': [500, 1000],
    'max_depth': [5, 6, 8],
    'learning_rate': [0.01, 0.05],
    'subsample': [0.7, 0.8],
    'colsample_bytree': [0.7, 0.8],
    'min_child_weight': [1, 3] # اضافه شده برای کنترل حساسیت به داده‌های پرت
}

# Best parameters (after tuning, can be used directly)
# BEST_PARAMS = {
#     'colsample_bytree': 1.0,
#     'learning_rate': 0.1,
#     'max_depth': 5,
#     'n_estimators': 100,
#     'subsample': 0.8
# }
BEST_PARAMS = {
    'objective': 'reg:tweedie',
    'tweedie_variance_power': 1.5,
    'eval_metric': 'rmse',
    'n_estimators': 1000,
    'learning_rate': 0.01,
    'max_depth': 6,
    'min_child_weight': 1,
    'colsample_bytree': 0.8,
    'subsample': 0.8,
    'random_state': 42
}

# تنظیمات ثابت (چیزهایی که نمی‌خواهیم تغییر کنند)
STATIC_PARAMS = {
    'objective': 'reg:tweedie',
    'n_jobs': -1,
    'random_state': RANDOM_STATE,
    'booster': 'gbtree'
}
