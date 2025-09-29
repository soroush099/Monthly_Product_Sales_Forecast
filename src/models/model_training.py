from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


def train_model_random_forest_regressor(x, y):
    rf_model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=5
    )
    rf_model.fit(x, y)
    return rf_model


def train_model_xgboost_regressor(x, y):
    xgb_model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    xgb_model.fit(x, y)
    return xgb_model

