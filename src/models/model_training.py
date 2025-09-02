from sklearn.ensemble import RandomForestRegressor


def train_model(x, y):
    rf_model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=5
    )
    rf_model.fit(x, y)
    return rf_model
