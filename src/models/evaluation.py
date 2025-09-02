from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_forecast(forecast_df, test_data):
    merged = forecast_df.merge(
        test_data,
        on=['Code', 'Year', 'Month'],
        how='inner',
        suffixes=('_pred', '_true')
    )

    if merged.empty:
        return None

    y_true = merged['MainQty']
    y_pred = merged['Predicted_Sales']

    return {
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': mean_squared_error(y_true, y_pred, squared=False),
        'R2': r2_score(y_true, y_pred)
    }
