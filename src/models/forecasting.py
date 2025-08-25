import numpy as np
import pandas as pd


def forecast_future(model, le, data, features, future_dates_df):
    live_data = data.copy()
    final_predictions = []

    for _, future_row in future_dates_df.iterrows():
        future_year = future_row['Year']
        future_month = future_row['Month']
        current_features = []
        codes_to_predict = le.classes_

        for code in codes_to_predict:
            history = live_data[live_data['Code'] == code].tail(12)
            if history.empty:
                continue

            lag_1 = history['MainQty'].iloc[-1]
            lag_2 = history['MainQty'].iloc[-2] if len(history) >= 2 else 0
            lag_3 = history['MainQty'].iloc[-3] if len(history) >= 3 else 0
            lag_12 = history['MainQty'].iloc[0] if len(history) >= 12 else 0
            rolling_mean = history['MainQty'].tail(3).mean()
            rolling_std = history['MainQty'].tail(3).std()

            current_features.append({
                'Code_Encoded': le.transform([code])[0],
                'Code': code,
                'Year': future_year,
                'Month': future_month,
                'Sales_Lag_1': lag_1,
                'Sales_Lag_2': lag_2,
                'Sales_Lag_3': lag_3,
                'Sales_Lag_12': lag_12,
                'Rolling_Mean_3': rolling_mean,
                'Rolling_Std_3': rolling_std if not np.isnan(rolling_std) else 0
            })

        if not current_features:
            continue

        features_df = pd.DataFrame(current_features)
        predictions = model.predict(features_df[features])

        for i, row in features_df.iterrows():
            predicted_qty = max(0, round(predictions[i]))

            new_row = {
                'Code': row['Code'], 'Year': row['Year'], 'Month': row['Month'], 'MainQty': predicted_qty
            }
            live_data = pd.concat([live_data, pd.DataFrame([new_row])], ignore_index=True)

            final_predictions.append({
                'Code': row['Code'], 'Year': row['Year'], 'Month': row['Month'],
                'Predicted_Sales': predicted_qty, 'Model': 'Seasonal_RF'
            })

    return pd.DataFrame(final_predictions)
