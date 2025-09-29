import numpy as np
import pandas as pd

from src.features.feature_engineering import create_features_for_prediction


def forecast_future(model, le, data, features, future_dates_df, max_lag=24):
    live_data = data.copy()
    final_predictions = []

    for _, future_row in future_dates_df.iterrows():
        future_year = future_row['Year']
        future_month = future_row['Month']
        current_features = []
        codes_to_predict = le.classes_

        for code in codes_to_predict:
            history = live_data[live_data['Code'] == code].sort_values(['Year', 'Month']).tail(max_lag)
            if history.empty:
                continue

            row_features = create_features_for_prediction(
                history_data=history,
                code=code,
                le=le,
                future_year=future_year,
                future_month=future_month,
                max_lag=max_lag
            )

            current_features.append(row_features)

        if not current_features:
            continue

        features_df = pd.DataFrame(current_features)

        missing_cols = set(features) - set(features_df.columns)
        for col in missing_cols:
            features_df[col] = 0

        predictions = model.predict(features_df[features])

        for i, (_, row) in enumerate(features_df.iterrows()):
            predicted_qty = max(0, round(predictions[i]))

            code_idx = np.where(le.transform(le.classes_) == row['Code_Encoded'])[0][0]
            actual_code = le.classes_[code_idx]

            new_row = {
                'Code': actual_code,
                'Year': future_year,
                'Month': future_month,
                'MainQty': predicted_qty
            }
            live_data = pd.concat([live_data, pd.DataFrame([new_row])], ignore_index=True)

            final_predictions.append({
                'Code': actual_code,
                'Year': future_year,
                'Month': future_month,
                'Predicted_Sales': predicted_qty,
                'Model': type(model).__name__
            })

    return pd.DataFrame(final_predictions)
