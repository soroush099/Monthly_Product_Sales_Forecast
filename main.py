import os
import pandas as pd
from src.data.data_loader import load_data
from src.features.feature_engineering import create_features
from src.models.model_training import train_model
from src.models.forecasting import forecast_future
from src.utils.auxiliary_comparison_chart import plot_results_comparison
from src.visualization.plots import plot_results

# Data file path
data_path = "data/MonthlySales_TopGoods_500_ByCode_And_Name.csv"
output_dir = "reports/figures"

# Load data
data = load_data(data_path)

# Feature Engineering
df, le = create_features(data, max_lag=24)

# Model training
features = ['Code_Encoded', 'Year', 'Month',
            'Sales_Lag_1', 'Sales_Lag_2', 'Sales_Lag_3', 'Sales_Lag_12',
            'Rolling_Mean_3', 'Rolling_Std_3']
x_train, y_train = df[features], df['MainQty']
model = train_model(x_train, y_train)

# Future prediction
future_dates_df = pd.DataFrame({'Year': [1404] * 8, 'Month': [5, 6, 7, 8, 9, 10, 11, 12]})
results_df = forecast_future(model, le, data, features, future_dates_df)

# Save results
df_predictions_pivoted = results_df.pivot_table(index=['Code', 'Model'], columns='Month', values='Predicted_Sales')
os.makedirs("reports", exist_ok=True)
df_predictions_pivoted.to_csv("reports/seasonal_forecast_results.csv")

# Drawing a diagram
predicted_codes = df_predictions_pivoted.index.get_level_values('Code').unique()
codes_to_plot = predicted_codes.tolist()
for code in codes_to_plot:
    plot_results(data, df_predictions_pivoted, [code], output_dir)
