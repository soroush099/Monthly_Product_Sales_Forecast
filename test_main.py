import os
import pandas as pd
from src.data.data_loader import load_data
from src.features.feature_engineering import create_features
from src.models.model_training import (train_model_random_forest_regressor)
from src.models.forecasting import forecast_future
from src.utils.auxiliary_comparison_chart import plot_results_comparison

data_path = "data/MonthlySales_TopGoods_500_ByCode_And_Name.csv"
external_data_path = "data/sample_new_data.csv"
output_dir = "reports/figures"
MAX_LAG = 24

print("=" * 60)
print("Starting Sales Forecast Processing")
print("=" * 60)

print("\n📊 Loading data...")
data = load_data(data_path)
print(f"✓ Number of records: {len(data):,}")
print(f"✓ Number of unique codes: {data['Code'].nunique()}")
print(
    f"✓ Time range: Year {data['Year'].min()} "
    f"Month {data['Month'].min()} to Year {data['Year'].max()} Month {data['Month'].max()}")

if os.path.exists(external_data_path):
    print(f"✓ External data file found: {external_data_path}")
else:
    print(f"⚠ External data file not found: {external_data_path}")

print("\n🔧 Creating features...")
df, le = create_features(data, max_lag=MAX_LAG, rolling_windows=[3, 6, 9, 12])
print(f"✓ Number of features created: {len(df.columns)}")
print(f"✓ Number of training samples: {len(df):,}")

feature_cols = [col for col in df.columns if col not in ['MainQty', 'Name', 'Date', 'Code']]
x_train = df[feature_cols]
y_train = df['MainQty']

print(f"\n🤖 Training XGBoost model...")
model = train_model_random_forest_regressor(x_train, y_train)
print("✓ Model successfully trained!")

future_dates_df = pd.DataFrame({
    'Year': [1404] * 8,
    'Month': [5, 6, 7, 8, 9, 10, 11, 12]
})

print(f"\n🔮 Forecasting for {len(future_dates_df)} future months...")
results_df = forecast_future(model, le, data, feature_cols, future_dates_df, max_lag=MAX_LAG)

if not results_df.empty:
    df_predictions_pivoted = results_df.pivot_table(
        index=['Code', 'Model'],
        columns='Month',
        values='Predicted_Sales',
        fill_value=0
    )

    os.makedirs("reports", exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    csv_path = "reports/seasonal_forecast_results.csv"
    df_predictions_pivoted.to_csv(csv_path)
    print(f"✓ Forecast results saved: {csv_path}")

    predicted_codes = df_predictions_pivoted.index.get_level_values('Code').unique()
    total_codes = len(predicted_codes)
    print(f"✓ Number of predicted codes: {total_codes}")

    print(f"\n📈 Creating comparison charts for {total_codes} codes...")
    print("This process may take a few minutes...")

    codes_to_plot = predicted_codes.tolist()

    successful_plots = 0
    failed_plots = 0

    for i, code in enumerate(codes_to_plot, 1):
        try:
            plot_results_comparison(
                data=data,
                df_predictions_pivoted=df_predictions_pivoted,
                codes_to_plot=[code],
                output_dir=output_dir,
                external_csv=external_data_path
            )
            successful_plots += 1

            if i % 10 == 0 or i == total_codes:
                print(f"  Progress: [{i}/{total_codes}] ({(i / total_codes) * 100:.1f}%) - Current code: {code}")

        except Exception as e:
            failed_plots += 1
            print(f"  ⚠ Error plotting code {code}: {str(e)[:50]}")

    print(f"\n✓ Chart creation completed!")
    print(f"  • Successful: {successful_plots} charts")
    print(f"  • Failed: {failed_plots} charts")
    print(f"  • Save location: {output_dir}/")

    print("\n" + "=" * 60)
    print("📊 Forecast Statistics")
    print("=" * 60)

    monthly_stats = results_df.groupby('Month')['Predicted_Sales'].agg([
        ('Average', 'mean'),
        ('Total', 'sum'),
        ('Maximum', 'max'),
        ('Minimum', 'min')
    ]).round(0)
    print("\nMonthly forecast statistics:")
    print(monthly_stats.to_string())

    print("\n🏆 Top 10 codes with highest sales forecast (8 months total):")
    top_codes = results_df.groupby('Code')['Predicted_Sales'].sum().nlargest(10)

    for rank, (code, total) in enumerate(top_codes.items(), 1):
        product_info = data[data['Code'] == code]
        if not product_info.empty:
            product_name = product_info['Name'].iloc[0][:40]
            avg_historical = product_info['MainQty'].mean()
            growth = ((total / 8 - avg_historical) / avg_historical * 100) if avg_historical > 0 else 0
            print(f"  {rank:2d}. Code {code}: {total:>10,.0f} units | {product_name}")
            print(f"      (Monthly average: {total / 8:,.0f} | Growth vs history: {growth:+.1f}%)")

    report_path = "reports/detailed_forecast_report.csv"
    detailed_report = results_df.merge(
        data[['Code', 'Name']].drop_duplicates(),
        on='Code',
        how='left'
    )
    detailed_report.to_csv(report_path, index=False, encoding='utf-8-sig')
    print(f"\n✓ Detailed report saved: {report_path}")

else:
    print("\n⚠ No forecasts were generated!")

print("\n" + "=" * 60)
print("✅ Processing completed successfully!")
print("=" * 60)

print("\n📁 Output files:")
print(f"  • Forecast results: reports/seasonal_forecast_results.csv")
print(f"  • Detailed report: reports/detailed_forecast_report.csv")
print(f"  • Charts: {output_dir}/plot_*.png")
if os.path.exists(external_data_path):
    print(f"  • External data displayed in charts with orange color")
