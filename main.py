import os
import pandas as pd
from src.data.data_loader import load_data
from src.features.feature_engineering import create_features
from src.models.model_training import (train_model_random_forest_regressor, train_model_xgboost_regressor)
from src.models.forecasting import forecast_future
from src.utils.auxiliary_comparison_chart import plot_results_comparison
from src.visualization.plots import plot_results

# Data file path
data_path = "data/MonthlySales_TopGoods_500_ByCode_And_Name.csv"
output_dir = "reports/figures"
MAX_LAG = 24

# Load data
print("در حال بارگذاری داده‌ها...")
data = load_data(data_path)
print(f"تعداد رکوردها: {len(data)}")
print(f"تعداد کدهای یکتا: {data['Code'].nunique()}")

# Feature Engineering
print("در حال ایجاد ویژگی‌ها...")
df, le = create_features(data, max_lag=MAX_LAG, rolling_windows=[3, 6, 9, 12])
print(f"تعداد ویژگی‌ها: {len(df.columns)}")

# Model training
feature_cols = [col for col in df.columns if col not in ['MainQty', 'Name', 'Date', 'Code']]
x_train = df[feature_cols]
y_train = df['MainQty']

print(f"\nدر حال آموزش مدل با {len(x_train)} نمونه...")
model = train_model_xgboost_regressor(x_train, y_train)

# Future prediction
future_dates_df = pd.DataFrame({
    'Year': [1404] * 8,
    'Month': [5, 6, 7, 8, 9, 10, 11, 12]
})

print("در حال پیش‌بینی...")
results_df = forecast_future(model, le, data, feature_cols, future_dates_df, max_lag=MAX_LAG)

# Save results
if not results_df.empty:
    df_predictions_pivoted = results_df.pivot_table(
        index=['Code', 'Model'],
        columns='Month',
        values='Predicted_Sales',
        fill_value=0
    )

    os.makedirs("reports", exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    df_predictions_pivoted.to_csv("reports/seasonal_forecast_results.csv")
    print(f"\n✓ نتایج در reports/seasonal_forecast_results.csv ذخیره شد")

# Drawing a diagram
predicted_codes = df_predictions_pivoted.index.get_level_values('Code').unique()
total_codes = len(predicted_codes)
print(f"✓ تعداد کدهای پیش‌بینی شده: {total_codes}")


codes_to_plot = predicted_codes.tolist()

print(f"\nدر حال رسم نمودار برای {len(codes_to_plot)} کد...")

for i, code in enumerate(codes_to_plot, 1):
    try:
        plot_results(data, df_predictions_pivoted, [code], output_dir)
        print(f"  [{i}/{len(codes_to_plot)}] نمودار کد {code} ذخیره شد")
    except Exception as e:
        print(f"  ⚠ خطا در رسم نمودار کد {code}: {e}")

print(f"\n✓ نمودارها در پوشه {output_dir} ذخیره شدند")

print("\n" + "=" * 50)
print("خلاصه پیش‌بینی‌ها:")
print("=" * 50)

monthly_avg = results_df.groupby('Month')['Predicted_Sales'].agg(['mean', 'sum', 'count'])
print("\nمیانگین فروش پیش‌بینی شده برای هر ماه:")
print(monthly_avg.round(2))

top_codes = results_df.groupby('Code')['Predicted_Sales'].sum().nlargest(5)
print("\n5 کد با بیشترین مجموع پیش‌بینی فروش:")
for code, total in top_codes.items():
    product_name = data[data['Code'] == code]['Name'].iloc[0] if not data[data['Code'] == code].empty else "نامشخص"
    print(f"  کد {code}: {total:,.0f} واحد - {product_name[:30]}")

else:
    print("⚠ هیچ پیش‌بینی انجام نشد!")

print("\n" + "=" * 50)
print("✓ پردازش با موفقیت به پایان رسید!")
print("=" * 50)