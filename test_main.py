import os
import pandas as pd
from src.data.data_loader import load_data
from src.features.feature_engineering import create_features
from src.models.model_training import (
    train_model_random_forest_regressor,
    train_model_xgboost_regressor
)
from src.models.forecasting import forecast_future
from src.utils.auxiliary_comparison_chart import plot_results_comparison  # تغییر ایمپورت

# تنظیمات
data_path = "data/MonthlySales_TopGoods_500_ByCode_And_Name.csv"
external_data_path = "data/sample_new_data.csv"  # فایل داده‌های جدید برای مقایسه
output_dir = "reports/figures"
MAX_LAG = 24

print("=" * 60)
print("شروع پردازش پیش‌بینی فروش")
print("=" * 60)

# بارگذاری داده‌ها
print("\n📊 در حال بارگذاری داده‌ها...")
data = load_data(data_path)
print(f"✓ تعداد رکوردها: {len(data):,}")
print(f"✓ تعداد کدهای یکتا: {data['Code'].nunique()}")
print(
    f"✓ بازه زمانی: سال {data['Year'].min()} ماه {data['Month'].min()} تا سال {data['Year'].max()} ماه {data['Month'].max()}")

# بررسی وجود فایل داده‌های خارجی
if os.path.exists(external_data_path):
    print(f"✓ فایل داده‌های خارجی یافت شد: {external_data_path}")
else:
    print(f"⚠ فایل داده‌های خارجی یافت نشد: {external_data_path}")

# ایجاد ویژگی‌ها
print("\n🔧 در حال ایجاد ویژگی‌ها...")
df, le = create_features(data, max_lag=MAX_LAG, rolling_windows=[3, 6, 9, 12])
print(f"✓ تعداد ویژگی‌های ایجاد شده: {len(df.columns)}")
print(f"✓ تعداد نمونه‌های آموزش: {len(df):,}")

# آموزش مدل
feature_cols = [col for col in df.columns if col not in ['MainQty', 'Name', 'Date', 'Code']]
x_train = df[feature_cols]
y_train = df['MainQty']

print(f"\n🤖 در حال آموزش مدل XGBoost...")
model = train_model_random_forest_regressor(x_train, y_train)
print("✓ مدل با موفقیت آموزش دید!")

# تعریف ماه‌های آینده برای پیش‌بینی
future_dates_df = pd.DataFrame({
    'Year': [1404] * 8,
    'Month': [5, 6, 7, 8, 9, 10, 11, 12]
})

print(f"\n🔮 در حال پیش‌بینی برای {len(future_dates_df)} ماه آینده...")
results_df = forecast_future(model, le, data, feature_cols, future_dates_df, max_lag=MAX_LAG)

# پردازش نتایج
if not results_df.empty:
    # ایجاد جدول محوری
    df_predictions_pivoted = results_df.pivot_table(
        index=['Code', 'Model'],
        columns='Month',
        values='Predicted_Sales',
        fill_value=0
    )

    # ایجاد پوشه‌ها
    os.makedirs("reports", exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # ذخیره نتایج در CSV
    csv_path = "reports/seasonal_forecast_results.csv"
    df_predictions_pivoted.to_csv(csv_path)
    print(f"✓ نتایج پیش‌بینی ذخیره شد: {csv_path}")

    # دریافت لیست کدها
    predicted_codes = df_predictions_pivoted.index.get_level_values('Code').unique()
    total_codes = len(predicted_codes)
    print(f"✓ تعداد کدهای پیش‌بینی شده: {total_codes}")

    # ==== رسم نمودار برای همه کدها ====
    print(f"\n📈 در حال رسم نمودار مقایسه‌ای برای {total_codes} کد...")
    print("این فرآیند ممکن است چند دقیقه زمان ببرد...")

    # تبدیل همه کدها به لیست
    codes_to_plot = predicted_codes.tolist()

    # شمارنده برای نمایش پیشرفت
    successful_plots = 0
    failed_plots = 0

    for i, code in enumerate(codes_to_plot, 1):
        try:
            # رسم نمودار با قابلیت مقایسه با داده‌های خارجی
            plot_results_comparison(
                data=data,
                df_predictions_pivoted=df_predictions_pivoted,
                codes_to_plot=[code],
                output_dir=output_dir,
                external_csv=external_data_path  # اضافه کردن داده‌های خارجی
            )
            successful_plots += 1

            # نمایش پیشرفت هر 10 نمودار
            if i % 10 == 0 or i == total_codes:
                print(f"  پیشرفت: [{i}/{total_codes}] ({(i / total_codes) * 100:.1f}%) - کد فعلی: {code}")

        except Exception as e:
            failed_plots += 1
            print(f"  ⚠ خطا در رسم نمودار کد {code}: {str(e)[:50]}")

    print(f"\n✓ رسم نمودارها تکمیل شد!")
    print(f"  • موفق: {successful_plots} نمودار")
    print(f"  • ناموفق: {failed_plots} نمودار")
    print(f"  • محل ذخیره: {output_dir}/")

    # نمایش آمار تکمیلی
    print("\n" + "=" * 60)
    print("📊 آمار پیش‌بینی‌ها")
    print("=" * 60)

    # میانگین ماهانه
    monthly_stats = results_df.groupby('Month')['Predicted_Sales'].agg([
        ('میانگین', 'mean'),
        ('مجموع', 'sum'),
        ('حداکثر', 'max'),
        ('حداقل', 'min')
    ]).round(0)
    print("\nآمار ماهانه پیش‌بینی‌ها:")
    print(monthly_stats.to_string())

    # 10 کد برتر
    print("\n🏆 10 کد با بیشترین پیش‌بینی فروش (مجموع 8 ماه):")
    top_codes = results_df.groupby('Code')['Predicted_Sales'].sum().nlargest(10)

    for rank, (code, total) in enumerate(top_codes.items(), 1):
        product_info = data[data['Code'] == code]
        if not product_info.empty:
            product_name = product_info['Name'].iloc[0][:40]
            avg_historical = product_info['MainQty'].mean()
            growth = ((total / 8 - avg_historical) / avg_historical * 100) if avg_historical > 0 else 0
            print(f"  {rank:2d}. کد {code}: {total:>10,.0f} واحد | {product_name}")
            print(f"      (میانگین ماهانه: {total / 8:,.0f} | رشد نسبت به گذشته: {growth:+.1f}%)")

    # ذخیره گزارش تفصیلی
    report_path = "reports/detailed_forecast_report.csv"
    detailed_report = results_df.merge(
        data[['Code', 'Name']].drop_duplicates(),
        on='Code',
        how='left'
    )
    detailed_report.to_csv(report_path, index=False, encoding='utf-8-sig')
    print(f"\n✓ گزارش تفصیلی ذخیره شد: {report_path}")

else:
    print("\n⚠ هیچ پیش‌بینی انجام نشد!")

print("\n" + "=" * 60)
print("✅ پردازش با موفقیت به پایان رسید!")
print("=" * 60)

# نمایش فایل‌های خروجی
print("\n📁 فایل‌های خروجی:")
print(f"  • نتایج پیش‌بینی: reports/seasonal_forecast_results.csv")
print(f"  • گزارش تفصیلی: reports/detailed_forecast_report.csv")
print(f"  • نمودارها: {output_dir}/plot_*.png")
if os.path.exists(external_data_path):
    print(f"  • داده‌های خارجی در نمودارها با رنگ نارنجی نمایش داده شده‌اند")