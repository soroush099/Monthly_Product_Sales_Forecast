# Monthly Product Sales Forecast

پروژه پیش‌بینی فروش ماهانه محصولات با استفاده از الگوریتم XGBoost

---

## فهرست مطالب

1. [معرفی پروژه](#معرفی-پروژه)
2. [ویژگی‌ها](#ویژگیها)
3. [ساختار پروژه](#ساختار-پروژه)
4. [نصب و راه‌اندازی](#نصب-و-راهاندازی)
5. [نحوه استفاده](#نحوه-استفاده)
6. [توضیح ماژول‌ها](#توضیح-ماژولها)
7. [Feature Engineering](#feature-engineering)
8. [مدل و ارزیابی](#مدل-و-ارزیابی)
9. [داشبورد تعاملی](#داشبورد-تعاملی)
10. [تنظیمات](#تنظیمات)

---

## معرفی پروژه

این پروژه یک سیستم پیش‌بینی فروش ماهانه برای محصولات مختلف است که با استفاده از موارد زیر توسعه یافته:

- XGBoost Regressor برای مدل‌سازی
- Time Series Features شامل Lag و Rolling
- Streamlit Dashboard برای نمایش تعاملی

### جریان کار Pipeline

```
Data Loader --> Feature Engineering --> Model Training --> Forecasting
                                                              |
                                                              v
                            Plots <-- Evaluation <------------+
```

مراحل اصلی:

1. بارگذاری و پاکسازی داده
2. ساخت ویژگی‌های Lag و Rolling
3. تقسیم داده به Train/Test
4. آموزش مدل XGBoost
5. ارزیابی مدل
6. پیش‌بینی ماه آینده
7. تولید نمودارها

---

## ویژگی‌ها

- Pipeline خودکار: اجرای تمام مراحل با یک دستور
- 24 Lag Feature: ویژگی‌های تاخیری تا 24 ماه
- Rolling Features: میانگین‌های متحرک 3، 6، 9 و 12 ماهه
- XGBoost Model: الگوریتم قدرتمند Gradient Boosting
- GridSearchCV: بهینه‌سازی خودکار Hyperparameters
- ارزیابی کامل: MAE, RMSE, R²
- داشبورد Streamlit: رابط کاربری تعاملی
- Interactive Plots: جابجایی بین محصولات
- ذخیره مدل: امکان بارگذاری مجدد مدل

---

## ساختار پروژه

```
monthly-product-sales-forecast/
|
+-- data/
|   +-- ModelAllData2.csv
|
+-- models/
|   +-- xgb_model.json
|
+-- reports/
|   +-- figures/
|   |   +-- feature_importance.png
|   |   +-- actual_vs_predicted.png
|   |   +-- residuals.png
|   |   +-- forecast_goods_*.png
|   +-- seasonal_forecast_results.csv
|
+-- src/
|   +-- __init__.py
|   |
|   +-- config/
|   |   +-- __init__.py
|   |   +-- base_config.py
|   |
|   +-- data/
|   |   +-- __init__.py
|   |   +-- data_loader.py
|   |
|   +-- features/
|   |   +-- __init__.py
|   |   +-- feature_engineering.py
|   |
|   +-- models/
|   |   +-- __init__.py
|   |   +-- model_training.py
|   |   +-- evaluation.py
|   |   +-- forecasting.py
|   |
|   +-- utils/
|   |   +-- __init__.py
|   |   +-- helpers.py
|   |   +-- jalali_utils.py
|   |   +-- auxiliary_comparison_chart.py
|   |
|   +-- visualization/
|       +-- __init__.py
|       +-- plots.py
|       +-- dashboard.py
|       +-- interactive_plots.py
|
+-- tests/
|   +-- test_pipeline.py
|
+-- notebooks/
|   +-- exploration.ipynb
|
+-- main.py
+-- app.py
+-- view_forecasts.py
+-- requirements.txt
+-- README.md
```

### توضیح پوشه‌ها

**data/** - فایل‌های داده خام و پردازش شده

**models/** - مدل‌های آموزش دیده XGBoost

**reports/** - خروجی‌ها شامل نمودارها و نتایج پیش‌بینی

**reports/figures/** - تصاویر نمودارها

**src/** - کد منبع اصلی پروژه

**src/config/** - تنظیمات و ثابت‌ها

**src/data/** - توابع بارگذاری و پاکسازی داده

**src/features/** - توابع Feature Engineering

**src/models/** - توابع آموزش، ارزیابی و پیش‌بینی

**src/utils/** - توابع کمکی عمومی

**src/visualization/** - توابع رسم نمودار و داشبورد

**tests/** - تست‌های خودکار

**notebooks/** - نوت‌بوک‌های Jupyter

### فایل‌های اصلی

**main.py** - نقطه ورود اصلی و اجرای کامل Pipeline

**app.py** - داشبورد تعاملی Streamlit

**view_forecasts.py** - نمایشگر تعاملی با Matplotlib

**requirements.txt** - لیست وابستگی‌های پروژه

---

## نصب و راه‌اندازی

### پیش‌نیازها

- Python 3.9 یا بالاتر
- pip

### مراحل نصب

کلون کردن پروژه:

```bash
git clone https://github.com/your-username/monthly-product-sales-forecast.git
cd monthly-product-sales-forecast
```

ساخت محیط مجازی:

```bash
python -m venv .venv
```

فعال‌سازی محیط مجازی در Windows:

```bash
.venv\Scripts\activate
```

فعال‌سازی محیط مجازی در Linux/Mac:

```bash
source .venv/bin/activate
```

نصب وابستگی‌ها:

```bash
pip install -r requirements.txt
```

قرار دادن فایل داده در پوشه data:

```
data/ModelAllData2.csv
```

### فایل requirements.txt

```
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
xgboost>=2.0.0
streamlit>=1.28.0
pytest>=7.0.0
```

---

## نحوه استفاده

### روش 1: اجرای Pipeline کامل

```bash
python main.py
```

خروجی نمونه:

```
============================================================
 1. LOADING AND CLEANING DATA
============================================================
Loaded 1,380,742 rows from data/ModelAllData2.csv
Cleaned data: 1,380,742 rows remaining

============================================================
 2. FEATURE ENGINEERING
============================================================
Aggregated to 328,770 monthly records
Filled missing months: 988,764 total records
Created 24 lag features
Created 4 rolling features
Feature engineering complete: 965,222 rows, 32 features
Unique GoodsIDs: 23,542

============================================================
 3. TRAIN/TEST SPLIT
============================================================
Train/Test Split:
   Split date: 2025-09-01
   Train size: 918,123
   Test size: 47,099

============================================================
 4. MODEL TRAINING
============================================================
Training model with best parameters...
Model training complete
Model saved to models/xgb_model.json

============================================================
 5. MODEL EVALUATION
============================================================
Model Evaluation Results:
   MAE:  3.82
   RMSE: 10.62
   R2:   0.790

============================================================
 6. FORECASTING
============================================================
Forecast generated for 23,542 products
Forecast month: 2025-12
Forecast results saved to reports/seasonal_forecast_results.csv

============================================================
 7. VISUALIZATION
============================================================
GoodId = 61590
Figure saved to reports/figures/feature_importance.png
Figure saved to reports/figures/actual_vs_predicted.png
Figure saved to reports/figures/residuals.png
Figure saved to reports/figures/forecast_goods_61590.png

============================================================
 PIPELINE COMPLETE
============================================================

Summary:
--------
Products: 23,542
Date Range: 2022-01 to 2025-11
Model Performance (Test Set):
  - MAE:  3.82
  - RMSE: 10.62
  - R2:   0.790
Figures saved to: reports/figures
Forecast saved to: reports/seasonal_forecast_results.csv
Model saved to: models/xgb_model.json
```

### روش 2: اجرای داشبورد تعاملی Streamlit

```bash
streamlit run app.py
```

مرورگر در آدرس http://localhost:8501 باز می‌شود.

### روش 3: نمایشگر تعاملی Matplotlib

```bash
python view_forecasts.py
```

---

## توضیح ماژول‌ها

### ماژول src/config/base_config.py

تنظیمات و ثابت‌های پروژه:

```python
# مسیرها
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RAW_DATA_PATH = DATA_DIR / "ModelAllData2.csv"
MODEL_SAVE_PATH = MODELS_DIR / "xgb_model.json"

# تنظیمات Feature
LAGS = list(range(1, 25))          # 24 lag
ROLLING_WINDOWS = [3, 6, 9, 12]    # 4 rolling

# تنظیمات مدل
TEST_MONTHS = 2
RANDOM_STATE = 42
BEST_PARAMS = {
    'n_estimators': 100,
    'max_depth': 5,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 1.0
}
```

### ماژول src/data/data_loader.py

توابع بارگذاری و پاکسازی داده:

```python
def load_raw_data(filepath=None):
    """بارگذاری داده از CSV."""
    
def clean_data(df):
    """پاکسازی داده شامل تبدیل تاریخ و حذف null."""
    
def load_and_clean_data(filepath=None):
    """بارگذاری و پاکسازی در یک مرحله."""
```

مثال استفاده:

```python
from src.data.data_loader import load_and_clean_data

df = load_and_clean_data("data/ModelAllData2.csv")
print(f"Rows: {len(df)}")
```

### ماژول src/features/feature_engineering.py

توابع ساخت ویژگی‌ها:

```python
def add_date_features(df):
    """اضافه کردن year, month, quarter."""

def aggregate_monthly(df):
    """تجمیع به سطح ماهانه."""

def fill_missing_months(monthly):
    """پر کردن ماه‌های خالی با صفر."""

def create_lag_features(df, lags=None):
    """ساخت ویژگی‌های lag."""

def create_rolling_features(df, windows=None):
    """ساخت ویژگی‌های rolling."""

def build_features(df):
    """اجرای کامل pipeline ویژگی‌سازی."""
```

مثال استفاده:

```python
from src.features.feature_engineering import build_features

monthly, feature_cols = build_features(df)
print(f"Features: {len(feature_cols)}")
# Features: 32
```

### ماژول src/models/model_training.py

توابع آموزش مدل:

```python
def train_test_split_temporal(df, feature_cols, test_months=2):
    """تقسیم زمانی train/test."""

def train_with_grid_search(X_train, y_train, param_grid=None):
    """آموزش با GridSearchCV."""

def train_with_best_params(X_train, y_train, params=None):
    """آموزش با پارامترهای از پیش تعیین شده."""

def save_model(model, filepath=None):
    """ذخیره مدل."""

def load_model(filepath=None):
    """بارگذاری مدل."""
```

مثال استفاده:

```python
from src.models.model_training import (
    train_test_split_temporal,
    train_with_best_params,
    save_model
)

X_train, X_test, y_train, y_test, split_date = train_test_split_temporal(
    monthly, feature_cols
)

model = train_with_best_params(X_train, y_train)
save_model(model)
```

### ماژول src/models/evaluation.py

توابع ارزیابی مدل:

```python
def evaluate_model(model, X_test, y_test, verbose=True):
    """ارزیابی مدل و برگرداندن metrics."""

def get_predictions(model, X):
    """دریافت پیش‌بینی‌ها."""

def evaluate_by_product(model, df, feature_cols):
    """ارزیابی به تفکیک محصول."""
```

مثال استفاده:

```python
from src.models.evaluation import evaluate_model

metrics = evaluate_model(model, X_test, y_test)
print(f"R2: {metrics['R2']:.3f}")
```

### ماژول src/models/forecasting.py

توابع پیش‌بینی:

```python
def forecast_next_month(model, monthly, feature_cols):
    """پیش‌بینی ماه آینده برای همه محصولات."""

def get_historical_with_predictions(model, monthly, feature_cols, goods_id):
    """دریافت تاریخچه + پیش‌بینی برای یک محصول."""

def save_forecast_results(forecast_df, filepath=None):
    """ذخیره نتایج پیش‌بینی."""
```

مثال استفاده:

```python
from src.models.forecasting import forecast_next_month

forecast_df = forecast_next_month(model, monthly, feature_cols)
print(forecast_df.head())
```

خروجی:

```
   GoodsID  forecast_month  predicted_next_month
0    12345      2025-12-01                  45.2
1    67890      2025-12-01                 123.7
```

### ماژول src/visualization/plots.py

توابع رسم نمودار:

```python
def plot_historical_and_predictions(
    model, monthly, feature_cols, goods_id,
    forecast_df=None, split_date=None,
    save_path=None, show=False
):
    """نمودار فروش + پیش‌بینی برای یک محصول."""

def plot_feature_importance(model, feature_cols, top_n=20, save_path=None):
    """نمودار اهمیت ویژگی‌ها."""

def plot_actual_vs_predicted(y_true, y_pred, save_path=None):
    """نمودار مقایسه واقعی و پیش‌بینی."""

def plot_residuals(y_true, y_pred, save_path=None):
    """نمودار توزیع خطاها."""
```

### ماژول src/utils/helpers.py

توابع کمکی:

```python
def set_seed(seed=42):
    """تنظیم random seed."""

def ensure_dir(path):
    """اطمینان از وجود دایرکتوری."""

def sample_goods_id(df, min_lag=8):
    """انتخاب تصادفی یک محصول معتبر."""

def print_section(title, char="=", width=60):
    """چاپ هدر بخش."""
```

---

## Feature Engineering

### ویژگی‌های ساخته شده

**Lag Features (24 عدد)**

- lag_1 تا lag_24
- فروش ماه‌های قبل

**Rolling Features (4 عدد)**

- rolling_3: میانگین 3 ماه گذشته
- rolling_6: میانگین 6 ماه گذشته
- rolling_9: میانگین 9 ماه گذشته
- rolling_12: میانگین 12 ماه گذشته

**Date Features (3 عدد)**

- year: سال
- month: ماه
- quarter: فصل

**Other Features (1 عدد)**

- Price: میانگین قیمت ماهانه

**مجموع: 32 ویژگی**

### فرآیند Feature Engineering

```
1. Raw Data (Daily)
       |
       v
2. Add Date Features (year, month, quarter)
       |
       v
3. Aggregate Monthly (sum MainQty, mean Price)
       |
       v
4. Fill Missing Months (با صفر)
       |
       v
5. Create Lag Features (lag_1 to lag_24)
       |
       v
6. Create Rolling Features (rolling_3, 6, 9, 12)
       |
       v
7. Monthly Data با 32 feature
```

---

## مدل و ارزیابی

### الگوریتم

- XGBoost Regressor
- Objective: reg:squarederror
- Cross-Validation: TimeSeriesSplit با 3 fold

### Hyperparameters

```
n_estimators: 100
max_depth: 5
learning_rate: 0.1
subsample: 0.8
colsample_bytree: 1.0
```

### معیارهای ارزیابی

**MAE (Mean Absolute Error): 3.82**

میانگین خطای مطلق

**RMSE (Root Mean Squared Error): 10.62**

ریشه میانگین مربعات خطا

**R2 (R-squared): 0.790**

ضریب تعیین - 79% واریانس توضیح داده شده

### Grid Search Parameters

```python
PARAM_GRID = {
    'n_estimators': [100, 300, 500],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.03, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}
```

مجموع: 108 ترکیب × 3 fold = 324 fit

---

## داشبورد تعاملی

### اجرا

```bash
streamlit run app.py
```

### امکانات داشبورد

**جستجوی محصول**

وارد کردن شماره GoodsID

**انتخاب از لیست**

Dropdown برای انتخاب محصول

**انتخاب تصادفی**

دکمه Random برای انتخاب تصادفی

**جابجایی**

دکمه‌های Previous و Next

**نمایش Metrics**

MAE, RMSE, R2 برای هر محصول

**پیش‌بینی**

نمایش پیش‌بینی ماه آینده

**جدول داده**

مشاهده داده‌های خام

**دانلود**

دانلود نمودار به صورت PNG

### نمای داشبورد

```
+----------------------------------------------------------+
|  Sales Forecast Dashboard                                 |
+-------------+--------------------------------------------+
|             |                                            |
| Selection   |      +----------------------------+        |
|             |      |                            |        |
| o Dropdown  |      |    [Forecast Chart]        |        |
| o Search    |      |                            |        |
| o Random    |      +----------------------------+        |
|             |                                            |
| [<] [>]     +--------------------------------------------+
|             |                                            |
| Products:   |  MAE: 3.82  |  RMSE: 10.62  |  R2: 0.790  |
|   23,542    |                                            |
|             +--------------------------------------------+
|             |                                            |
|             |  Forecast: 2025-12 -> 45.2 units           |
|             |                                            |
+-------------+--------------------------------------------+
```

---

## تنظیمات

### تغییر مسیر داده

در فایل src/config/base_config.py:

```python
RAW_DATA_PATH = DATA_DIR / "your_data.csv"
```

### تغییر تعداد Lags

در فایل src/config/base_config.py:

```python
LAGS = list(range(1, 13))  # فقط 12 ماه
```

### تغییر تعداد ماه‌های تست

در فایل src/config/base_config.py:

```python
TEST_MONTHS = 3  # 3 ماه آخر برای تست
```

### اجرای Grid Search

در فایل main.py:

```python
pipeline = main(
    run_grid_search=True,
    save_plots=True
)
```

---

## تست

### اجرای تست‌ها

```bash
pytest tests/ -v
```

### تست دستی

```python
# تست بارگذاری داده
from src.data.data_loader import load_and_clean_data
df = load_and_clean_data()
assert len(df) > 0

# تست ساخت ویژگی
from src.features.feature_engineering import build_features
monthly, features = build_features(df)
assert len(features) == 32

# تست مدل
from src.models.model_training import load_model
model = load_model()
assert model is not None
```

---

## نمونه خروجی‌ها

### فایل پیش‌بینی

فایل: reports/seasonal_forecast_results.csv

```
GoodsID,forecast_month,predicted_next_month
12345,2025-12-01,45.23
67890,2025-12-01,123.67
11111,2025-12-01,8.45
```

### نمودارها

- reports/figures/feature_importance.png
- reports/figures/actual_vs_predicted.png
- reports/figures/residuals.png
- reports/figures/forecast_goods_*.png

---

## استفاده جداگانه از ماژول‌ها

### فقط بارگذاری داده

```python
from src.data.data_loader import load_and_clean_data

df = load_and_clean_data("path/to/data.csv")
```

### فقط پیش‌بینی با مدل موجود

```python
from src.models.model_training import load_model
from src.models.forecasting import forecast_next_month

model = load_model()
forecast = forecast_next_month(model, monthly, feature_cols)
```

### فقط رسم نمودار

```python
from src.visualization.plots import plot_historical_and_predictions

plot_historical_and_predictions(
    model, monthly, feature_cols, 
    goods_id=12345,
    save_path="my_plot.png"
)
```

---

## لایسنس

Nothing

---

## نویسنده

GitHub: @soroush099

---

## منابع

- XGBoost Documentation: https://xgboost.readthedocs.io/
- Streamlit Documentation: https://docs.streamlit.io/
- Scikit-learn Documentation: https://scikit-learn.org/
```

---
