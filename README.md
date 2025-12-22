# 📊 Monthly Product Sales Forecast

پروژه پیش‌بینی فروش ماهانه محصولات با استفاده از الگوریتم XGBoost

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 📋 فهرست مطالب

- [معرفی پروژه](#-معرفی-پروژه)
- [ویژگی‌ها](#-ویژگیها)
- [ساختار پروژه](#-ساختار-پروژه)
- [نصب و راه‌اندازی](#-نصب-و-راهاندازی)
- [نحوه استفاده](#-نحوه-استفاده)
- [توضیح ماژول‌ها](#-توضیح-ماژولها)
- [Feature Engineering](#-feature-engineering)
- [مدل و ارزیابی](#-مدل-و-ارزیابی)
- [داشبورد تعاملی](#-داشبورد-تعاملی)
- [تنظیمات](#-تنظیمات)

---

## 🎯 معرفی پروژه

این پروژه یک سیستم پیش‌بینی فروش ماهانه برای محصولات مختلف است که با استفاده از:

- **XGBoost Regressor** برای مدل‌سازی
- **Time Series Features** شامل Lag و Rolling
- **Streamlit Dashboard** برای نمایش تعاملی توسعه یافته است.


## ✨ ویژگی‌ها

| ویژگی | توضیح |
|-------|-------|
| 🔄 **Pipeline خودکار** | اجرای تمام مراحل با یک دستور |
| 📊 **24 Lag Feature** | ویژگی‌های تاخیری تا 24 ماه |
| 📈 **Rolling Features** | میانگین‌های متحرک 3، 6، 9 و 12 ماهه |
| 🎯 **XGBoost Model** | الگوریتم قدرتمند Gradient Boosting |
| 🔍 **GridSearchCV** | بهینه‌سازی خودکار Hyperparameters |
| 📉 **ارزیابی کامل** | MAE, RMSE, R² |
| 🖥️ **داشبورد Streamlit** | رابط کاربری تعاملی |
| 🔘 **Interactive Plots** | جابجایی بین محصولات |
| 💾 **ذخیره مدل** | امکان بارگذاری مجدد مدل |

---

## 📁 ساختار پروژه

```
monthly-product-sales-forecast/
│
├── data/
│   └── ModelAllData2.csv
│
├── models/
│   └── xgb_model.json
│
├── reports/
│   ├── figures/
│   │   ├── feature_importance.png
│   │   ├── actual_vs_predicted.png
│   │   ├── residuals.png
│   │   └── forecast_goods_*.png
│   └── seasonal_forecast_results.csv
│
├── src/
│   ├── __init__.py
│   │
│   ├── config/
│   │   ├── __init__.py
│   │   └── base_config.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   └── data_loader.py
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   └── feature_engineering.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── model_training.py
│   │   ├── evaluation.py
│   │   └── forecasting.py
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── helpers.py
│   │   ├── jalali_utils.py
│   │   └── auxiliary_comparison_chart.py
│   │
│   └── visualization/
│       ├── __init__.py
│       ├── plots.py
│       ├── dashboard.py
│       └── interactive_plots.py
│
├── tests/
│   └── test_pipeline.py
│
├── notebooks/
│   └── exploration.ipynb
│
├── main.py
├── app.py
├── view_forecasts.py
├── requirements.txt
└── README.md
```

### توضیح پوشه‌ها

| پوشه | توضیح |
|------|-------|
| `data/` | فایل‌های داده خام و پردازش شده |
| `models/` | مدل‌های آموزش دیده (XGBoost) |
| `reports/` | خروجی‌ها شامل نمودارها و نتایج پیش‌بینی |
| `reports/figures/` | تصاویر نمودارها |
| `src/` | کد منبع اصلی پروژه |
| `src/config/` | تنظیمات و ثابت‌ها |
| `src/data/` | توابع بارگذاری و پاکسازی داده |
| `src/features/` | توابع Feature Engineering |
| `src/models/` | توابع آموزش، ارزیابی و پیش‌بینی |
| `src/utils/` | توابع کمکی عمومی |
| `src/visualization/` | توابع رسم نمودار و داشبورد |
| `tests/` | تست‌های خودکار |
| `notebooks/` | نوت‌بوک‌های Jupyter |

### فایل‌های اصلی

| فایل | توضیح |
|------|-------|
| `main.py` | نقطه ورود اصلی - اجرای کامل Pipeline |
| `app.py` | داشبورد تعاملی Streamlit |
| `view_forecasts.py` | نمایشگر تعاملی با Matplotlib |
| `requirements.txt` | لیست وابستگی‌های پروژه |


---

## 🚀 نصب و راه‌اندازی

### پیش‌نیازها

- Python 3.9+
- pip

### مراحل نصب

```bash
# 1. کلون کردن پروژه
git clone https://github.com/your-username/monthly-product-sales-forecast.git
cd monthly-product-sales-forecast

# 2. ساخت محیط مجازی
python -m venv .venv

# 3. فعال‌سازی محیط مجازی
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# 4. نصب وابستگی‌ها
pip install -r requirements.txt

# 5. قرار دادن فایل داده
# data/ModelAllData2.csv
requirements.txt
txt

pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
xgboost>=2.0.0
streamlit>=1.28.0
pytest>=7.0.0
💻 نحوه استفاده
1️⃣ اجرای Pipeline کامل
Bash

python main.py
خروجی نمونه:

text

============================================================
 1. LOADING AND CLEANING DATA
============================================================
✅ Loaded 1,380,742 rows from data/ModelAllData2.csv
✅ Cleaned data: 1,380,742 rows remaining

============================================================
 2. FEATURE ENGINEERING
============================================================
✅ Aggregated to 328,770 monthly records
✅ Filled missing months: 988,764 total records
✅ Created 24 lag features
✅ Created 4 rolling features
✅ Feature engineering complete: 965,222 rows, 32 features
   Unique GoodsIDs: 23,542

============================================================
 3. TRAIN/TEST SPLIT
============================================================
✅ Train/Test Split:
   Split date: 2025-09-01
   Train size: 918,123
   Test size: 47,099

============================================================
 4. MODEL TRAINING
============================================================
🔄 Training model with best parameters...
✅ Model training complete
✅ Model saved to models/xgb_model.json

============================================================
 5. MODEL EVALUATION
============================================================
📊 Model Evaluation Results:
   MAE:  3.82
   RMSE: 10.62
   R²:   0.790

============================================================
 6. FORECASTING
============================================================
✅ Forecast generated for 23,542 products
   Forecast month: 2025-12
✅ Forecast results saved to reports/seasonal_forecast_results.csv

============================================================
 7. VISUALIZATION
============================================================
GoodId = 61590
✅ Figure saved to reports/figures/feature_importance.png
✅ Figure saved to reports/figures/actual_vs_predicted.png
✅ Figure saved to reports/figures/residuals.png
✅ Figure saved to reports/figures/forecast_goods_61590.png

============================================================
 PIPELINE COMPLETE ✓
============================================================

    Summary:
    --------
    • Products: 23,542
    • Date Range: 2022-01 to 2025-11
    • Model Performance (Test Set):
      - MAE:  3.82
      - RMSE: 10.62
      - R²:   0.790
    • Figures saved to: reports/figures
    • Forecast saved to: reports/seasonal_forecast_results.csv
    • Model saved to: models/xgb_model.json
2️⃣ اجرای داشبورد تعاملی Streamlit
Bash

streamlit run app.py
مرورگر در آدرس http://localhost:8501 باز می‌شود.

3️⃣ نمایشگر تعاملی Matplotlib
Bash

python view_forecasts.py
📦 توضیح ماژول‌ها
src/config/base_config.py
تنظیمات و ثابت‌های پروژه:

Python

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
src/data/data_loader.py
توابع بارگذاری و پاکسازی داده:

Python

def load_raw_data(filepath=None) -> pd.DataFrame:
    """بارگذاری داده از CSV."""
    
def clean_data(df) -> pd.DataFrame:
    """پاکسازی داده:
    - تبدیل تاریخ
    - حذف null
    - تبدیل نوع داده
    """
    
def load_and_clean_data(filepath=None) -> pd.DataFrame:
    """بارگذاری و پاکسازی در یک مرحله."""
مثال استفاده:

Python

from src.data.data_loader import load_and_clean_data

df = load_and_clean_data("data/ModelAllData2.csv")
print(f"Rows: {len(df)}")
src/features/feature_engineering.py
توابع ساخت ویژگی‌ها:

Python

def add_date_features(df) -> pd.DataFrame:
    """اضافه کردن year, month, quarter."""

def aggregate_monthly(df) -> pd.DataFrame:
    """تجمیع به سطح ماهانه."""

def fill_missing_months(monthly) -> pd.DataFrame:
    """پر کردن ماه‌های خالی با صفر."""

def create_lag_features(df, lags=None) -> Tuple[pd.DataFrame, List[str]]:
    """ساخت ویژگی‌های lag."""

def create_rolling_features(df, windows=None) -> Tuple[pd.DataFrame, List[str]]:
    """ساخت ویژگی‌های rolling."""

def build_features(df) -> Tuple[pd.DataFrame, List[str]]:
    """اجرای کامل pipeline ویژگی‌سازی."""
مثال استفاده:

Python

from src.features.feature_engineering import build_features

monthly, feature_cols = build_features(df)
print(f"Features: {len(feature_cols)}")
# Features: 32
src/models/model_training.py
توابع آموزش مدل:

Python

def train_test_split_temporal(df, feature_cols, test_months=2):
    """تقسیم زمانی train/test."""
    return X_train, X_test, y_train, y_test, split_date

def train_with_grid_search(X_train, y_train, param_grid=None):
    """آموزش با GridSearchCV."""
    return best_model, best_params

def train_with_best_params(X_train, y_train, params=None):
    """آموزش با پارامترهای از پیش تعیین شده."""
    return model

def save_model(model, filepath=None):
    """ذخیره مدل."""

def load_model(filepath=None):
    """بارگذاری مدل."""
    return model
مثال استفاده:

Python

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
src/models/evaluation.py
توابع ارزیابی مدل:

Python

def evaluate_model(model, X_test, y_test, verbose=True) -> dict:
    """ارزیابی مدل و برگرداندن metrics."""
    return {'MAE': ..., 'RMSE': ..., 'R2': ...}

def get_predictions(model, X) -> np.ndarray:
    """دریافت پیش‌بینی‌ها."""

def evaluate_by_product(model, df, feature_cols) -> pd.DataFrame:
    """ارزیابی به تفکیک محصول."""
مثال استفاده:

Python

from src.models.evaluation import evaluate_model

metrics = evaluate_model(model, X_test, y_test)
print(f"R²: {metrics['R2']:.3f}")
src/models/forecasting.py
توابع پیش‌بینی:

Python

def forecast_next_month(model, monthly, feature_cols) -> pd.DataFrame:
    """پیش‌بینی ماه آینده برای همه محصولات."""

def get_historical_with_predictions(model, monthly, feature_cols, goods_id):
    """دریافت تاریخچه + پیش‌بینی برای یک محصول."""

def save_forecast_results(forecast_df, filepath=None):
    """ذخیره نتایج پیش‌بینی."""
مثال استفاده:

Python

from src.models.forecasting import forecast_next_month

forecast_df = forecast_next_month(model, monthly, feature_cols)
print(forecast_df.head())
#    GoodsID  forecast_month  predicted_next_month
# 0    12345      2025-12-01                  45.2
# 1    67890      2025-12-01                 123.7
src/visualization/plots.py
توابع رسم نمودار:

Python

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
src/utils/helpers.py
توابع کمکی:

Python

def set_seed(seed=42):
    """تنظیم random seed."""

def ensure_dir(path) -> Path:
    """اطمینان از وجود دایرکتوری."""

def sample_goods_id(df, min_lag=8) -> int:
    """انتخاب تصادفی یک محصول معتبر."""

def print_section(title, char="=", width=60):
    """چاپ هدر بخش."""
🔧 Feature Engineering
ویژگی‌های ساخته شده
نوع	نام	تعداد	توضیح
Lag	lag_1 تا lag_24	24	فروش ماه‌های قبل
Rolling	rolling_3, rolling_6, rolling_9, rolling_12	4	میانگین متحرک
Date	year, month, quarter	3	ویژگی‌های تاریخ
Other	Price	1	میانگین قیمت
مجموع		32	
فرآیند Feature Engineering
text

Raw Data (Daily)
       │
       ▼
┌─────────────────┐
│ Add Date Feats  │  year, month, quarter
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Aggregate Monthly│  sum(MainQty), mean(Price)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Fill Missing    │  ماه‌های خالی = 0
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Create Lags     │  lag_1, lag_2, ..., lag_24
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Create Rolling  │  rolling_3, 6, 9, 12
└────────┬────────┘
         │
         ▼
   Monthly Data
   (با 32 feature)
📈 مدل و ارزیابی
الگوریتم
XGBoost Regressor
Objective: reg:squarederror
Cross-Validation: TimeSeriesSplit با 3 fold
Hyperparameters
پارامتر	مقدار
n_estimators	100
max_depth	5
learning_rate	0.1
subsample	0.8
colsample_bytree	1.0
معیارهای ارزیابی
معیار	مقدار	توضیح
MAE	3.82	میانگین خطای مطلق
RMSE	10.62	ریشه میانگین مربعات خطا
R²	0.790	ضریب تعیین (79% واریانس)
Grid Search Parameters
Python

PARAM_GRID = {
    'n_estimators': [100, 300, 500],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.03, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}
# Total: 108 combinations × 3 folds = 324 fits
🖥️ داشبورد تعاملی
اجرا
Bash

streamlit run app.py
امکانات
امکان	توضیح
🔍 جستجو	وارد کردن GoodsID
📋 Dropdown	انتخاب از لیست
🎲 Random	انتخاب تصادفی
⬅️➡️ Navigation	جابجایی بین محصولات
📊 Metrics	نمایش MAE, RMSE, R²
🔮 Forecast	پیش‌بینی ماه آینده
📋 Data Table	مشاهده داده خام
📥 Download	دانلود نمودار
نمای داشبورد
text

┌────────────────────────────────────────────────────────────┐
│  📊 Sales Forecast Dashboard                               │
├──────────────┬─────────────────────────────────────────────┤
│              │                                             │
│ 🔍 Selection │      ┌─────────────────────────────┐        │
│              │      │                             │        │
│ ○ Dropdown   │      │    [نمودار پیش‌بینی]        │        │
│ ○ Search     │      │                             │        │
│ ○ Random     │      └─────────────────────────────┘        │
│              │                                             │
│ [⬅️] [➡️]    ├─────────────────────────────────────────────┤
│              │                                             │
│ 📦 Products: │  MAE: 3.82  │  RMSE: 10.62  │  R²: 0.790   │
│    23,542    │                                             │
│              ├─────────────────────────────────────────────┤
│              │                                             │
│              │  🔮 Forecast: 2025-12 → 45.2 units          │
│              │                                             │
└──────────────┴─────────────────────────────────────────────┘
⚙️ تنظیمات
تغییر مسیر داده
Python

# در src/config/base_config.py
RAW_DATA_PATH = DATA_DIR / "your_data.csv"
تغییر تعداد Lags
Python

# در src/config/base_config.py
LAGS = list(range(1, 13))  # فقط 12 ماه
تغییر تعداد ماه‌های تست
Python

# در src/config/base_config.py
TEST_MONTHS = 3  # 3 ماه آخر برای تست
اجرای Grid Search
Python

# در main.py
pipeline = main(
    run_grid_search=True,  # فعال کردن GridSearch
    save_plots=True
)
🧪 تست
اجرای تست‌ها
Bash

pytest tests/ -v
تست دستی
Python

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
📊 نمونه خروجی‌ها
فایل پیش‌بینی (seasonal_forecast_results.csv)
csv

GoodsID,forecast_month,predicted_next_month
12345,2025-12-01,45.23
67890,2025-12-01,123.67
11111,2025-12-01,8.45
...
نمودارها
نمودار	فایل
Feature Importance	reports/figures/feature_importance.png
Actual vs Predicted	reports/figures/actual_vs_predicted.png
Residuals	reports/figures/residuals.png
Sample Forecast	reports/figures/forecast_goods_*.png
🔄 نحوه استفاده از ماژول‌ها به صورت جداگانه
فقط بارگذاری داده
Python

from src.data.data_loader import load_and_clean_data
df = load_and_clean_data("path/to/data.csv")
فقط پیش‌بینی با مدل موجود
Python

from src.models.model_training import load_model
from src.models.forecasting import forecast_next_month

model = load_model()
forecast = forecast_next_month(model, monthly, feature_cols)
فقط رسم نمودار
Python

from src.visualization.plots import plot_historical_and_predictions

plot_historical_and_predictions(
    model, monthly, feature_cols, 
    goods_id=12345,
    save_path="my_plot.png"
)
📝 لایسنس
MIT License

👨‍💻 نویسنده
GitHub: @your-username
🙏 منابع
XGBoost Documentation
Streamlit Documentation
Scikit-learn Documentation
text


---

این README شامل:

| بخش | محتوا |
|-----|-------|
| معرفی | توضیح پروژه و pipeline |
| ویژگی‌ها | جدول امکانات |
| ساختار | درخت کامل فایل‌ها |
| نصب | دستورات گام به گام |
| استفاده | سه روش اجرا |
| ماژول‌ها | توضیح هر فایل + مثال کد |
| Features | جدول و فرآیند |
| مدل | پارامترها و metrics |
| داشبورد | امکانات و نمای UI |
| تنظیمات | نحوه تغییر config |
| تست | دستورات pytest |
