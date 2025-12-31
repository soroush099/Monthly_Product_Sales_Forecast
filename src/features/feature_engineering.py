import pandas as pd
import numpy as np
import jdatetime
from typing import List, Tuple

from src.config.base_config import LAGS, ROLLING_WINDOWS


def add_jalali_features(df: pd.DataFrame) -> pd.DataFrame:
    """CONVERT TO PERSIAN CALENDAR."""
    df = df.copy()

    def get_jalali_date(g_date):
        return jdatetime.date.fromgregorian(date=g_date)

    jalali_dates = df['date'].apply(get_jalali_date)

    df['j_year'] = jalali_dates.apply(lambda x: x.year)
    df['j_month'] = jalali_dates.apply(lambda x: x.month)

    df['is_esfand'] = (df['j_month'] == 12).astype(int)
    df['is_farvardin'] = (df['j_month'] == 1).astype(int)
    df['is_shahrivar'] = (df['j_month'] == 6).astype(int)
    df['months_to_norouz'] = 12 - df['j_month']

    print("✅ Jalali features added.")
    return df


def aggregate_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate data to monthly level."""
    df['g_year'] = df['Miladi'].dt.year
    df['g_month'] = df['Miladi'].dt.month

    monthly = (
        df.groupby(['GoodsID', 'g_year', 'g_month'], as_index=False)
        .agg({'MainQty': 'sum', 'Price': 'mean'})
    )

    monthly['date'] = pd.to_datetime(
        monthly[['g_year', 'g_month']].rename(columns={'g_year': 'year', 'g_month': 'month'}).assign(day=1)
    )

    return monthly


def fill_missing_months(monthly: pd.DataFrame) -> pd.DataFrame:
    """Fill missing months."""
    min_date = monthly['date'].min()
    max_date = monthly['date'].max()
    full_months = pd.date_range(min_date, max_date, freq='MS')

    full_index = pd.MultiIndex.from_product(
        [monthly['GoodsID'].unique(), full_months],
        names=['GoodsID', 'date']
    )

    monthly_full = monthly.set_index(['GoodsID', 'date']).reindex(full_index).reset_index()
    monthly_full['MainQty'] = monthly_full['MainQty'].fillna(0)
    monthly_full['Price'] = monthly_full.groupby('GoodsID')['Price'].ffill().bfill()

    # Safety check: If Price is still NaN (no history), fill with 0
    monthly_full['Price'] = monthly_full['Price'].fillna(0)

    return monthly_full


def create_advanced_stats(df: pd.DataFrame, windows: List[int]) -> Tuple[pd.DataFrame, List[str]]:
    """Create EMA and Volatility."""
    df = df.copy()
    features = []

    for w in windows:
        mean_col = f'rolling_mean_{w}'
        df[mean_col] = df.groupby('GoodsID')['MainQty'].transform(lambda x: x.shift(1).rolling(w).mean())
        features.append(mean_col)

        ema_col = f'ema_{w}'
        df[ema_col] = df.groupby('GoodsID')['MainQty'].transform(lambda x: x.shift(1).ewm(span=w).mean())
        features.append(ema_col)

        max_col = f'rolling_max_{w}'
        df[max_col] = df.groupby('GoodsID')['MainQty'].transform(lambda x: x.shift(1).rolling(w).max())
        features.append(max_col)

    print(f"✅ Advanced stats created.")
    return df, features


def create_price_momentum_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Create Price Dynamics and Sales Momentum (With Infinity Protection)."""
    df = df.copy()
    feats = []

    # 1. Price Ratio
    df['rolling_price_6'] = df.groupby('GoodsID')['Price'].transform(lambda x: x.shift(1).rolling(6).mean())

    # جلوگیری از تقسیم بر صفر (اگر قیمت میانگین 0 بود، تقسیم انجام نشود)
    # اضافه کردن یک عدد بسیار کوچک (epsilon) به مخرج
    df['price_ratio'] = df['Price'] / (df['rolling_price_6'] + 1e-6)

    # پاکسازی نهایی: تبدیل inf به 1 و NaN به 1
    df['price_ratio'] = df['price_ratio'].replace([np.inf, -np.inf], 1).fillna(1)
    feats.append('price_ratio')

    # 2. Year-over-Year Growth
    if 'lag_12' in df.columns:
        # مخرج کسر + 1 شده تا صفر نشود
        df['yoy_growth'] = (df['MainQty'].shift(1) - df['lag_12']) / (df['lag_12'] + 1)

        # پاکسازی نهایی: تبدیل inf به 0 (رشد صفر)
        df['yoy_growth'] = df['yoy_growth'].replace([np.inf, -np.inf], 0).fillna(0)
        feats.append('yoy_growth')

    return df, feats


def build_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Master Pipeline."""
    monthly = aggregate_monthly(df)
    monthly = fill_missing_months(monthly)
    monthly = add_jalali_features(monthly)
    jalali_cols = ['j_month', 'is_esfand', 'is_farvardin', 'is_shahrivar', 'months_to_norouz']

    lags = LAGS
    monthly = monthly.sort_values(['GoodsID', 'date'])
    lag_cols = []
    for lag in lags:
        col = f'lag_{lag}'
        monthly[col] = monthly.groupby('GoodsID')['MainQty'].shift(lag)
        lag_cols.append(col)

    monthly, stat_cols = create_advanced_stats(monthly, ROLLING_WINDOWS)
    monthly, mom_cols = create_price_momentum_features(monthly)

    monthly = monthly.dropna(subset=lag_cols).reset_index(drop=True)

    # SAFETY CLEAN: Replace any lingering infinity values in the whole dataframe
    monthly = monthly.replace([np.inf, -np.inf], 0)

    all_features = lag_cols + stat_cols + mom_cols + jalali_cols + ['Price']

    print(f"✅ Final Dataset: {len(monthly):,} rows, {len(all_features)} features")
    return monthly, all_features


def get_feature_columns(lags=None, windows=None) -> List[str]:
    if lags is None: lags = LAGS
    if windows is None: windows = ROLLING_WINDOWS

    lag_cols = [f'lag_{l}' for l in lags]
    stat_cols = []
    for w in windows:
        stat_cols.extend([f'rolling_mean_{w}', f'ema_{w}', f'rolling_max_{w}'])

    mom_cols = ['price_ratio', 'yoy_growth']
    jalali_cols = ['j_month', 'is_esfand', 'is_farvardin', 'is_shahrivar', 'months_to_norouz']

    return lag_cols + stat_cols + mom_cols + jalali_cols + ['Price']
