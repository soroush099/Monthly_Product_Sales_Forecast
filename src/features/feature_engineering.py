"""
Feature engineering module for sales forecasting.
"""
import pandas as pd
import numpy as np
from typing import List, Tuple

from src.config.base_config import LAGS, ROLLING_WINDOWS


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add basic date-based features.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with 'Miladi' datetime column.

    Returns
    -------
    pd.DataFrame
        Dataframe with added date features.
    """
    df = df.copy()
    df['year'] = df['Miladi'].dt.year
    df['month'] = df['Miladi'].dt.month
    df['quarter'] = df['Miladi'].dt.quarter
    return df


def aggregate_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate data to monthly level per GoodsID.

    Parameters
    ----------
    df : pd.DataFrame
        Daily-level dataframe.

    Returns
    -------
    pd.DataFrame
        Monthly aggregated dataframe.
    """
    monthly = (
        df.groupby(['GoodsID', 'year', 'month', 'quarter'], as_index=False)
        .agg({'MainQty': 'sum', 'Price': 'mean'})
    )

    # Create proper datetime for each month
    monthly['date'] = pd.to_datetime(
        monthly[['year', 'month']].assign(day=1)
    )

    print(f"✅ Aggregated to {len(monthly):,} monthly records")
    return monthly


def fill_missing_months(monthly: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing months with zeros to create continuous time series.

    Parameters
    ----------
    monthly : pd.DataFrame
        Monthly aggregated dataframe.

    Returns
    -------
    pd.DataFrame
        Dataframe with all months filled.
    """
    min_date = monthly['date'].min()
    max_date = monthly['date'].max()

    full_months = pd.date_range(min_date, max_date, freq='MS')

    # Cartesian product of GoodsID and months
    full_index = pd.MultiIndex.from_product(
        [monthly['GoodsID'].unique(), full_months],
        names=['GoodsID', 'date']
    )

    monthly_full = (
        monthly.set_index(['GoodsID', 'date'])
        .reindex(full_index)
        .reset_index()
    )

    # Fill missing values
    monthly_full['MainQty'] = monthly_full['MainQty'].fillna(0)
    monthly_full['Price'] = monthly_full['Price'].fillna(0)
    monthly_full['year'] = monthly_full['date'].dt.year
    monthly_full['month'] = monthly_full['date'].dt.month
    monthly_full['quarter'] = monthly_full['date'].dt.quarter

    print(f"✅ Filled missing months: {len(monthly_full):,} total records")
    return monthly_full


def create_lag_features(
        df: pd.DataFrame,
        lags: List[int] = None
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Create lag features for MainQty.

    Parameters
    ----------
    df : pd.DataFrame
        Monthly dataframe sorted by GoodsID and date.
    lags : list of int, optional
        List of lag periods. Defaults to config LAGS.

    Returns
    -------
    tuple
        (DataFrame with lag features, list of lag feature names)
    """
    if lags is None:
        lags = LAGS

    df = df.copy()
    df = df.sort_values(['GoodsID', 'date'])

    lag_features = []
    for lag in lags:
        col_name = f'lag_{lag}'
        df[col_name] = df.groupby('GoodsID')['MainQty'].shift(lag)
        lag_features.append(col_name)

    print(f"✅ Created {len(lag_features)} lag features")
    return df, lag_features


def create_rolling_features(
        df: pd.DataFrame,
        windows: List[int] = None
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Create rolling mean features for MainQty.

    Parameters
    ----------
    df : pd.DataFrame
        Monthly dataframe with lag features.
    windows : list of int, optional
        List of rolling window sizes. Defaults to config ROLLING_WINDOWS.

    Returns
    -------
    tuple
        (DataFrame with rolling features, list of rolling feature names)
    """
    if windows is None:
        windows = ROLLING_WINDOWS

    df = df.copy()

    rolling_features = []
    for window in windows:
        col_name = f'rolling_{window}'
        df[col_name] = (
            df.groupby('GoodsID')['MainQty']
            .transform(lambda x: x.shift(1).rolling(window).mean())
        )
        rolling_features.append(col_name)

    print(f"✅ Created {len(rolling_features)} rolling features")
    return df, rolling_features


def build_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Full feature engineering pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned raw dataframe.

    Returns
    -------
    tuple
        (Feature-engineered monthly dataframe, list of all feature column names)
    """
    # Add date features
    df = add_date_features(df)

    # Aggregate to monthly
    monthly = aggregate_monthly(df)

    # Fill missing months
    monthly = fill_missing_months(monthly)

    # Create lag features
    monthly, lag_features = create_lag_features(monthly)

    # Create rolling features
    monthly, rolling_features = create_rolling_features(monthly)

    # Drop rows with all null features
    monthly = monthly.dropna(
        how='all',
        subset=lag_features + rolling_features
    ).reset_index(drop=True)

    # Compile all feature columns
    all_features = (
            lag_features +
            rolling_features +
            ['Price', 'year', 'month', 'quarter']
    )

    print(f"✅ Feature engineering complete: {len(monthly):,} rows, {len(all_features)} features")
    print(f"   Unique GoodsIDs: {monthly['GoodsID'].nunique():,}")

    return monthly, all_features


def get_feature_columns(
        lags: List[int] = None,
        windows: List[int] = None
) -> List[str]:
    """
    Get list of all feature column names.

    Parameters
    ----------
    lags : list of int, optional
        List of lag periods.
    windows : list of int, optional
        List of rolling window sizes.

    Returns
    -------
    list
        List of feature column names.
    """
    if lags is None:
        lags = LAGS
    if windows is None:
        windows = ROLLING_WINDOWS

    lag_features = [f'lag_{lag}' for lag in lags]
    rolling_features = [f'rolling_{w}' for w in windows]

    return lag_features + rolling_features + ['Price', 'year', 'month', 'quarter']