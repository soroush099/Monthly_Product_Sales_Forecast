"""
Data loading and basic cleaning module.
"""
import pandas as pd
from pathlib import Path
from typing import Union

from src.config.base_config import RAW_DATA_PATH


def load_raw_data(filepath: Union[str, Path] = None) -> pd.DataFrame:
    """
    Load raw sales data from CSV file.

    Parameters
    ----------
    filepath : str or Path, optional
        Path to the CSV file. If None, uses default from config.

    Returns
    -------
    pd.DataFrame
        Raw dataframe loaded from CSV.
    """
    if filepath is None:
        filepath = RAW_DATA_PATH

    df = pd.read_csv(filepath)
    print(f"✅ Loaded {len(df):,} rows from {filepath}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform basic data cleaning.

    - Convert 'Miladi' to datetime
    - Drop rows with missing critical values
    - Convert 'MainQty' to numeric
    - Convert 'IsHoliday' to int

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe.
    """
    df = df.copy()

    # Convert date column
    df['Miladi'] = pd.to_datetime(df['Miladi'], errors='coerce')

    # Drop rows with missing critical values
    df = df.dropna(subset=['Miladi', 'GoodsID', 'MainQty'])

    # Ensure numeric quantity
    df['MainQty'] = pd.to_numeric(df['MainQty'], errors='coerce').fillna(0)

    # Convert IsHoliday to numeric (if exists)
    if 'IsHoliday' in df.columns:
        df['IsHoliday'] = df['IsHoliday'].astype(int)

    print(f"✅ Cleaned data: {len(df):,} rows remaining")
    return df


def load_and_clean_data(filepath: Union[str, Path] = None) -> pd.DataFrame:
    """
    Load and clean data in one step.

    Parameters
    ----------
    filepath : str or Path, optional
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe.
    """
    df = load_raw_data(filepath)
    df = clean_data(df)
    return df