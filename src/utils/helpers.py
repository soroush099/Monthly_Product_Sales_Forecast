"""
Helper functions and utilities.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Union
import random


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.

    Parameters
    ----------
    seed : int
        Random seed value.
    """
    np.random.seed(seed)
    random.seed(seed)


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if it doesn't.

    Parameters
    ----------
    path : str or Path
        Directory path.

    Returns
    -------
    Path
        Path object.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def sample_goods_id(
    df: pd.DataFrame,
    min_records: int = 8,
    random_state: int = None
) -> int:
    """
    Sample a random GoodsID that has sufficient data.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with GoodsID column.
    min_records : int
        Minimum number of records required.
    random_state : int, optional
        Random state for reproducibility.

    Returns
    -------
    int
        Sampled GoodsID.
    """
    # Filter products with enough data
    valid_products = (
        df.groupby('GoodsID')
          .size()
          .reset_index(name='count')
    )
    valid_products = valid_products[valid_products['count'] >= min_records]

    if random_state is not None:
        np.random.seed(random_state)

    return np.random.choice(valid_products['GoodsID'])


def get_data_summary(df: pd.DataFrame) -> dict:
    """
    Get summary statistics of the dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to summarize.

    Returns
    -------
    dict
        Summary statistics.
    """
    return {
        'n_rows': len(df),
        'n_products': df['GoodsID'].nunique() if 'GoodsID' in df.columns else None,
        'date_range': (
            df['date'].min().strftime('%Y-%m-%d'),
            df['date'].max().strftime('%Y-%m-%d')
        ) if 'date' in df.columns else None,
        'columns': list(df.columns)
    }


def print_section(title: str, char: str = "=", width: int = 60) -> None:
    """
    Print a formatted section header.

    Parameters
    ----------
    title : str
        Section title.
    char : str
        Character to use for the line.
    width : int
        Width of the line.
    """
    print("\n" + char * width)
    print(f" {title}")
    print(char * width)