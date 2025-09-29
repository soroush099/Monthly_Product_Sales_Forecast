import pandas as pd
import numpy as np
import jdatetime
from sklearn.preprocessing import LabelEncoder


def create_features(data: pd.DataFrame, max_lag=12, rolling_windows=[3, 6, 12]):
    df = data.copy()

    def shamsi_to_gregorian(row):
        return jdatetime.date(row['Year'], row['Month'], 1).togregorian()
    df['Date'] = df.apply(shamsi_to_gregorian, axis=1)

    df['Quarter'] = df['Date'].dt.quarter
    df['DayOfYear'] = df['Date'].dt.dayofyear

    for lag in range(1, max_lag + 1):
        df[f'Sales_Lag_{lag}'] = df.groupby('Code')['MainQty'].shift(lag)

    for window in rolling_windows:
        df[f'Rolling_Mean_{window}'] = (
            df.groupby('Code')['MainQty']
            .shift(1)
            .rolling(window=window, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
    df['Rolling_Std_3'] = (
        df.groupby('Code')['MainQty']
        .shift(1)
        .rolling(window=rolling_windows, min_periods=1)
        .std()
        .reset_index(level=0, drop=True)
    )

    df['Sales_LastYear'] = df.groupby('Code')['MainQty'].shift(12)

    le = LabelEncoder()
    df['Code_Encoded'] = le.fit_transform(df['Code'])

    df.dropna(inplace=True)

    return df, le
