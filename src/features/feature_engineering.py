import pandas as pd
import numpy as np
import jdatetime
from sklearn.preprocessing import LabelEncoder
from ..utils.jalali_utils import gregorian_to_jalali, get_jalali_season


def create_features(data: pd.DataFrame, max_lag=12, rolling_windows=None):
    if rolling_windows is None:
        rolling_windows = [3, 6, 9,  12]
    df = data.copy()

    # دیگه نیازی به تبدیل تاریخ به شمسی نیست.
    def shamsi_to_gregorian(row):
        return jdatetime.date(row['Year'], row['Month'], 1).togregorian()
    df['Date'] = df.apply(shamsi_to_gregorian, axis=1)
    df['Date'] = pd.to_datetime(df['Date'])

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
        df[f'Rolling_Std_{window}'] = (
            df.groupby('Code')['MainQty']
            .shift(1)
            .rolling(window=window, min_periods=1)
            .std()
            .reset_index(level=0, drop=True)
        )

    df['Sales_LastYear'] = df.groupby('Code')['MainQty'].shift(12)

    le = LabelEncoder()
    df['Code_Encoded'] = le.fit_transform(df['Code'])

    df.dropna(inplace=True)

    return df, le


def create_features_for_prediction(history_data, code, le, future_year, future_month,
                                   max_lag=24, rolling_windows=None):

    if rolling_windows is None:
        rolling_windows = [3, 6, 12]
    future_date = jdatetime.date(int(future_year), int(future_month), 1).togregorian()
    future_date = pd.to_datetime(future_date)

    features = {
        'Code_Encoded': le.transform([code])[0],
        'Year': future_year,
        'Month': future_month,
        'Quarter': future_date.quarter,
        'DayOfYear': future_date.dayofyear
    }

    for lag in range(1, max_lag + 1):
        if len(history_data) >= lag:
            features[f'Sales_Lag_{lag}'] = history_data['MainQty'].iloc[-lag]
        else:
            features[f'Sales_Lag_{lag}'] = 0

    for window in rolling_windows:
        if len(history_data) >= window:
            rolling_data = history_data['MainQty'].tail(window)
            features[f'Rolling_Mean_{window}'] = rolling_data.mean()
            rolling_std = rolling_data.std()
            features[f'Rolling_Std_{window}'] = rolling_std if not np.isnan(rolling_std) else 0
        else:
            features[f'Rolling_Mean_{window}'] = history_data['MainQty'].mean() if not history_data.empty else 0
            features[f'Rolling_Std_{window}'] = 0

    features['Sales_LastYear'] = history_data['MainQty'].iloc[-12] if len(history_data) >= 12 else 0

    return features
