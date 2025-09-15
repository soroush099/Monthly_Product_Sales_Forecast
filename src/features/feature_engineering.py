import pandas as pd
from sklearn.preprocessing import LabelEncoder


def create_features(data: pd.DataFrame, max_lag=12, rolling_window=3):
    df = data.copy()

    for lag in range(1, max_lag + 1):
        df[f'Sales_Lag_{lag}'] = df.groupby('Code')['MainQty'].shift(lag)

    df['Rolling_Mean_3'] = df.groupby('Code')['MainQty'].shift(1).rolling(window=rolling_window, min_periods=1).mean()
    df['Rolling_Std_3'] = df.groupby('Code')['MainQty'].shift(1).rolling(window=rolling_window, min_periods=1).std()

    df.dropna(inplace=True)

    le = LabelEncoder()
    df['Code_Encoded'] = le.fit_transform(df['Code'])

    return df, le
