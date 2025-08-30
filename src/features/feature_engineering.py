import pandas as pd
from sklearn.preprocessing import LabelEncoder


def create_features(data: pd.DataFrame):
    df = data.copy()

    for lag in [1, 2, 3]:
        df[f'Sales_Lag_{lag}'] = df.groupby('Code')['MainQty'].shift(lag)

    df['Sales_Lag_12'] = df.groupby('Code')['MainQty'].shift(12)
    df['Rolling_Mean_3'] = df.groupby('Code')['MainQty'].shift(1).rolling(window=3, min_periods=1).mean()
    df['Rolling_Std_3'] = df.groupby('Code')['MainQty'].shift(1).rolling(window=3, min_periods=1).std()

    df.dropna(inplace=True)

    le = LabelEncoder()
    df['Code_Encoded'] = le.fit_transform(df['Code'])

    return df, le
