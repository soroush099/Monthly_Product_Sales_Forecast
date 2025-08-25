import pandas as pd


def load_data(file_path: str):
    data = pd.read_csv(file_path)
    data.columns = data.columns.str.strip()
    data.sort_values(['Code', 'Year', 'Month'], inplace=True)
    return data
