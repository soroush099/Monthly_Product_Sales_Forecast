import pandas as pd
import numpy as np


class SalesDataProcessor:
    def __init__(self, file_path: str, auto_process: bool = True, n_lags: int = 24):
        """
        Parameters:
        -----------
        file_path : str
            file path
        auto_process : bool
            Automatic data processing (default True)
        n_lags : int
            Number of lags (default 24)
        """
        self.file_path = file_path
        self.n_lags = n_lags
        self.raw_data = None
        self.filled_data = None
        self.lag_features = None

        if auto_process:
            self.load_and_process()

    def load_and_process(self):
        """Full data loading and processing"""
        self.raw_data = self._load_data()
        print(f"✅ Load: {len(self.raw_data)} records")

        self.filled_data = self._fill_missing_months()
        added = len(self.filled_data) - len(self.raw_data)
        print(f"✅ Missing months: {added} records added")

        self.lag_features = self._create_lag_features()
        print(f"✅ Lag features: {self.lag_features.shape}")

        return self.lag_features

    def _load_data(self):
        """Loading data from file"""
        data = pd.read_csv(self.file_path)
        data.columns = data.columns.str.strip()
        data.sort_values(['Code', 'Year', 'Month'], inplace=True)
        return data

    def _fill_missing_months(self):
        """Making up for lost months"""
        if self.raw_data is None:
            raise ValueError("Load the data first")

        result_list = []

        for code, group in self.raw_data.groupby('Code'):
            group = group.sort_values(['Year', 'Month']).reset_index(drop=True)

            group['date_key'] = group['Year'] * 100 + group['Month']
            existing_data = dict(zip(group['date_key'], group['MainQty']))

            min_date = group['date_key'].min()
            max_date = group['date_key'].max()

            min_year = min_date // 100
            min_month = min_date % 100
            max_year = max_date // 100
            max_month = max_date % 100

            complete_data = []
            current_year = min_year
            current_month = min_month

            while (current_year * 100 + current_month) <= max_date:
                date_key = current_year * 100 + current_month

                if date_key in existing_data:
                    complete_data.append({
                        'Code': code,
                        'Year': current_year,
                        'Month': current_month,
                        'MainQty': existing_data[date_key],
                        'Interpolated': False
                    })
                else:
                    interpolated_value = self._calculate_interpolated_value(
                        current_year, current_month, existing_data
                    )

                    complete_data.append({
                        'Code': code,
                        'Year': current_year,
                        'Month': current_month,
                        'MainQty': interpolated_value,
                        'Interpolated': True
                    })

                current_month += 1
                if current_month > 12:
                    current_month = 1
                    current_year += 1

            result_list.extend(complete_data)

        return pd.DataFrame(result_list)

    @staticmethod
    def _calculate_interpolated_value(year, month, existing_data):
        """Calculating the interpolated value for a missing month"""
        prev_value = None
        next_value = None

        # Search for previous value
        temp_year, temp_month = year, month
        for _ in range(12):
            temp_month -= 1
            if temp_month < 1:
                temp_month = 12
                temp_year -= 1
            temp_key = temp_year * 100 + temp_month
            if temp_key in existing_data:
                prev_value = existing_data[temp_key]
                break

        # Search for the next value
        temp_year, temp_month = year, month
        for _ in range(12):
            temp_month += 1
            if temp_month > 12:
                temp_month = 1
                temp_year += 1
            temp_key = temp_year * 100 + temp_month
            if temp_key in existing_data:
                next_value = existing_data[temp_key]
                break

        # Calculating the average
        if prev_value is not None and next_value is not None:
            return (prev_value + next_value) / 2
        elif prev_value is not None:
            return prev_value
        elif next_value is not None:
            return next_value
        else:
            return 0

    def _create_lag_features(self):
        """creation lag features"""
        if self.filled_data is None:
            raise ValueError("Fill in the missing months first.")

        result_list = []

        for code, group in self.filled_data.groupby('Code'):
            group = group.sort_values(['Year', 'Month']).reset_index(drop=True)

            values = group['MainQty'].tail(self.n_lags).values

            if len(values) < self.n_lags:
                values = np.pad(values, (self.n_lags - len(values), 0),
                                mode='constant', constant_values=np.nan)

            row = {'Code': code}

            for i in range(self.n_lags):
                lag_num = i + 1
                value_idx = self.n_lags - lag_num
                row[f'lag{lag_num}'] = values[value_idx]

            result_list.append(row)

        result_df = pd.DataFrame(result_list)

        columns_order = ['Code'] + [f'lag{i}' for i in range(1, self.n_lags + 1)]
        result_df = result_df[columns_order]

        return result_df

    def save_results(self, output_path: str = None):
        """save results"""
        if self.lag_features is not None:
            if output_path is None:
                output_path = self.file_path.replace('.csv', '_processed.csv')
            self.lag_features.to_csv(output_path, index=False)
            print(f"✅ Results saved in {output_path}")


def load_data(file_path: str):
    processor = SalesDataProcessor(file_path, auto_process=True)
    return processor.lag_features
