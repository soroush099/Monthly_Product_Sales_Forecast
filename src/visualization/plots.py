import matplotlib.pyplot as plt
import os
import pandas as pd
import arabic_reshaper
from bidi.algorithm import get_display


def plot_results(data, df_predictions_pivoted, codes_to_plot, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for code_to_plot in codes_to_plot:
        plt.figure(figsize=(12, 6))
        actual_data = data[data['Code'] == code_to_plot]

        product_name = actual_data['Name'].iloc[0] if not actual_data.empty else ""

        for year, color, marker, style in [(1402, 'black', 'o', '-'), (1403, 'red', 's', '-'),
                                           (1404, 'green', 'D', '-')]:
            year_data = actual_data[actual_data['Year'] == year]
            if not year_data.empty:
                plt.plot(year_data['Month'], year_data['MainQty'], marker=marker, linestyle=style, color=color,
                         label=f'Actual {year}')

        if code_to_plot in df_predictions_pivoted.index.get_level_values('Code'):
            prediction_data = df_predictions_pivoted.loc[code_to_plot].iloc[0]
            months_numeric = pd.to_numeric(prediction_data.index)
            plt.plot(months_numeric, prediction_data.values, marker='^', linestyle='--', color='blue',
                     label='Prediction 1404')

        reshaped_text = arabic_reshaper.reshape(product_name)
        bidi_text = get_display(reshaped_text)

        plt.title(f'Seasonal Forecast vs. Actuals for Code: {code_to_plot} - {bidi_text}', fontsize=14)
        plt.xlabel('Month')
        plt.ylabel('Sales Quantity')
        plt.xticks(range(1, 13))
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'plot_{code_to_plot}.png'), dpi=150)
        plt.close()
