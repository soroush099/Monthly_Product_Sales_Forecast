"""
Auxiliary comparison charts and visualization utilities.
"""
import pandas as pd
import matplotlib.pyplot as plt
from typing import List


def compare_products(
        monthly: pd.DataFrame,
        goods_ids: List[int],
        save_path: str = None,
        show: bool = True
) -> None:
    """
    Compare sales trends across multiple products.

    Parameters
    ----------
    monthly : pd.DataFrame
        Monthly dataframe.
    goods_ids : list of int
        List of GoodsIDs to compare.
    save_path : str, optional
        Path to save the figure.
    show : bool
        Whether to display the plot.
    """
    plt.figure(figsize=(14, 6))

    for goods_id in goods_ids:
        product_df = monthly[monthly['GoodsID'] == goods_id].sort_values('date')
        plt.plot(
            product_df['date'],
            product_df['MainQty'],
            marker='o',
            label=f'GoodsID {goods_id}'
        )

    plt.title("Sales Comparison Across Products")
    plt.xlabel("Date")
    plt.ylabel("Sales Quantity")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()