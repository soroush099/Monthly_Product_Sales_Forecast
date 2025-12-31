"""
Model evaluation module.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from xgboost import XGBRegressor


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Percentage Error.

    Parameters
    ----------
    y_true : array-like
        True values.
    y_pred : array-like
        Predicted values.

    Returns
    -------
    float
        MAPE value as percentage.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Avoid division by zero
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan

    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def evaluate_model(
        model: XGBRegressor,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        verbose: bool = True
) -> Dict[str, float]:
    """
    Evaluate model performance on test set.

    Parameters
    ----------
    model : XGBRegressor
        Trained model.
    X_test : pd.DataFrame
        Test features.
    y_test : pd.Series
        Test target.
    verbose : bool
        Whether to print results.

    Returns
    -------
    dict
        Dictionary with evaluation metrics.
    """
    y_pred = model.predict(X_test)

    metrics = {
        'MAE': mean_absolute_error(y_test, y_pred),
        'RMSE': root_mean_squared_error(y_test, y_pred),
        'R2': r2_score(y_test, y_pred),
        'MAPE': calculate_mape(y_test, y_pred)  # ✅ اضافه شد
    }

    if verbose:
        print("\n📊 Model Evaluation Results:")
        print(f"   MAE:  {metrics['MAE']:.2f}")
        print(f"   RMSE: {metrics['RMSE']:.2f}")
        print(f"   R²:   {metrics['R2']:.3f}")
        print(f"   MAPE: {metrics['MAPE']:.2f}%")  # ✅ اضافه شد

    return metrics


def get_predictions(
        model: XGBRegressor,
        X: pd.DataFrame
) -> np.ndarray:
    """
    Get predictions from the model.

    Parameters
    ----------
    model : XGBRegressor
        Trained model.
    X : pd.DataFrame
        Features to predict on.

    Returns
    -------
    np.ndarray
        Predictions.
    """
    return model.predict(X)


def evaluate_by_product(
        model: XGBRegressor,
        df: pd.DataFrame,
        feature_cols: list,
        target_col: str = 'MainQty'
) -> pd.DataFrame:
    """
    Evaluate model performance per product (GoodsID).

    Parameters
    ----------
    model : XGBRegressor
        Trained model.
    df : pd.DataFrame
        Dataframe with features and target.
    feature_cols : list
        List of feature column names.
    target_col : str
        Name of target column.

    Returns
    -------
    pd.DataFrame
        Dataframe with per-product metrics sorted by MAPE.
    """
    df = df.copy()
    df['predicted'] = model.predict(df[feature_cols])

    results = []
    for goods_id in df['GoodsID'].unique():
        product_df = df[df['GoodsID'] == goods_id]
        y_true = product_df[target_col].values
        y_pred = product_df['predicted'].values

        results.append({
            'GoodsID': goods_id,
            'MAE': mean_absolute_error(y_true, y_pred),
            'RMSE': root_mean_squared_error(y_true, y_pred),
            'R2': r2_score(y_true, y_pred) if len(y_true) > 1 else np.nan,
            'MAPE': calculate_mape(y_true, y_pred),  # ✅ اضافه شد
            'avg_sales': y_true.mean(),  # ✅ میانگین فروش
            'n_samples': len(product_df)
        })

    result_df = pd.DataFrame(results)

    # مرتب‌سازی بر اساس MAPE (بدترین‌ها اول)
    return result_df.sort_values('MAPE', ascending=False).reset_index(drop=True)


def print_worst_products(
        product_metrics: pd.DataFrame,
        top_n: int = 10
) -> None:
    """
    Print products with highest MAPE.

    Parameters
    ----------
    product_metrics : pd.DataFrame
        Output from evaluate_by_product.
    top_n : int
        Number of worst products to show.
    """
    print(f"\n⚠️ Top {top_n} Products with Highest Error (MAPE):")
    print("-" * 70)

    worst = product_metrics.head(top_n)

    for _, row in worst.iterrows():
        print(f"   GoodsID: {row['GoodsID']:>10} | "
              f"MAPE: {row['MAPE']:>6.1f}% | "
              f"MAE: {row['MAE']:>8.1f} | "
              f"Avg Sales: {row['avg_sales']:>8.1f} | "
              f"Samples: {row['n_samples']:>3}")