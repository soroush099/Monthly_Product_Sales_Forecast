"""
Interactive Streamlit Dashboard for Sales Forecasting.
Fixed Navigation Logic & Removed duplicate page config.
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import io
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parents[2]))

# from src.config.base_config import MODEL_SAVE_PATH
# from src.models.model_training import load_model


def plot_product_forecast(
        model,
        monthly: pd.DataFrame,
        feature_cols: list,
        goods_id: int,
        forecast_df: pd.DataFrame = None,
        split_date: pd.Timestamp = None
):
    """Create forecast plot using Matplotlib."""

    # Filter product data
    product_df = monthly[monthly['GoodsID'] == goods_id].copy()
    product_df = product_df.sort_values('date')

    if len(product_df) == 0:
        st.warning(f"⚠️ No data found for GoodsID {goods_id}")
        return None, None

    # Predict using the model passed from main app
    product_df['predicted'] = model.predict(product_df[feature_cols])

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Actual sales
    ax.plot(
        product_df['date'],
        product_df['MainQty'],
        label='Actual Sales',
        marker='o',
        color='steelblue',
        linewidth=2,
        markersize=4
    )

    # Model predictions
    ax.plot(
        product_df['date'],
        product_df['predicted'],
        label='Model Prediction',
        linestyle='--',
        marker='.',
        color='orange',
        linewidth=2
    )

    # Forecast for next month
    if forecast_df is not None:
        forecast_row = forecast_df[forecast_df['GoodsID'] == goods_id]
        if not forecast_row.empty:
            ax.scatter(
                forecast_row['forecast_month'],
                forecast_row['predicted_next_month'],
                color='red',
                s=200,
                marker='*',
                label='Next Month Forecast',
                zorder=5
            )

    # Split date marker
    if split_date is not None:
        ax.axvline(
            split_date,
            color='gray',
            linestyle=':',
            linewidth=2,
            label='Train/Test Split'
        )

    ax.set_title(f"GoodsID {goods_id} — Sales Forecast", fontsize=14, fontweight='bold')
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Sales Quantity", fontsize=12)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    return fig, product_df


def calculate_product_metrics(product_df: pd.DataFrame) -> dict:
    """Calculate metrics for a specific product."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    y_true = product_df['MainQty']
    y_pred = product_df['predicted']

    return {
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'R²': r2_score(y_true, y_pred),
        'Total Sales': y_true.sum(),
        'Avg Monthly Sales': y_true.mean(),
        'Data Points': len(y_true)
    }


def fig_to_bytes(fig):
    """Convert matplotlib figure to bytes for download."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    return buf.getvalue()


def run_dashboard(monthly: pd.DataFrame, feature_cols: list, forecast_df: pd.DataFrame, split_date):
    """
    Main dashboard function.
    Note: Model is loaded in app.py and passed via session_state implicitly
    OR we can just instantiate a predictor here if needed.
    Ideally, app.py should pass the 'model' object, but to keep your structure:
    """

    # 1. بازیابی مدل از Session State (چون در app.py لود شده)
    # اگر مدل وجود نداشت (نباید اتفاق بیفتد)، ارور می‌دهیم
    if 'model' not in st.session_state:
        # Fallback logic: Try to get model from the passed function arguments if possible?
        # But since run_dashboard signature is fixed, we assume app.py put model in session_state.
        # Let's check if we can import load_model here as fallback
        try:
            from src.config.base_config import MODEL_SAVE_PATH
            from src.models.model_training import load_model
            model = load_model(MODEL_SAVE_PATH)
        except:
            st.error("Model not found in session state or disk!")
            return
    else:
        # اگر app.py مدل را در session_state نگذاشته باشد، باید آنجا اصلاح شود.
        # اما بیایید فرض کنیم مدل را نداریم و باید لود کنیم:
        # راه حل بهتر: app.py مدل را به run_dashboard پاس بدهد.
        # اما چون سیگنیچر تابع شما فیکس است، اینجا لود می‌کنیم:
        from src.config.base_config import MODEL_SAVE_PATH
        from src.models.model_training import load_model
        model = load_model(MODEL_SAVE_PATH)

    st.markdown("---")

    # Sidebar for product selection
    st.sidebar.header("🔍 Product Selection")

    # Get valid products
    valid_products = monthly['GoodsID'].unique()
    valid_products = sorted(valid_products)
    total_products = len(valid_products)

    # --- اصلاح منطق دکمه‌ها با Session State ---
    if 'selected_idx' not in st.session_state:
        st.session_state.selected_idx = 0

    # Navigation buttons logic
    col_prev, col_next = st.sidebar.columns(2)

    with col_prev:
        if st.button("⬅️ Previous"):
            st.session_state.selected_idx = (st.session_state.selected_idx - 1) % total_products
            st.rerun()

    with col_next:
        if st.button("Next ➡️"):
            st.session_state.selected_idx = (st.session_state.selected_idx + 1) % total_products
            st.rerun()

    # Get current product based on index
    current_idx = st.session_state.selected_idx
    selected_goods_id = valid_products[current_idx]

    # نمایش دراپ‌داون (که با دکمه‌ها سینک باشد)
    # اگر کاربر از دراپ‌داون انتخاب کرد، ایندکس آپدیت شود
    selected_from_dropdown = st.sidebar.selectbox(
        "Jump to Product:",
        options=valid_products,
        index=current_idx,
        key="dropdown_selector"
    )

    # اگر انتخاب دراپ‌داون با ایندکس فعلی فرق داشت، یعنی کاربر دستی عوض کرده
    if selected_from_dropdown != selected_goods_id:
        st.session_state.selected_idx = list(valid_products).index(selected_from_dropdown)
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.info(f"📦 Total Products: {total_products:,}")
    st.sidebar.text(f"Current Index: {current_idx + 1}")

    # Main content
    st.header(f"📦 Product: {selected_goods_id}")

    # Create plot
    fig, product_df = plot_product_forecast(
        model, monthly, feature_cols,
        selected_goods_id, forecast_df, split_date
    )

    if fig is not None:
        # Display plot
        st.pyplot(fig)
        plt.close(fig) # بستن فیگور برای جلوگیری از مصرف حافظه

        # Metrics
        st.markdown("---")
        st.subheader("📈 Performance Metrics")

        metrics = calculate_product_metrics(product_df)

        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("MAE", f"{metrics['MAE']:.2f}")
        with col2: st.metric("RMSE", f"{metrics['RMSE']:.2f}")
        with col3: st.metric("R²", f"{metrics['R²']:.3f}")
        with col4: st.metric("Data Points", f"{metrics['Data Points']}")

        col5, col6 = st.columns(2)
        with col5: st.metric("Total Sales", f"{metrics['Total Sales']:,.0f}")
        with col6: st.metric("Avg Monthly", f"{metrics['Avg Monthly Sales']:.1f}")

        # Next month forecast
        st.markdown("---")
        st.subheader("🔮 Next Month Forecast")

        if forecast_df is not None:
            forecast_row = forecast_df[forecast_df['GoodsID'] == selected_goods_id]
            if not forecast_row.empty:
                forecast_value = forecast_row['predicted_next_month'].values[0]
                forecast_month = forecast_row['forecast_month'].values[0]

                st.success(f"📅 **{pd.Timestamp(forecast_month).strftime('%Y-%m')}**: "
                           f"Predicted Sales = **{forecast_value:.1f}** units")
            else:
                st.info("No forecast available for this product")

        # Data table
        with st.expander("📋 View Raw Data"):
            display_df = product_df[['date', 'MainQty', 'predicted']].copy()
            display_df.columns = ['Date', 'Actual Sales', 'Predicted Sales']
            display_df['Error'] = display_df['Actual Sales'] - display_df['Predicted Sales']
            st.dataframe(display_df.sort_values('Date', ascending=False), use_container_width=True)

        # Download button
        st.sidebar.markdown("---")
        st.sidebar.download_button(
            label="📥 Download Plot",
            data=fig_to_bytes(fig),
            file_name=f"forecast_goods_{selected_goods_id}.png",
            mime="image/png"
        )
