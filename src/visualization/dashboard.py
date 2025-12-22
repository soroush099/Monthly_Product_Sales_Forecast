"""
Interactive Streamlit Dashboard for Sales Forecasting.
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parents[2]))

from src.config.base_config import MODEL_SAVE_PATH, FIGURES_DIR
from src.models.model_training import load_model
from src.utils.helpers import ensure_dir


def load_data_and_model():
    """Load model and data from session or disk."""
    if 'model' not in st.session_state:
        st.session_state.model = load_model(MODEL_SAVE_PATH)
    return st.session_state.model


def plot_product_forecast(
        model,
        monthly: pd.DataFrame,
        feature_cols: list,
        goods_id: int,
        forecast_df: pd.DataFrame = None,
        split_date: pd.Timestamp = None
):
    """Create forecast plot for a specific product."""

    # Filter product data
    product_df = monthly[monthly['GoodsID'] == goods_id].copy()
    product_df = product_df.sort_values('date')

    if len(product_df) == 0:
        st.warning(f"⚠️ No data found for GoodsID {goods_id}")
        return None

    # Predict
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
        markersize=6
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


def run_dashboard(monthly: pd.DataFrame, feature_cols: list, forecast_df: pd.DataFrame, split_date):
    """Run the Streamlit dashboard."""

    st.set_page_config(
        page_title="Sales Forecast Dashboard",
        page_icon="📊",
        layout="wide"
    )

    st.title("📊 Sales Forecast Dashboard")
    st.markdown("---")

    # Load model
    model = load_data_and_model()

    # Sidebar for product selection
    st.sidebar.header("🔍 Product Selection")

    # Get valid products (with enough data)
    valid_products = monthly[~monthly['lag_8'].isna()]['GoodsID'].unique()
    valid_products = sorted(valid_products)

    # Product selection methods
    selection_method = st.sidebar.radio(
        "Selection Method:",
        ["Dropdown", "Search", "Random"]
    )

    if selection_method == "Dropdown":
        selected_goods_id = st.sidebar.selectbox(
            "Select GoodsID:",
            options=valid_products,
            index=0
        )

    elif selection_method == "Search":
        search_id = st.sidebar.number_input(
            "Enter GoodsID:",
            min_value=int(min(valid_products)),
            max_value=int(max(valid_products)),
            value=int(valid_products[0])
        )
        if search_id in valid_products:
            selected_goods_id = search_id
        else:
            st.sidebar.warning(f"GoodsID {search_id} not found!")
            selected_goods_id = valid_products[0]

    else:  # Random
        if st.sidebar.button("🎲 Random Product"):
            st.session_state.random_id = np.random.choice(valid_products)

        selected_goods_id = st.session_state.get('random_id', valid_products[0])

    st.sidebar.markdown("---")
    st.sidebar.info(f"📦 Total Products: {len(valid_products):,}")

    # Navigation buttons
    st.sidebar.markdown("### ⬅️➡️ Navigation")
    col1, col2 = st.sidebar.columns(2)

    current_idx = list(valid_products).index(selected_goods_id)

    with col1:
        if st.button("⬅️ Previous"):
            new_idx = (current_idx - 1) % len(valid_products)
            selected_goods_id = valid_products[new_idx]
            st.rerun()

    with col2:
        if st.button("Next ➡️"):
            new_idx = (current_idx + 1) % len(valid_products)
            selected_goods_id = valid_products[new_idx]
            st.rerun()

    # Main content
    st.header(f"📦 Product: {selected_goods_id}")

    # Create plot
    result = plot_product_forecast(
        model, monthly, feature_cols,
        selected_goods_id, forecast_df, split_date
    )

    if result is not None:
        fig, product_df = result

        # Display plot
        st.pyplot(fig)
        plt.close(fig)

        # Metrics
        st.markdown("---")
        st.subheader("📈 Performance Metrics")

        metrics = calculate_product_metrics(product_df)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("MAE", f"{metrics['MAE']:.2f}")
        with col2:
            st.metric("RMSE", f"{metrics['RMSE']:.2f}")
        with col3:
            st.metric("R²", f"{metrics['R²']:.3f}")
        with col4:
            st.metric("Data Points", f"{metrics['Data Points']}")

        col5, col6 = st.columns(2)
        with col5:
            st.metric("Total Sales", f"{metrics['Total Sales']:,.0f}")
        with col6:
            st.metric("Avg Monthly", f"{metrics['Avg Monthly Sales']:.1f}")

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


def fig_to_bytes(fig):
    """Convert matplotlib figure to bytes for download."""
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    return buf.getvalue()


# For running directly
if __name__ == "__main__":
    st.error("Please run this from the main app entry point!")