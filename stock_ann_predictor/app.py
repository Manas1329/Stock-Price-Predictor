# =============================================================================
# app.py — Streamlit Web Application
# =============================================================================
# A clean, interactive web UI for the stock prediction system.
#
# Run with:
#   streamlit run app.py
#
# Features:
#   - Sidebar controls for ticker, date range, model settings
#   - Live download + preprocessing + training
#   - Interactive Plotly charts
#   - Metrics display
#   - Download predictions as CSV
# =============================================================================

import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ANN Stock Predictor",
    page_icon=":material/show_chart:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:wght@400');

    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #1565C0, #7B1FA2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #666;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .ui-icon {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-family: 'Material Symbols Rounded';
        font-size: 1.1em;
        line-height: 1;
        vertical-align: -0.14em;
    }
    .ui-icon--title {
        font-size: 1.25em;
        margin-right: 0.4rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f8f9ff, #e8ecff);
        border-left: 4px solid #1565C0;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .stAlert { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


def icon(name):
    return f'<span class="ui-icon">{name}</span>'


def title_icon(name):
    return f'<span class="ui-icon ui-icon--title">{name}</span>'


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    f'<p class="main-title">{title_icon("show_chart")}ANN Stock Price Predictor</p>',
    unsafe_allow_html=True,
)
st.markdown('<p class="subtitle">Artificial Neural Network · Built with TensorFlow & Streamlit</p>', unsafe_allow_html=True)

# ── Sidebar Controls ──────────────────────────────────────────────────────────
st.sidebar.markdown(f'{icon("settings")} <strong>Configuration</strong>', unsafe_allow_html=True)

ticker = st.sidebar.text_input(
    "Stock Ticker",
    value="AAPL",
    help="Examples: AAPL, TSLA, GOOGL, MSFT, INFY.NS"
).upper()

col1, col2 = st.sidebar.columns(2)
start_date = col1.date_input("Start Date", value=pd.to_datetime("2018-01-01"))
end_date   = col2.date_input("End Date",   value=pd.to_datetime("2024-01-01"))

st.sidebar.markdown(f'{icon("neurology")} <strong>Model Settings</strong>', unsafe_allow_html=True)
sequence_len  = st.sidebar.slider("Sequence Length (days)", 20, 120, 60, 10)
epochs        = st.sidebar.slider("Epochs",                 10, 100,  50,  5)
batch_size    = st.sidebar.selectbox("Batch Size", [16, 32, 64, 128], index=1)
test_split    = st.sidebar.slider("Test Split %", 10, 40, 20, 5) / 100
use_tech      = st.sidebar.checkbox("Use Technical Indicators", value=True)

hidden_layers_str = st.sidebar.text_input(
    "Hidden Layers (neurons, comma-separated)",
    value="128,64,32",
    help="e.g. '128,64,32' → 3 layers with 128, 64, 32 neurons"
)

run_btn = st.sidebar.button("Run Prediction", type="primary", use_container_width=True)

# ── Helper: parse hidden layers ───────────────────────────────────────────────
def parse_layers(s):
    try:
        return [int(x.strip()) for x in s.split(",") if x.strip().isdigit()]
    except:
        return [128, 64, 32]


# ── Main Pipeline (runs when button clicked) ──────────────────────────────────
if run_btn:
    import sys
    sys.path.insert(0, os.path.dirname(__file__))

    # Dynamically update config values before importing modules
    import config
    config.TICKER              = ticker
    config.START_DATE          = str(start_date)
    config.END_DATE            = str(end_date)
    config.SEQUENCE_LENGTH     = sequence_len
    config.EPOCHS              = epochs
    config.BATCH_SIZE          = batch_size
    config.TEST_SPLIT          = test_split
    config.USE_TECHNICAL_INDICATORS = use_tech
    config.HIDDEN_LAYERS       = parse_layers(hidden_layers_str)

    from src.data_loader  import download_stock_data
    from src.preprocessor import preprocess, inverse_transform_close
    from src.model        import build_ann, get_callbacks
    from sklearn.metrics  import mean_squared_error, mean_absolute_error, r2_score

    # ── Step 1: Download ──────────────────────────────────────────────────────
    with st.spinner(f"Downloading {ticker} data from Yahoo Finance..."):
        try:
            df = download_stock_data(
                ticker=ticker,
                start=str(start_date),
                end=str(end_date)
            )
            st.success(f"Downloaded {len(df):,} trading days of {ticker} data", icon=":material/check_circle:")
        except Exception as e:
            st.error(f"Failed to download data: {e}", icon=":material/error:")
            st.stop()

    # ── Raw data preview ──────────────────────────────────────────────────────
    with st.expander("Raw Data Preview", expanded=False):
        st.dataframe(df.tail(10), use_container_width=True)

        # Price chart
        fig_raw = go.Figure()
        fig_raw.add_trace(go.Scatter(
            x=df["Date"], y=df["Close"],
            mode="lines", name="Close Price",
            line=dict(color="#1565C0", width=1.5)
        ))
        fig_raw.add_trace(go.Bar(
            x=df["Date"], y=df["Volume"],
            name="Volume", yaxis="y2",
            marker_color="#E3F2FD", opacity=0.4
        ))
        fig_raw.update_layout(
            title=f"{ticker} Historical Price & Volume",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            yaxis2=dict(title="Volume", overlaying="y", side="right", showgrid=False),
            height=350,
            legend=dict(orientation="h", y=1.02)
        )
        st.plotly_chart(fig_raw, use_container_width=True)

    # ── Step 2: Preprocess ────────────────────────────────────────────────────
    with st.spinner("Preprocessing & building feature sequences..."):
        X_train, X_test, y_train, y_test, scaler, feature_cols = preprocess(df)
        st.info(
            f"Features: {len(feature_cols)} | "
            f"Train: {len(X_train):,} | Test: {len(X_test):,} samples",
            icon=":material/tune:",
        )

    # ── Step 3: Train ─────────────────────────────────────────────────────────
    progress_bar = st.progress(0, text="Training ANN model...")

    import tensorflow as tf
    from tensorflow.keras.callbacks import Callback

    class StreamlitProgress(Callback):
        """Custom Keras callback to update the Streamlit progress bar."""
        def __init__(self, total_epochs):
            super().__init__()
            self.total = total_epochs

        def on_epoch_end(self, epoch, logs=None):
            progress = (epoch + 1) / self.total
            loss = logs.get("val_loss", 0)
            progress_bar.progress(
                progress,
                text=f"Epoch {epoch+1}/{self.total} — val_loss: {loss:.6f}"
            )

    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_ann(input_shape=input_shape)

    callbacks = get_callbacks() + [StreamlitProgress(total_epochs=epochs)]

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=0
    )
    progress_bar.progress(1.0, text="Training complete!")

    # ── Step 4: Predict & Evaluate ────────────────────────────────────────────
    y_pred_scaled = model.predict(X_test, verbose=0).flatten()
    y_pred_real   = inverse_transform_close(scaler, y_pred_scaled, feature_cols)
    y_true_real   = inverse_transform_close(scaler, y_test,        feature_cols)

    mse  = mean_squared_error(y_true_real, y_pred_real)
    rmse = float(np.sqrt(mse))
    mae  = float(mean_absolute_error(y_true_real, y_pred_real))
    r2   = float(r2_score(y_true_real, y_pred_real))
    mape = float(np.mean(np.abs((y_true_real - y_pred_real) / (y_true_real + 1e-10))) * 100)

    # ── Metrics Row ───────────────────────────────────────────────────────────
    st.subheader("Performance Metrics")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("RMSE",  f"${rmse:.2f}",   help="Root Mean Squared Error — avg error in dollars")
    m2.metric("MAE",   f"${mae:.2f}",    help="Mean Absolute Error")
    m3.metric("R²",    f"{r2:.4f}",      help="1.0 = perfect fit")
    m4.metric("MAPE",  f"{mape:.2f}%",   help="Mean Absolute Percentage Error")
    m5.metric("Epochs",f"{len(history.history['loss'])}", help="Epochs trained (EarlyStopping may reduce this)")

    # ── Predictions Chart ─────────────────────────────────────────────────────
    n_test     = len(y_true_real)
    test_dates = pd.to_datetime(df["Date"]).iloc[-n_test:].reset_index(drop=True)

    st.subheader("Actual vs Predicted Prices")
    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(
        x=test_dates, y=y_true_real,
        mode="lines", name="Actual",
        line=dict(color="#1565C0", width=2)
    ))
    fig_pred.add_trace(go.Scatter(
        x=test_dates, y=y_pred_real,
        mode="lines", name="Predicted",
        line=dict(color="#E53935", width=1.8, dash="dash")
    ))
    fig_pred.update_layout(
        xaxis_title="Date", yaxis_title="Price (USD)",
        height=420,
        legend=dict(orientation="h", y=1.05),
        hovermode="x unified"
    )
    st.plotly_chart(fig_pred, use_container_width=True)

    # ── Training Loss Chart ───────────────────────────────────────────────────
    st.subheader("Training History")
    ep_range = list(range(1, len(history.history["loss"]) + 1))
    fig_hist = make_subplots(rows=1, cols=2, subplot_titles=["Loss (MSE)", "MAE"])
    fig_hist.add_trace(go.Scatter(x=ep_range, y=history.history["loss"],
                                  name="Train Loss", line=dict(color="#2196F3")), row=1, col=1)
    fig_hist.add_trace(go.Scatter(x=ep_range, y=history.history["val_loss"],
                                  name="Val Loss",   line=dict(color="#FF5722", dash="dash")), row=1, col=1)
    fig_hist.add_trace(go.Scatter(x=ep_range, y=history.history["mae"],
                                  name="Train MAE",  line=dict(color="#4CAF50")), row=1, col=2)
    fig_hist.add_trace(go.Scatter(x=ep_range, y=history.history["val_mae"],
                                  name="Val MAE",    line=dict(color="#FF9800", dash="dash")), row=1, col=2)
    fig_hist.update_layout(height=320, showlegend=True)
    st.plotly_chart(fig_hist, use_container_width=True)

    # ── Download Predictions CSV ──────────────────────────────────────────────
    results_df = pd.DataFrame({
        "Date":      test_dates.values,
        "Actual":    y_true_real.round(2),
        "Predicted": y_pred_real.round(2),
        "Error":     (y_pred_real - y_true_real).round(2),
        "Error_%":   ((y_pred_real - y_true_real) / y_true_real * 100).round(2)
    })

    st.subheader("Download Results")
    csv = results_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Predictions (CSV)",
        data=csv,
        file_name=f"{ticker}_predictions.csv",
        mime="text/csv",
        use_container_width=True
    )
    st.dataframe(results_df.tail(15), use_container_width=True)

else:
    # ── Landing state (before running) ───────────────────────────────────────
    st.info(
        "Configure your settings in the sidebar and click Run Prediction to start.",
        icon=":material/swipe_left:",
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"### {icon('download')} Data Collection", unsafe_allow_html=True)
        st.markdown("Downloads real stock data from Yahoo Finance using `yfinance`")
    with col2:
        st.markdown(f"### {icon('tune')} Preprocessing", unsafe_allow_html=True)
        st.markdown("Adds technical indicators (RSI, MACD, Bollinger Bands) and normalizes features")
    with col3:
        st.markdown(f"### {icon('neurology')} ANN Model", unsafe_allow_html=True)
        st.markdown("Trains a deep neural network with Dense layers, BatchNorm, and Dropout")

    st.markdown("---")
    st.markdown("**Supported tickers**: Any Yahoo Finance symbol — `AAPL`, `TSLA`, `GOOGL`, `MSFT`, `INFY.NS`, `TCS.NS`, etc.")
