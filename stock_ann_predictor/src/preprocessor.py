# =============================================================================
# src/preprocessor.py — Data Preprocessing & Feature Engineering Module
# =============================================================================
# Responsible for: cleaning data, adding technical indicators, normalizing
# values with MinMaxScaler, and creating (X, y) sequence pairs for the ANN.
#
# WHY PREPROCESSING MATTERS:
#   Neural networks are sensitive to the scale of input values. If one feature
#   ranges from 0–1 and another from 0–10000, the network will bias toward the
#   larger one. Normalization (scaling to 0–1) fixes this.
# =============================================================================

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from config import (
    TARGET_COLUMN, SEQUENCE_LENGTH, TEST_SPLIT,
    USE_TECHNICAL_INDICATORS, PROCESSED_DATA_PATH, SCALER_SAVE_PATH
)


# ── Technical Indicator Helpers ───────────────────────────────────────────────

def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds Simple Moving Averages (SMA) for 10, 20, and 50 days.
    SMA smooths out short-term fluctuations to reveal longer-term trends.
    """
    df["SMA_10"] = df["Close"].rolling(window=10).mean()   # 2-week average
    df["SMA_20"] = df["Close"].rolling(window=20).mean()   # 1-month average
    df["SMA_50"] = df["Close"].rolling(window=50).mean()   # 2.5-month average
    return df


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Adds the Relative Strength Index (RSI) — a momentum indicator.

    RSI ranges from 0–100:
      - RSI > 70 → stock may be OVERBOUGHT (potential price drop)
      - RSI < 30 → stock may be OVERSOLD  (potential price rise)
    """
    delta = df["Close"].diff()              # Day-over-day price changes

    gain = delta.clip(lower=0)             # Keep only positive changes (gains)
    loss = -delta.clip(upper=0)            # Keep only negative changes (losses)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / (avg_loss + 1e-10)    # Relative Strength (add tiny value to avoid /0)
    df["RSI"] = 100 - (100 / (1 + rs))   # RSI formula

    return df


def add_macd(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds MACD (Moving Average Convergence Divergence).
    MACD = 12-day EMA − 26-day EMA
    Signal = 9-day EMA of MACD

    Crossovers between MACD and Signal line are classic buy/sell signals.
    EMA = Exponential Moving Average (recent prices weighted more heavily).
    """
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()  # Short-term EMA
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()  # Long-term EMA

    df["MACD"]        = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"]   = df["MACD"] - df["MACD_Signal"]   # Histogram = divergence

    return df


def add_bollinger_bands(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Adds Bollinger Bands — volatility indicator.
    Upper Band = SMA + 2 * StdDev
    Lower Band = SMA - 2 * StdDev

    Price touching upper band → potentially overbought.
    Price touching lower band → potentially oversold.
    """
    sma   = df["Close"].rolling(window=window).mean()
    std   = df["Close"].rolling(window=window).std()
    df["BB_Upper"] = sma + (2 * std)
    df["BB_Lower"] = sma - (2 * std)
    df["BB_Width"] = df["BB_Upper"] - df["BB_Lower"]   # Measures current volatility

    return df


# ── Main Preprocessing Pipeline ───────────────────────────────────────────────

def preprocess(df: pd.DataFrame) -> tuple:
    """
    Full preprocessing pipeline:
      1. Select & add features (technical indicators)
      2. Drop NaN rows created by rolling windows
      3. Normalize all features to [0, 1]
      4. Create (X, y) sequence pairs
      5. Train/test split

    Returns:
        X_train, X_test  : Input sequences   shape → (samples, SEQUENCE_LENGTH, features)
        y_train, y_test  : Target values     shape → (samples,)
        scaler           : Fitted MinMaxScaler (needed to inverse-transform predictions)
        feature_cols     : List of feature column names used
    """

    print("\n" + "="*55)
    print("  🔧 Preprocessing Pipeline Starting")
    print("="*55)

    df = df.copy()  # Don't modify the original DataFrame

    # ── Step 1: Add technical indicators ─────────────────────────────────────
    if USE_TECHNICAL_INDICATORS:
        print("\n  [1/5] Adding technical indicators...")
        df = add_moving_averages(df)
        df = add_rsi(df)
        df = add_macd(df)
        df = add_bollinger_bands(df)
        print("       SMA_10/20/50 ✓  RSI ✓  MACD ✓  Bollinger Bands ✓")

    # ── Step 2: Select feature columns ────────────────────────────────────────
    # We include OHLCV + all technical indicators we added
    base_cols       = ["Open", "High", "Low", "Close", "Volume"]
    indicator_cols  = ["SMA_10", "SMA_20", "SMA_50",
                       "RSI",
                       "MACD", "MACD_Signal", "MACD_Hist",
                       "BB_Upper", "BB_Lower", "BB_Width"] if USE_TECHNICAL_INDICATORS else []

    feature_cols = base_cols + [c for c in indicator_cols if c in df.columns]
    df = df[feature_cols].copy()

    # ── Step 3: Drop NaN rows (from rolling window calculations) ──────────────
    print(f"\n  [2/5] Dropping NaN rows from rolling windows...")
    before = len(df)
    df.dropna(inplace=True)
    print(f"       Rows before: {before}  →  after: {len(df)}")

    # ── Step 4: Normalize features to [0, 1] ──────────────────────────────────
    print(f"\n  [3/5] Normalizing {len(feature_cols)} features to [0, 1]...")
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)   # Fit ON training data only
    # shape: (n_days, n_features)

    # Save scaler so we can inverse-transform predictions later
    os.makedirs(os.path.dirname(SCALER_SAVE_PATH), exist_ok=True)
    with open(SCALER_SAVE_PATH, "wb") as f:
        pickle.dump(scaler, f)
    print(f"       Scaler saved to {SCALER_SAVE_PATH} ✓")

    # ── Step 5: Save processed data ───────────────────────────────────────────
    processed_df = pd.DataFrame(scaled_data, columns=feature_cols)
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    processed_df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"       Processed data saved to {PROCESSED_DATA_PATH} ✓")

    # ── Step 6: Create sequences (X, y pairs) ─────────────────────────────────
    print(f"\n  [4/5] Creating sequences (look-back = {SEQUENCE_LENGTH} days)...")
    #
    # HOW SEQUENCES WORK:
    #   Day 0..59  features  →  predict Day 60's Close
    #   Day 1..60  features  →  predict Day 61's Close
    #   Day 2..61  features  →  predict Day 62's Close
    #   ...
    # Each X[i] is a 2D matrix of shape (60, n_features)
    # Each y[i] is a scalar — the scaled Close price on day i+60

    close_col_idx = feature_cols.index(TARGET_COLUMN)  # Index of "Close" column

    X, y = [], []
    for i in range(SEQUENCE_LENGTH, len(scaled_data)):
        X.append(scaled_data[i - SEQUENCE_LENGTH : i, :])   # Window of 60 days
        y.append(scaled_data[i, close_col_idx])              # Next day's Close price

    X = np.array(X)   # shape: (samples, 60, n_features)
    y = np.array(y)   # shape: (samples,)

    print(f"       X shape: {X.shape}  |  y shape: {y.shape}")

    # ── Step 7: Train/Test Split ───────────────────────────────────────────────
    print(f"\n  [5/5] Splitting data  (train={int((1-TEST_SPLIT)*100)}% / test={int(TEST_SPLIT*100)}%)...")
    split_idx = int(len(X) * (1 - TEST_SPLIT))

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"       X_train: {X_train.shape}  |  X_test: {X_test.shape}")
    print(f"       y_train: {y_train.shape}  |  y_test: {y_test.shape}")
    print(f"\n  ✅ Preprocessing complete!")

    return X_train, X_test, y_train, y_test, scaler, feature_cols


def inverse_transform_close(scaler, values: np.ndarray, feature_cols: list) -> np.ndarray:
    """
    Inverse-transforms the scaled 'Close' predictions back to real price values.

    The scaler was fitted on ALL features, so we must reconstruct a full-width
    dummy matrix to use inverse_transform correctly.

    Parameters:
        scaler       : The fitted MinMaxScaler object
        values       : 1D array of scaled Close values
        feature_cols : List of all feature column names

    Returns:
        1D numpy array of real-world price values
    """
    close_idx = feature_cols.index(TARGET_COLUMN)
    n_features = len(feature_cols)

    # Create dummy matrix filled with zeros, same width as all features
    dummy = np.zeros((len(values), n_features))
    dummy[:, close_idx] = values.flatten()   # Place our predictions in the Close column

    # Inverse transform the entire dummy matrix, then extract Close column
    return scaler.inverse_transform(dummy)[:, close_idx]


# ── Quick test when run directly ─────────────────────────────────────────────
if __name__ == "__main__":
    from src.data_loader import load_raw_data
    df = load_raw_data()
    X_train, X_test, y_train, y_test, scaler, feature_cols = preprocess(df)
    print(f"\nFeatures used ({len(feature_cols)}): {feature_cols}")
