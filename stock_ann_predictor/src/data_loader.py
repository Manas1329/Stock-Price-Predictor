# =============================================================================
# src/data_loader.py — Data Collection Module
# =============================================================================
# Responsible for: downloading stock data from Yahoo Finance, validating it,
# and saving it as a CSV to data/raw/. This keeps all I/O logic in one place.
# =============================================================================

import os
import yfinance as yf
import pandas as pd
from config import TICKER, START_DATE, END_DATE, RAW_DATA_PATH
import time
import requests
from requests import Session

session = Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
})

def download_stock_data(
    ticker: str = TICKER,
    start: str  = START_DATE,
    end: str    = END_DATE,
    save_path: str = RAW_DATA_PATH
) -> pd.DataFrame:
    """
    Downloads historical OHLCV stock data from Yahoo Finance.

    Parameters:
        ticker    : Stock symbol, e.g. "AAPL"
        start     : Start date string "YYYY-MM-DD"
        end       : End date string  "YYYY-MM-DD"
        save_path : Where to save the raw CSV

    Returns:
        df : pandas DataFrame with columns [Open, High, Low, Close, Volume]
    """

    print(f"\n{'='*55}")
    print(f"  📥 Downloading {ticker} stock data")
    print(f"  📅 Period : {start}  →  {end}")
    print(f"{'='*55}")

    # ── Download from Yahoo Finance ──────────────────────────────────────────
    # yf.download returns a DataFrame indexed by Date with OHLCV columns
    time.sleep(2)
    df = yf.download(ticker, start=start, end=end, progress=True)

    # ── Validate the download ────────────────────────────────────────────────
    if df.empty:
        raise ValueError(
            f"❌ No data returned for ticker '{ticker}'. "
            "Check the symbol and date range."
        )

    # ── Flatten MultiIndex columns if present (yfinance quirk) ──────────────
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # ── Reset index so 'Date' becomes a regular column ───────────────────────
    df.reset_index(inplace=True)

    # ── Drop rows with any missing values ────────────────────────────────────
    original_len = len(df)
    df.dropna(inplace=True)
    dropped = original_len - len(df)
    if dropped > 0:
        print(f"  ⚠️  Dropped {dropped} rows with missing values.")

    # ── Save to CSV ──────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n  ✅ Downloaded {len(df)} trading days of data")
    print(f"  💾 Saved to  : {save_path}")
    print(f"\n  Columns : {list(df.columns)}")
    print(f"  Date range : {df['Date'].iloc[0]}  →  {df['Date'].iloc[-1]}")
    print(f"\n  Sample (first 3 rows):\n{df.head(3).to_string(index=False)}")

    return df


def load_raw_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Loads previously saved raw CSV data from disk.

    Parameters:
        path : Path to the raw CSV file

    Returns:
        df : pandas DataFrame
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"❌ Raw data not found at '{path}'.\n"
            "   Run download_stock_data() first."
        )

    df = pd.read_csv(path, parse_dates=["Date"])
    print(f"✅ Loaded raw data from '{path}'  —  {len(df)} rows")
    return df


# ── Quick test when run directly ─────────────────────────────────────────────
if __name__ == "__main__":
    df = download_stock_data()
    print("\n📊 Data Info:")
    print(df.info())
    print("\n📈 Basic Statistics:")
    print(df.describe().round(2))
