# =============================================================================
# main.py — Project Entry Point
# =============================================================================
# Run this file to execute the complete pipeline:
#   1. Download stock data
#   2. Preprocess & feature engineer
#   3. Build ANN model
#   4. Train with callbacks
#   5. Evaluate & visualize
#
# Usage:
#   python main.py
# =============================================================================

import time
import numpy as np
import pandas as pd

from config import (
    TICKER, START_DATE, END_DATE,
    SEQUENCE_LENGTH, TEST_SPLIT
)
from src.data_loader   import download_stock_data
from src.preprocessor  import preprocess
from src.train         import train_model, plot_training_history
from src.evaluate      import evaluate


def main():
    start_time = time.time()

    print("\n" + "█"*55)
    print("█" + " "*53 + "█")
    print("█    🧠  ANN STOCK PRICE PREDICTOR              █")
    print("█    Built for: Computer Engineering Project    █")
    print("█" + " "*53 + "█")
    print("█"*55)

    # =========================================================================
    # STEP 1: Data Collection
    # =========================================================================
    print("\n\n📥 STEP 1: Downloading Stock Data")
    print("-"*40)
    df = download_stock_data(ticker=TICKER, start=START_DATE, end=END_DATE)

    # =========================================================================
    # STEP 2: Preprocessing
    # =========================================================================
    print("\n\n🔧 STEP 2: Preprocessing & Feature Engineering")
    print("-"*40)
    X_train, X_test, y_train, y_test, scaler, feature_cols = preprocess(df)

    # Prepare test dates for plot x-axis
    # The test set starts at (total - test_split) percentage, offset by SEQUENCE_LENGTH
    n_test   = X_test.shape[0]
    all_dates = pd.to_datetime(df["Date"]) if "Date" in df.columns else None
    if all_dates is not None:
        test_dates = all_dates.iloc[-n_test:].reset_index(drop=True)
    else:
        test_dates = None

    # =========================================================================
    # STEP 3 & 4: Build & Train Model
    # =========================================================================
    print("\n\n🚀 STEP 3 & 4: Building & Training ANN")
    print("-"*40)
    model, history = train_model(X_train, y_train, X_test, y_test)

    # Plot training loss curves
    plot_training_history(history, save=True)

    # =========================================================================
    # STEP 5: Evaluate & Visualize
    # =========================================================================
    print("\n\n📊 STEP 5: Evaluation & Visualization")
    print("-"*40)
    metrics = evaluate(
        model       = model,
        X_test      = X_test,
        y_test      = y_test,
        scaler      = scaler,
        feature_cols= feature_cols,
        ticker      = TICKER,
        dates       = test_dates
    )

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    elapsed = time.time() - start_time
    print("\n\n" + "="*55)
    print("  🎉 PIPELINE COMPLETE")
    print("="*55)
    print(f"  Ticker       : {TICKER}")
    print(f"  Data period  : {START_DATE} → {END_DATE}")
    print(f"  Features     : {len(feature_cols)}  ({', '.join(feature_cols[:5])}...)")
    print(f"  Sequence len : {SEQUENCE_LENGTH} days")
    print(f"  Test split   : {int(TEST_SPLIT*100)}%")
    print(f"\n  📈 Final Metrics:")
    print(f"     RMSE   = ${metrics['RMSE']:.2f}")
    print(f"     MAE    = ${metrics['MAE']:.2f}")
    print(f"     R²     = {metrics['R²']:.4f}")
    print(f"     MAPE   = {metrics['MAPE']:.2f}%")
    print(f"\n  ⏱  Total time  : {elapsed:.1f} seconds")
    print(f"\n  📁 Saved files :")
    print(f"     models/ann_model.h5")
    print(f"     models/scaler.pkl")
    print(f"     data/predictions_plot.png")
    print(f"     data/training_history.png")
    print("="*55 + "\n")


if __name__ == "__main__":
    main()
