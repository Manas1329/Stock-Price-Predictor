# =============================================================================
# src/evaluate.py — Evaluation & Visualization Module
# =============================================================================
# Responsible for: generating predictions on test data, computing performance
# metrics, and creating publication-quality plots.
#
# METRICS EXPLAINED:
#   MSE  (Mean Squared Error)      : Average of squared errors. Penalizes large
#                                    errors heavily. Unit: (dollars²)
#   RMSE (Root Mean Squared Error) : Square root of MSE. Same unit as price ($).
#                                    "On average, predictions are off by $X"
#   MAE  (Mean Absolute Error)     : Average absolute error. Less sensitive to
#                                    outliers than RMSE. Unit: dollars ($)
#   R²   (R-squared / Coefficient of Determination): How much variance in actual
#                                    prices is explained by our model.
#                                    1.0 = perfect  |  0.0 = no better than mean
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from config import PLOT_SAVE_PATH
from src.preprocessor import inverse_transform_close


def make_predictions(model, X_test: np.ndarray) -> np.ndarray:
    """
    Runs the model on test input to produce scaled predictions.

    Parameters:
        model  : Trained Keras model
        X_test : Test sequences shape (samples, 60, n_features)

    Returns:
        y_pred : 1D array of scaled predictions
    """
    print("\n  🔮 Generating predictions on test set...")
    y_pred_scaled = model.predict(X_test, verbose=0)   # shape: (samples, 1)
    y_pred_scaled = y_pred_scaled.flatten()             # shape: (samples,)
    print(f"     Generated {len(y_pred_scaled)} predictions ✓")
    return y_pred_scaled


def compute_metrics(
    y_true_real: np.ndarray,
    y_pred_real: np.ndarray
) -> dict:
    """
    Computes regression performance metrics on real (un-scaled) price values.

    Parameters:
        y_true_real : Actual stock prices in dollars
        y_pred_real : Predicted stock prices in dollars

    Returns:
        metrics : dict with MSE, RMSE, MAE, R², MAPE
    """

    mse  = mean_squared_error(y_true_real, y_pred_real)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_true_real, y_pred_real)
    r2   = r2_score(y_true_real, y_pred_real)

    # MAPE: Mean Absolute Percentage Error — intuitive % measure
    mape = np.mean(np.abs((y_true_real - y_pred_real) / (y_true_real + 1e-10))) * 100

    metrics = {
        "MSE"  : round(mse,  4),
        "RMSE" : round(rmse, 4),
        "MAE"  : round(mae,  4),
        "R²"   : round(r2,   4),
        "MAPE" : round(mape, 4)
    }

    return metrics


def print_metrics(metrics: dict, ticker: str = "Stock") -> None:
    """Prints a formatted performance metrics table."""
    print("\n" + "="*45)
    print(f"  📊 Performance Metrics — {ticker}")
    print("="*45)
    print(f"  {'Metric':<8} {'Value':>12}")
    print("-"*45)
    print(f"  {'MSE':<8} {metrics['MSE']:>12.4f}  (lower = better)")
    print(f"  {'RMSE':<8} ${metrics['RMSE']:>11.2f}  (avg error in $)")
    print(f"  {'MAE':<8} ${metrics['MAE']:>11.2f}  (avg absolute error)")
    print(f"  {'R²':<8} {metrics['R²']:>12.4f}  (closer to 1 = better)")
    print(f"  {'MAPE':<8} {metrics['MAPE']:>11.2f}%  (% error)")
    print("="*45)


def plot_predictions(
    y_true_real: np.ndarray,
    y_pred_real: np.ndarray,
    ticker: str = "AAPL",
    metrics: dict = None,
    dates: pd.Series = None,
    save_path: str = PLOT_SAVE_PATH
) -> None:
    """
    Creates a comprehensive multi-panel visualization:
      Panel 1: Predicted vs Actual prices (main comparison)
      Panel 2: Prediction error over time
      Panel 3: Scatter plot (perfect = diagonal line)
      Panel 4: Error distribution histogram
    """

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(
        f"ANN Stock Price Prediction — {ticker}",
        fontsize=16, fontweight="bold", y=0.98
    )

    x_axis = dates if dates is not None else np.arange(len(y_true_real))

    # ── Panel 1: Predicted vs Actual ─────────────────────────────────────────
    ax1 = fig.add_subplot(2, 2, (1, 2))   # Spans top row

    ax1.plot(x_axis, y_true_real,
             label="Actual Price",    color="#1565C0", linewidth=1.8, alpha=0.9)
    ax1.plot(x_axis, y_pred_real,
             label="Predicted Price", color="#E53935", linewidth=1.5,
             linestyle="--", alpha=0.85)

    # Shaded area between actual and predicted (shows where errors are large)
    ax1.fill_between(x_axis, y_true_real, y_pred_real,
                     alpha=0.1, color="#FF9800", label="Prediction Error")

    ax1.set_title("Actual vs Predicted Stock Price", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Date" if dates is not None else "Test Sample")
    ax1.set_ylabel("Price (USD)")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    if dates is not None:
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax1.get_xticklabels(), rotation=30, ha="right")

    # Add metrics annotation box
    if metrics:
        annotation = (
            f"RMSE: ${metrics['RMSE']:.2f}\n"
            f"MAE:  ${metrics['MAE']:.2f}\n"
            f"R²:   {metrics['R²']:.4f}\n"
            f"MAPE: {metrics['MAPE']:.2f}%"
        )
        ax1.annotate(
            annotation,
            xy=(0.02, 0.97), xycoords="axes fraction",
            fontsize=9, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                      edgecolor="gray", alpha=0.8)
        )

    # ── Panel 2: Error over time ──────────────────────────────────────────────
    ax2 = fig.add_subplot(2, 2, 3)

    errors = y_pred_real - y_true_real
    colors = ["#E53935" if e > 0 else "#1565C0" for e in errors]

    ax2.bar(range(len(errors)), errors, color=colors, alpha=0.6, width=1.0)
    ax2.axhline(y=0, color="black", linewidth=1.2)
    ax2.set_title("Prediction Error Over Time", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Test Sample")
    ax2.set_ylabel("Error (Predicted − Actual) $")
    ax2.grid(True, alpha=0.3)

    red_patch   = plt.Rectangle((0,0),1,1, color="#E53935", alpha=0.6, label="Over-predicted")
    blue_patch  = plt.Rectangle((0,0),1,1, color="#1565C0", alpha=0.6, label="Under-predicted")
    ax2.legend(handles=[red_patch, blue_patch], fontsize=8)

    # ── Panel 3: Scatter (Actual vs Predicted) ────────────────────────────────
    ax3 = fig.add_subplot(2, 2, 4)

    ax3.scatter(y_true_real, y_pred_real,
                alpha=0.4, color="#7B1FA2", s=15, label="Predictions")

    # Perfect prediction line (y = x)
    min_val = min(y_true_real.min(), y_pred_real.min())
    max_val = max(y_true_real.max(), y_pred_real.max())
    ax3.plot([min_val, max_val], [min_val, max_val],
             color="red", linewidth=2, linestyle="--", label="Perfect Prediction")

    ax3.set_title("Actual vs Predicted (Scatter)", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Actual Price ($)")
    ax3.set_ylabel("Predicted Price ($)")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    if metrics:
        ax3.text(0.05, 0.95, f"R² = {metrics['R²']:.4f}",
                 transform=ax3.transAxes, fontsize=10,
                 verticalalignment="top",
                 bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save
    import os
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n  📊 Prediction plot saved → {save_path}")
    plt.close()


def evaluate(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    scaler,
    feature_cols: list,
    ticker: str = "AAPL",
    dates: pd.Series = None
) -> dict:
    """
    Full evaluation pipeline:
      1. Generate predictions
      2. Inverse-transform to real dollar values
      3. Compute metrics
      4. Print metrics table
      5. Generate and save plot

    Returns:
        metrics dict
    """

    # Step 1: Predict (scaled values)
    y_pred_scaled = make_predictions(model, X_test)

    # Step 2: Convert back to real dollar prices
    y_pred_real = inverse_transform_close(scaler, y_pred_scaled, feature_cols)
    y_true_real = inverse_transform_close(scaler, y_test,        feature_cols)

    # Step 3 & 4: Metrics
    metrics = compute_metrics(y_true_real, y_pred_real)
    print_metrics(metrics, ticker)

    # Step 5: Plot
    plot_predictions(
        y_true_real=y_true_real,
        y_pred_real=y_pred_real,
        ticker=ticker,
        metrics=metrics,
        dates=dates
    )

    return metrics
