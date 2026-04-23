# =============================================================================
# src/train.py — Model Training Module
# =============================================================================
# Responsible for: fitting the ANN on training data, tracking training history,
# and generating a training loss curve plot.
#
# WHAT HAPPENS DURING TRAINING:
#   For each epoch:
#     1. Feed batches of (X_train, y_train) through the network
#     2. Compute prediction error (loss = MSE)
#     3. Backpropagate: adjust weights to reduce the error
#     4. Evaluate on validation set (X_test, y_test) — no weight updates here
#   Repeat until EarlyStopping triggers or EPOCHS is reached.
# =============================================================================

import os
import matplotlib
matplotlib.use("Agg")   # Use non-interactive backend (safe for scripts)
import matplotlib.pyplot as plt
import tensorflow as tf
from config import EPOCHS, BATCH_SIZE, MODEL_SAVE_PATH
from src.model import build_ann, get_callbacks, print_model_summary


def train_model(
    X_train, y_train,
    X_test,  y_test
) -> tuple:
    """
    Builds, compiles, and trains the ANN model.

    Parameters:
        X_train : Training input  shape (samples, 60, n_features)
        y_train : Training labels shape (samples,)
        X_test  : Validation input
        y_test  : Validation labels

    Returns:
        model   : Trained Keras model (best weights from checkpoint)
        history : Keras History object (contains loss/mae per epoch)
    """

    print("\n" + "="*55)
    print("  🚀 Starting Model Training")
    print("="*55)

    # ── Derive input shape from data ──────────────────────────────────────────
    # X_train.shape = (n_samples, sequence_length, n_features)
    # input_shape = (sequence_length, n_features) — one sample, no batch dim
    input_shape = (X_train.shape[1], X_train.shape[2])
    print(f"\n  Input shape  : {input_shape}")
    print(f"  Train samples: {len(X_train):,}")
    print(f"  Test  samples: {len(X_test):,}")
    print(f"  Epochs       : {EPOCHS}")
    print(f"  Batch size   : {BATCH_SIZE}")

    # ── Build the ANN ─────────────────────────────────────────────────────────
    model = build_ann(input_shape=input_shape)
    print_model_summary(model)

    # ── Ensure models directory exists ────────────────────────────────────────
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

    # ── Train ─────────────────────────────────────────────────────────────────
    print("\n  📊 Training in progress...\n")
    history = model.fit(
        X_train, y_train,
        epochs          = EPOCHS,
        batch_size      = BATCH_SIZE,
        validation_data = (X_test, y_test),   # Monitor generalization live
        callbacks       = get_callbacks(),    # EarlyStopping + ReduceLR + Checkpoint
        verbose         = 1
    )

    # ── Summarize training result ──────────────────────────────────────────────
    final_train_loss = history.history["loss"][-1]
    final_val_loss   = history.history["val_loss"][-1]
    epochs_ran       = len(history.history["loss"])

    print(f"\n  ✅ Training complete!")
    print(f"     Epochs ran       : {epochs_ran}")
    print(f"     Final train loss : {final_train_loss:.6f}")
    print(f"     Final val loss   : {final_val_loss:.6f}")
    print(f"     Best model saved : {MODEL_SAVE_PATH}")

    return model, history


def plot_training_history(history, save: bool = True) -> None:
    """
    Plots training vs validation loss curves over epochs.
    This helps diagnose:
      - Overfitting  : train loss ↓ but val loss ↑
      - Underfitting : both losses remain high
      - Good fit     : both losses decrease and converge
    """

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Model Training History", fontsize=14, fontweight="bold")

    epochs_range = range(1, len(history.history["loss"]) + 1)

    # ── Plot 1: Loss (MSE) ────────────────────────────────────────────────────
    axes[0].plot(epochs_range, history.history["loss"],
                 label="Train Loss", color="#2196F3", linewidth=2)
    axes[0].plot(epochs_range, history.history["val_loss"],
                 label="Val Loss",   color="#FF5722", linewidth=2, linestyle="--")
    axes[0].set_title("Loss (MSE)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Mean Squared Error")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # ── Plot 2: MAE ───────────────────────────────────────────────────────────
    axes[1].plot(epochs_range, history.history["mae"],
                 label="Train MAE", color="#4CAF50", linewidth=2)
    axes[1].plot(epochs_range, history.history["val_mae"],
                 label="Val MAE",   color="#FF9800", linewidth=2, linestyle="--")
    axes[1].set_title("Mean Absolute Error (MAE)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAE")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save:
        path = "data/training_history.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  📊 Training history plot saved → {path}")

    plt.close()


def load_trained_model(path: str = MODEL_SAVE_PATH) -> tf.keras.Model:
    """Loads a saved model from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ No saved model at '{path}'. Train first.")
    model = tf.keras.models.load_model(path)
    print(f"✅ Model loaded from '{path}'")
    return model
