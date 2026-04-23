# =============================================================================
# src/model.py — ANN Architecture Definition
# =============================================================================
# Responsible for: defining the neural network layers, compiling with
# optimizer & loss function, and printing a human-readable summary.
#
# ANN ARCHITECTURE OVERVIEW:
#
#   Input Layer   → Flattens the (60 days × N features) matrix into 1D
#   Hidden Layer 1→ 128 neurons + ReLU + BatchNorm + Dropout
#   Hidden Layer 2→  64 neurons + ReLU + BatchNorm + Dropout
#   Hidden Layer 3→  32 neurons + ReLU + BatchNorm + Dropout
#   Output Layer  →   1 neuron  (predicts the next day's scaled Close price)
#
# WHY THESE CHOICES:
#   - ReLU activation : avoids vanishing gradient problem
#   - BatchNorm       : stabilizes and speeds up training
#   - Dropout         : randomly disables neurons to prevent overfitting
#   - Adam optimizer  : adaptive learning rate — works well for most problems
#   - MSE loss        : mean squared error, standard for regression tasks
# =============================================================================

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense, Dropout, BatchNormalization, Flatten, Input
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)
from config import HIDDEN_LAYERS, DROPOUT_RATE, LEARNING_RATE, MODEL_SAVE_PATH


def build_ann(input_shape: tuple) -> tf.keras.Model:
    """
    Builds and compiles the ANN model.

    Parameters:
        input_shape : Shape of ONE input sample, e.g. (60, 14)
                      = 60 days × 14 features (no batch dimension)

    Returns:
        model : Compiled Keras Sequential model
    """

    model = Sequential(name="Stock_ANN_Predictor")

    # ── Input Layer ───────────────────────────────────────────────────────────
    # Keras needs to know the shape of one input sample.
    # Our input is 2D: (SEQUENCE_LENGTH, n_features) = e.g. (60, 14)
    model.add(Input(shape=input_shape))

    # ── Flatten Layer ─────────────────────────────────────────────────────────
    # Dense layers expect 1D input. Flatten converts (60, 14) → (840,)
    # This is what makes it an ANN vs an LSTM — we treat time steps as flat features.
    model.add(Flatten())

    # ── Hidden Layers ─────────────────────────────────────────────────────────
    # We loop through HIDDEN_LAYERS = [128, 64, 32] to create 3 dense layers.
    # Adding more layers lets the network learn increasingly abstract patterns.
    for i, units in enumerate(HIDDEN_LAYERS):

        # Dense layer: every input neuron connects to every neuron in this layer
        model.add(Dense(
            units=units,
            activation="relu",   # ReLU: f(x) = max(0, x) — efficient & non-linear
            name=f"hidden_{i+1}"
        ))

        # BatchNormalization: normalizes each layer's output during training.
        # This helps training converge faster and reduces sensitivity to init.
        model.add(BatchNormalization(name=f"bn_{i+1}"))

        # Dropout: randomly sets DROPOUT_RATE fraction of neurons to 0 each batch.
        # This forces the network to learn redundant representations → less overfitting.
        model.add(Dropout(rate=DROPOUT_RATE, name=f"dropout_{i+1}"))

    # ── Output Layer ──────────────────────────────────────────────────────────
    # Single neuron with NO activation (linear) because we're doing regression.
    # We want the raw predicted number, not a probability.
    model.add(Dense(units=1, activation="linear", name="output"))

    # ── Compile ───────────────────────────────────────────────────────────────
    # Adam: adaptive learning rate optimizer — adjusts learning rate per parameter.
    # MSE (Mean Squared Error): penalizes large errors more heavily than MAE.
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="mean_squared_error",
        metrics=["mae"]   # Also track Mean Absolute Error for human readability
    )

    return model


def get_callbacks() -> list:
    """
    Returns a list of Keras callbacks for smarter training:

    1. EarlyStopping     — Stops training if val_loss doesn't improve for N epochs.
                           Prevents wasting time and prevents overfitting.

    2. ReduceLROnPlateau — Cuts learning rate by 50% when val_loss stagnates.
                           Helps the model fine-tune near the optimal solution.

    3. ModelCheckpoint   — Saves the BEST model (lowest val_loss) automatically.
                           Even if later epochs overfit, we keep the best version.
    """

    early_stop = EarlyStopping(
        monitor="val_loss",    # Watch validation loss
        patience=10,           # Stop after 10 epochs of no improvement
        restore_best_weights=True,  # Roll back to the best weights
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,            # Multiply LR by 0.5 when triggered
        patience=5,            # Wait 5 epochs before reducing
        min_lr=1e-7,           # Don't go below this learning rate
        verbose=1
    )

    checkpoint = ModelCheckpoint(
        filepath=MODEL_SAVE_PATH,
        monitor="val_loss",
        save_best_only=True,   # Only save when val_loss improves
        verbose=1
    )

    return [early_stop, reduce_lr, checkpoint]


def print_model_summary(model: tf.keras.Model) -> None:
    """Pretty-prints the model architecture with parameter counts."""
    print("\n" + "="*55)
    print("  🧠 ANN Model Architecture")
    print("="*55)
    model.summary()
    total_params = model.count_params()
    print(f"\n  Total trainable parameters: {total_params:,}")
    print("="*55 + "\n")


# ── Quick test when run directly ─────────────────────────────────────────────
if __name__ == "__main__":
    from config import SEQUENCE_LENGTH, USE_TECHNICAL_INDICATORS

    # Approximate feature count
    n_features = 14 if USE_TECHNICAL_INDICATORS else 5
    model = build_ann(input_shape=(SEQUENCE_LENGTH, n_features))
    print_model_summary(model)
