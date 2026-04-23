# =============================================================================
# config.py — Central Configuration File
# =============================================================================
# All global settings live here. Change values in this file to experiment
# with different stocks, date ranges, or model parameters without touching
# any other file in the project.
# =============================================================================

# ── Stock Settings ────────────────────────────────────────────────────────────
TICKER     = "AAPL"           # Stock ticker symbol (Apple Inc.)
                              # Other examples: "TSLA", "GOOGL", "MSFT", "INFY.NS"
START_DATE = "2018-01-01"     # Start of historical data range
END_DATE   = "2024-01-01"     # End of historical data range

# ── Data Settings ─────────────────────────────────────────────────────────────
TARGET_COLUMN   = "Close"     # Column we want to predict (Closing price)
SEQUENCE_LENGTH = 60          # Number of past days used to predict the next day
                              # e.g., use last 60 trading days → predict day 61
TEST_SPLIT      = 0.2         # 20% of data reserved for testing (80% for training)

# ── Feature Engineering ───────────────────────────────────────────────────────
USE_TECHNICAL_INDICATORS = True   # Add RSI, MACD, Moving Averages as features

# ── ANN Model Hyperparameters ─────────────────────────────────────────────────
EPOCHS          = 50          # Training iterations over the full dataset
BATCH_SIZE      = 32          # Samples per gradient update (powers of 2 work best)
LEARNING_RATE   = 0.001       # How fast the model learns (Adam optimizer default)
DROPOUT_RATE    = 0.2         # Fraction of neurons to randomly drop (prevents overfitting)

# ── Layer Architecture ────────────────────────────────────────────────────────
# List of neuron counts for each hidden Dense layer
HIDDEN_LAYERS   = [128, 64, 32]   # 3 hidden layers with decreasing neurons

# ── File Paths ────────────────────────────────────────────────────────────────
RAW_DATA_PATH       = "data/raw/stock_data.csv"
PROCESSED_DATA_PATH = "data/processed/processed_data.csv"
SCALER_SAVE_PATH    = "models/scaler.pkl"
MODEL_SAVE_PATH     = "models/ann_model.h5"
PLOT_SAVE_PATH      = "data/predictions_plot.png"
