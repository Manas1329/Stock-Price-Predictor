# 📈 Stock Price Prediction using Artificial Neural Network (ANN)

> **Computer Engineering Final Year Project**
> Built with Python · TensorFlow/Keras · Streamlit

---

## 🧠 Overview

This project predicts future stock closing prices using a multi-layer
Artificial Neural Network (ANN). It ingests historical OHLCV data from
Yahoo Finance, engineers technical indicators (RSI, MACD, Bollinger Bands,
Moving Averages), trains a deep ANN, and visualizes predicted vs actual prices.

---

## 📁 Folder Structure

```
stock_ann_predictor/
├── data/
│   ├── raw/                    ← Downloaded CSV from Yahoo Finance
│   └── processed/              ← Normalized feature data
├── models/
│   ├── ann_model.h5            ← Saved trained model
│   └── scaler.pkl              ← Saved MinMaxScaler
├── notebooks/
│   └── 01_exploration.ipynb   ← EDA notebook
├── src/
│   ├── __init__.py
│   ├── data_loader.py          ← yfinance download logic
│   ├── preprocessor.py         ← Normalization + sequence creation
│   ├── model.py                ← ANN architecture
│   ├── train.py                ← Training loop + callbacks
│   └── evaluate.py             ← Metrics + visualization
├── app.py                      ← Streamlit web UI
├── main.py                     ← CLI entry point
├── config.py                   ← All settings (ticker, epochs, etc.)
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

```bash
# 1. Clone / download the project
cd stock_ann_predictor

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate       # Mac/Linux
venv\Scripts\activate          # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Running the Project

### Option A — Command Line (full pipeline)
```bash
python main.py
```

### Option B — Streamlit Web App
```bash
streamlit run app.py
```

---

## 🔧 Configuration

Edit `config.py` to change:

| Setting | Default | Description |
|--------|---------|-------------|
| `TICKER` | `"AAPL"` | Stock symbol |
| `START_DATE` | `"2018-01-01"` | Data start |
| `END_DATE` | `"2024-01-01"` | Data end |
| `SEQUENCE_LENGTH` | `60` | Look-back window (days) |
| `EPOCHS` | `50` | Training iterations |
| `HIDDEN_LAYERS` | `[128, 64, 32]` | ANN layer sizes |

---

## 🧠 ANN Architecture

```
Input (60 days × N features)
    ↓
Flatten → (60 × N,)
    ↓
Dense(128) → BatchNorm → Dropout(0.2)
    ↓
Dense(64)  → BatchNorm → Dropout(0.2)
    ↓
Dense(32)  → BatchNorm → Dropout(0.2)
    ↓
Dense(1)  [Linear output — regression]
```

---

## 📊 Features Used

**Raw OHLCV**: Open, High, Low, Close, Volume

**Technical Indicators**:
- SMA 10, 20, 50 (Simple Moving Averages)
- RSI 14 (Relative Strength Index)
- MACD + Signal + Histogram
- Bollinger Bands (Upper, Lower, Width)

---

## 📈 Output

- `data/predictions_plot.png` — 4-panel prediction visualization
- `data/training_history.png` — Loss & MAE curves
- `models/ann_model.h5`       — Saved model weights
- Metrics: RMSE, MAE, R², MAPE

---

## 🛠 Tech Stack

| Library | Purpose |
|---------|---------|
| `yfinance` | Stock data download |
| `pandas / numpy` | Data manipulation |
| `scikit-learn` | Normalization, metrics |
| `TensorFlow/Keras` | ANN model |
| `matplotlib` | Visualizations |
| `streamlit` | Web UI |
| `plotly` | Interactive charts |

---

*Made with ❤️ for Computer Engineering Project*
