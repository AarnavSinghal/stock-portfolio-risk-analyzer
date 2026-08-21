"""Train a separate LSTM per ticker — each model only sees that ticker's own history."""
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
MODEL_DIR = DATA_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TICKERS = ["AAPL", "AMZN", "GOOGL", "MSFT"]


def load_ticker_data(ticker: str):
    X = np.load(DATA_DIR / f"lstm_X_{ticker}.npy")
    y = np.load(DATA_DIR / f"lstm_y_{ticker}.npy")
    return X, y


def chronological_split(X, y, test_frac: float = 0.15):
    n = len(X)
    split_idx = int(n * (1 - test_frac))
    return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]


def build_model(seq_shape):
    model = models.Sequential([
        layers.Input(shape=seq_shape),
        layers.LSTM(32, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(16),
        layers.Dropout(0.2),
        layers.Dense(8, activation="relu"),
        layers.Dense(1, activation="linear"),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def train_ticker(ticker: str):
    print(f"\n{'='*50}\nTraining model for {ticker}\n{'='*50}")

    X, y = load_ticker_data(ticker)
    X_train, X_test, y_train, y_test = chronological_split(X, y)
    print(f"Train: {X_train.shape[0]} sequences, Test: {X_test.shape[0]} sequences")

    model = build_model(seq_shape=X_train.shape[1:])

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=8, restore_best_weights=True
    )

    model.fit(
        X_train, y_train,
        validation_split=0.15,
        epochs=80,
        batch_size=32,
        callbacks=[early_stop],
        verbose=0,  # quiet per-ticker to keep output readable across 4 runs
    )

    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"{ticker} — Test MSE: {test_loss:.6f}, Test MAE: {test_mae:.6f}")

    model_path = MODEL_DIR / f"volatility_lstm_{ticker}.keras"
    model.save(model_path)
    print(f"Saved to {model_path}")

    return test_mae


def main():
    results = {}
    for ticker in TICKERS:
        results[ticker] = train_ticker(ticker)

    print(f"\n{'='*50}\nSummary (Test MAE per ticker)\n{'='*50}")
    for ticker, mae in results.items():
        print(f"{ticker}: {mae:.6f}")


if __name__ == "__main__":
    main()