"""Train an LSTM to forecast next-day return volatility from prepared sequences."""
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
MODEL_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    X = np.load(DATA_DIR / "lstm_X.npy")
    y = np.load(DATA_DIR / "lstm_y.npy")
    return X, y


def train_test_split_chronological(X, y, test_frac: float = 0.15):
    """Split by time (not randomly) since this is a time series problem."""
    n = len(X)
    split_idx = int(n * (1 - test_frac))
    return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]


def build_model(input_shape):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(32, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(16),
        layers.Dropout(0.2),
        layers.Dense(8, activation="relu"),
        layers.Dense(1, activation="linear"),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def main():
    X, y = load_data()
    print(f"Loaded X={X.shape}, y={y.shape}")

    X_train, X_test, y_train, y_test = train_test_split_chronological(X, y)
    print(f"Train: {X_train.shape[0]} sequences, Test: {X_test.shape[0]} sequences")

    model = build_model(input_shape=X_train.shape[1:])
    model.summary()

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        validation_split=0.15,
        epochs=50,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1,
    )

    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest MSE: {test_loss:.6f}")
    print(f"Test MAE: {test_mae:.6f}")

    model.save(MODEL_DIR / "volatility_lstm.keras")
    print(f"Model saved to {MODEL_DIR / 'volatility_lstm.keras'}")


if __name__ == "__main__":
    main()