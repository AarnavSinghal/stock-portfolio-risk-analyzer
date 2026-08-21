"""Train an LSTM (with ticker embeddings) to forecast next-day return volatility."""
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
MODEL_DIR = DATA_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

N_TICKERS = 4
EMBEDDING_DIM = 4


def load_data():
    X = np.load(DATA_DIR / "lstm_X.npy")
    ticker_ids = np.load(DATA_DIR / "lstm_ticker_ids.npy")
    y = np.load(DATA_DIR / "lstm_y.npy")
    return X, ticker_ids, y


def chronological_split(X, ticker_ids, y, test_frac: float = 0.15):
    n = len(X)
    split_idx = int(n * (1 - test_frac))
    return (
        X[:split_idx], X[split_idx:],
        ticker_ids[:split_idx], ticker_ids[split_idx:],
        y[:split_idx], y[split_idx:],
    )


def build_model(seq_shape):
    seq_input = layers.Input(shape=seq_shape, name="sequence_input")
    ticker_input = layers.Input(shape=(1,), name="ticker_input")

    ticker_embed = layers.Embedding(input_dim=N_TICKERS, output_dim=EMBEDDING_DIM)(ticker_input)
    ticker_embed = layers.Flatten()(ticker_embed)  # (batch, EMBEDDING_DIM)
    ticker_embed = layers.RepeatVector(seq_shape[0])(ticker_embed)  # repeat across timesteps

    ticker_embed_reshaped = layers.Reshape((seq_shape[0], EMBEDDING_DIM))(ticker_embed)
    merged = layers.Concatenate(axis=-1)([seq_input, ticker_embed_reshaped])

    x = layers.LSTM(32, return_sequences=True)(merged)
    x = layers.Dropout(0.2)(x)
    x = layers.LSTM(16)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(8, activation="relu")(x)
    output = layers.Dense(1, activation="linear")(x)

    model = models.Model(inputs=[seq_input, ticker_input], outputs=output)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def main():
    X, ticker_ids, y = load_data()
    print(f"Loaded X={X.shape}, ticker_ids={ticker_ids.shape}, y={y.shape}")

    X_train, X_test, tid_train, tid_test, y_train, y_test = chronological_split(X, ticker_ids, y)
    print(f"Train: {X_train.shape[0]} sequences, Test: {X_test.shape[0]} sequences")

    model = build_model(seq_shape=X_train.shape[1:])
    model.summary()

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=8, restore_best_weights=True
    )

    history = model.fit(
        [X_train, tid_train], y_train,
        validation_split=0.15,
        epochs=80,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1,
    )

    test_loss, test_mae = model.evaluate([X_test, tid_test], y_test, verbose=0)
    print(f"\nTest MSE: {test_loss:.6f}")
    print(f"Test MAE: {test_mae:.6f}")

    model.save(MODEL_DIR / "volatility_lstm.keras")
    print(f"Model saved to {MODEL_DIR / 'volatility_lstm.keras'}")


if __name__ == "__main__":
    main()