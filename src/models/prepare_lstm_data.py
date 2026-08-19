"""Prepare sequences for LSTM volatility forecasting from MySQL price data."""
import os
from pathlib import Path

import mysql.connector
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

SEQ_LEN = 20  # lookback window in trading days
ROLLING_VOL_WINDOW = 5


def get_connection():
    return mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
    )


def load_price_matrix(conn) -> pd.DataFrame:
    query = """
        SELECT t.symbol, dp.price_date, dp.close_price
        FROM daily_prices dp
        JOIN tickers t ON t.id = dp.ticker_id
        ORDER BY dp.price_date
    """
    df = pd.read_sql(query, conn)
    price_matrix = df.pivot(index="price_date", columns="symbol", values="close_price")
    price_matrix.index = pd.to_datetime(price_matrix.index)
    return price_matrix.astype(float)


def build_ticker_sequences(returns: pd.Series, vol: pd.Series, ticker_onehot: np.ndarray):
    """Build (X, y) sequences for a single ticker's return/vol series."""
    X, y = [], []
    values = np.column_stack([returns.values, vol.values])
    targets = returns.abs().shift(-1).values  # next-day |return|

    for i in range(SEQ_LEN, len(values) - 1):
        window = values[i - SEQ_LEN:i]
        window_with_id = np.hstack([window, np.tile(ticker_onehot, (SEQ_LEN, 1))])
        X.append(window_with_id)
        y.append(targets[i])

    return np.array(X), np.array(y)


def main():
    conn = get_connection()
    prices = load_price_matrix(conn)
    conn.close()

    log_returns = np.log(prices / prices.shift(1)).dropna()
    rolling_vol = log_returns.rolling(ROLLING_VOL_WINDOW).std().dropna()

    common_index = log_returns.index.intersection(rolling_vol.index)
    log_returns = log_returns.loc[common_index]
    rolling_vol = rolling_vol.loc[common_index]

    tickers = list(log_returns.columns)
    n_tickers = len(tickers)

    all_X, all_y = [], []
    for i, ticker in enumerate(tickers):
        onehot = np.eye(n_tickers)[i]
        X_t, y_t = build_ticker_sequences(log_returns[ticker], rolling_vol[ticker], onehot)
        all_X.append(X_t)
        all_y.append(y_t)
        print(f"{ticker}: {X_t.shape[0]} sequences")

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)

    print(f"\nTotal dataset: X={X.shape}, y={y.shape}")

    out_dir = Path(__file__).resolve().parents[2] / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "lstm_X.npy", X)
    np.save(out_dir / "lstm_y.npy", y)
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()