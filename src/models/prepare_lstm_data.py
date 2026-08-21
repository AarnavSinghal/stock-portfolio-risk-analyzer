"""Prepare per-ticker sequences for LSTM volatility forecasting."""
import os
from pathlib import Path

import mysql.connector
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

SEQ_LEN = 20
ROLLING_VOL_WINDOW = 5


def get_connection():
    return mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
    )


def load_price_and_volume(conn):
    query = """
        SELECT t.symbol, dp.price_date, dp.close_price, dp.volume
        FROM daily_prices dp
        JOIN tickers t ON t.id = dp.ticker_id
        ORDER BY dp.price_date
    """
    df = pd.read_sql(query, conn)
    price_matrix = df.pivot(index="price_date", columns="symbol", values="close_price").astype(float)
    volume_matrix = df.pivot(index="price_date", columns="symbol", values="volume").astype(float)
    price_matrix.index = pd.to_datetime(price_matrix.index)
    volume_matrix.index = pd.to_datetime(volume_matrix.index)
    return price_matrix, volume_matrix


def build_sequences(returns: pd.Series, vol: pd.Series, log_volume: pd.Series):
    X, y = [], []
    values = np.column_stack([returns.values, vol.values, log_volume.values])
    targets = returns.abs().shift(-1).values

    for i in range(SEQ_LEN, len(values) - 1):
        X.append(values[i - SEQ_LEN:i])
        y.append(targets[i])

    return np.array(X), np.array(y)


def main():
    conn = get_connection()
    prices, volume = load_price_and_volume(conn)
    conn.close()

    log_returns = np.log(prices / prices.shift(1)).dropna()
    rolling_vol = log_returns.rolling(ROLLING_VOL_WINDOW).std().dropna()
    log_volume = np.log(volume + 1)

    common_index = log_returns.index.intersection(rolling_vol.index).intersection(log_volume.index)
    log_returns = log_returns.loc[common_index]
    rolling_vol = rolling_vol.loc[common_index]
    log_volume = log_volume.loc[common_index]

    out_dir = Path(__file__).resolve().parents[2] / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    for ticker in log_returns.columns:
        ticker_log_vol = log_volume[ticker]
        ticker_log_vol_norm = (ticker_log_vol - ticker_log_vol.mean()) / ticker_log_vol.std()

        X, y = build_sequences(log_returns[ticker], rolling_vol[ticker], ticker_log_vol_norm)
        np.save(out_dir / f"lstm_X_{ticker}.npy", X)
        np.save(out_dir / f"lstm_y_{ticker}.npy", y)
        print(f"{ticker}: X={X.shape}, y={y.shape}")

    print(f"\nSaved per-ticker datasets to {out_dir}")


if __name__ == "__main__":
    main()