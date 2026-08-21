"""Prepare sequences for LSTM volatility forecasting from MySQL price data."""
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


def build_ticker_sequences(returns: pd.Series, vol: pd.Series, log_volume: pd.Series, ticker_idx: int):
    """Build (X, ticker_ids, y) sequences for a single ticker's series."""
    X, ticker_ids, y = [], [], []
    values = np.column_stack([returns.values, vol.values, log_volume.values])
    targets = returns.abs().shift(-1).values  # next-day |return|

    for i in range(SEQ_LEN, len(values) - 1):
        window = values[i - SEQ_LEN:i]
        X.append(window)
        ticker_ids.append(ticker_idx)
        y.append(targets[i])

    return np.array(X), np.array(ticker_ids), np.array(y)


def main():
    conn = get_connection()
    prices, volume = load_price_and_volume(conn)
    conn.close()

    log_returns = np.log(prices / prices.shift(1)).dropna()
    rolling_vol = log_returns.rolling(ROLLING_VOL_WINDOW).std().dropna()
    log_volume = np.log(volume + 1)  # +1 avoids log(0)

    common_index = log_returns.index.intersection(rolling_vol.index).intersection(log_volume.index)
    log_returns = log_returns.loc[common_index]
    rolling_vol = rolling_vol.loc[common_index]
    log_volume = log_volume.loc[common_index]

    # Normalize log_volume per ticker (z-score) so it's on a similar scale to returns/vol
    log_volume = (log_volume - log_volume.mean()) / log_volume.std()

    tickers = list(log_returns.columns)

    all_X, all_ticker_ids, all_y = [], [], []
    for i, ticker in enumerate(tickers):
        X_t, ids_t, y_t = build_ticker_sequences(
            log_returns[ticker], rolling_vol[ticker], log_volume[ticker], i
        )
        all_X.append(X_t)
        all_ticker_ids.append(ids_t)
        all_y.append(y_t)
        print(f"{ticker}: {X_t.shape[0]} sequences")

    X = np.concatenate(all_X, axis=0)
    ticker_ids = np.concatenate(all_ticker_ids, axis=0)
    y = np.concatenate(all_y, axis=0)

    print(f"\nTotal dataset: X={X.shape}, ticker_ids={ticker_ids.shape}, y={y.shape}")

    out_dir = Path(__file__).resolve().parents[2] / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "lstm_X.npy", X)
    np.save(out_dir / "lstm_ticker_ids.npy", ticker_ids)
    np.save(out_dir / "lstm_y.npy", y)

    # Save ticker name -> id mapping for inference later
    with open(out_dir / "ticker_index.txt", "w") as f:
        for i, t in enumerate(tickers):
            f.write(f"{t},{i}\n")

    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()