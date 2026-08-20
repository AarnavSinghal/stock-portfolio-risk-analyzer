"""Generate next-day volatility forecasts using the trained LSTM and save to MySQL."""
import os
from datetime import datetime, timedelta
from pathlib import Path

import mysql.connector
import numpy as np
import pandas as pd
import tensorflow as tf
from dotenv import load_dotenv

load_dotenv()

SEQ_LEN = 20
ROLLING_VOL_WINDOW = 5
MODEL_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "models" / "volatility_lstm.keras"


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


def get_ticker_ids(conn) -> dict:
    cursor = conn.cursor()
    cursor.execute("SELECT symbol, id FROM tickers")
    result = dict(cursor.fetchall())
    cursor.close()
    return result


def build_latest_sequence(returns: pd.Series, vol: pd.Series, ticker_onehot: np.ndarray) -> np.ndarray:
    """Build the most recent SEQ_LEN-day window for inference."""
    values = np.column_stack([returns.values, vol.values])
    window = values[-SEQ_LEN:]
    window_with_id = np.hstack([window, np.tile(ticker_onehot, (SEQ_LEN, 1))])
    return window_with_id


def main():
    conn = get_connection()
    prices = load_price_matrix(conn)
    ticker_ids = get_ticker_ids(conn)

    log_returns = np.log(prices / prices.shift(1)).dropna()
    rolling_vol = log_returns.rolling(ROLLING_VOL_WINDOW).std().dropna()

    common_index = log_returns.index.intersection(rolling_vol.index)
    log_returns = log_returns.loc[common_index]
    rolling_vol = rolling_vol.loc[common_index]

    tickers = list(log_returns.columns)
    n_tickers = len(tickers)

    model = tf.keras.models.load_model(MODEL_PATH)

    last_date = common_index.max()
    forecast_date = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")

    cursor = conn.cursor()
    for i, ticker in enumerate(tickers):
        onehot = np.eye(n_tickers)[i]
        seq = build_latest_sequence(log_returns[ticker], rolling_vol[ticker], onehot)
        seq = seq.reshape(1, SEQ_LEN, -1)

        pred_vol = float(model.predict(seq, verbose=0)[0][0])

        ticker_id = ticker_ids[ticker]
        cursor.execute(
            """INSERT INTO volatility_forecasts
               (ticker_id, forecast_date, predicted_volatility, horizon_days)
               VALUES (%s, %s, %s, %s)""",
            (ticker_id, forecast_date, pred_vol, 1),
        )
        print(f"{ticker}: predicted next-day volatility = {pred_vol:.4%}")

    conn.commit()
    cursor.close()
    conn.close()
    print(f"\nForecasts saved for {forecast_date}")


if __name__ == "__main__":
    main()