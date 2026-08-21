"""Generate next-day volatility forecasts using per-ticker LSTMs and save to MySQL."""
import os
from datetime import timedelta
from pathlib import Path

import mysql.connector
import numpy as np
import pandas as pd
import tensorflow as tf
from dotenv import load_dotenv

load_dotenv()

SEQ_LEN = 20
ROLLING_VOL_WINDOW = 5
MODEL_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "models"


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


def get_ticker_ids(conn) -> dict:
    cursor = conn.cursor()
    cursor.execute("SELECT symbol, id FROM tickers")
    result = dict(cursor.fetchall())
    cursor.close()
    return result


def main():
    conn = get_connection()
    prices, volume = load_price_and_volume(conn)
    ticker_db_ids = get_ticker_ids(conn)

    log_returns = np.log(prices / prices.shift(1)).dropna()
    rolling_vol = log_returns.rolling(ROLLING_VOL_WINDOW).std().dropna()
    log_volume = np.log(volume + 1)

    common_index = log_returns.index.intersection(rolling_vol.index).intersection(log_volume.index)
    log_returns = log_returns.loc[common_index]
    rolling_vol = rolling_vol.loc[common_index]
    log_volume = log_volume.loc[common_index]

    last_date = common_index.max()
    forecast_date = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")

    cursor = conn.cursor()
    for symbol in log_returns.columns:
        model_path = MODEL_DIR / f"volatility_lstm_{symbol}.keras"
        if not model_path.exists():
            print(f"Skipping {symbol}: no trained model found at {model_path}")
            continue

        model = tf.keras.models.load_model(model_path)

        ticker_log_vol = log_volume[symbol]
        ticker_log_vol_norm = (ticker_log_vol - ticker_log_vol.mean()) / ticker_log_vol.std()

        values = np.column_stack([
            log_returns[symbol].values,
            rolling_vol[symbol].values,
            ticker_log_vol_norm.values,
        ])
        seq = values[-SEQ_LEN:].reshape(1, SEQ_LEN, -1)

        pred_vol = float(model.predict(seq, verbose=0)[0][0])

        ticker_db_id = ticker_db_ids[symbol]
        cursor.execute(
            """INSERT INTO volatility_forecasts
               (ticker_id, forecast_date, predicted_volatility, horizon_days)
               VALUES (%s, %s, %s, %s)""",
            (ticker_db_id, forecast_date, pred_vol, 1),
        )
        print(f"{symbol}: predicted next-day volatility = {pred_vol:.4%}")

    conn.commit()
    cursor.close()
    conn.close()
    print(f"\nForecasts saved for {forecast_date}")


if __name__ == "__main__":
    main()