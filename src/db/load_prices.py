"""Load CSV price data from data/raw/ into MySQL tables."""
import math
import os
from pathlib import Path

import mysql.connector
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def get_connection():
    """Open a MySQL connection using credentials from .env."""
    return mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
    )


def clean(value):
    """Convert NaN/None to a proper SQL NULL-safe value."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def upsert_ticker(cursor, symbol: str) -> int:
    """Insert ticker if new, return its ticker_id."""
    cursor.execute(
        "INSERT INTO tickers (symbol) VALUES (%s) "
        "ON DUPLICATE KEY UPDATE symbol = symbol",
        (symbol,),
    )
    cursor.execute("SELECT id FROM tickers WHERE symbol = %s", (symbol,))
    return cursor.fetchone()[0]


def load_ticker_csv(cursor, ticker_id: int, csv_path: Path) -> int:
    """Load one ticker's CSV into daily_prices, skipping duplicates."""
    df = pd.read_csv(csv_path, skiprows=[1, 2], index_col=0)
    df.index = pd.to_datetime(df.index)

    rows = []
    for idx, row in df.iterrows():
        volume = row["Volume"]
        rows.append((
            ticker_id,
            idx.strftime("%Y-%m-%d"),
            clean(row["Open"]),
            clean(row["High"]),
            clean(row["Low"]),
            clean(row["Close"]),
            clean(int(volume)) if pd.notnull(volume) else None,
        ))

    cursor.executemany(
        """INSERT INTO daily_prices
           (ticker_id, price_date, open_price, high_price, low_price, close_price, volume)
           VALUES (%s, %s, %s, %s, %s, %s, %s)
           ON DUPLICATE KEY UPDATE
               open_price = VALUES(open_price),
               close_price = VALUES(close_price)""",
        rows,
    )
    return len(rows)


def main():
    conn = get_connection()
    cursor = conn.cursor()

    csv_files = sorted(RAW_DIR.glob("*.csv"))
    for csv_path in csv_files:
        symbol = csv_path.stem
        ticker_id = upsert_ticker(cursor, symbol)
        n_rows = load_ticker_csv(cursor, ticker_id, csv_path)
        print(f"{symbol}: loaded {n_rows} rows (ticker_id={ticker_id})")

    conn.commit()
    cursor.close()
    conn.close()
    print("Done.")


if __name__ == "__main__":
    main()