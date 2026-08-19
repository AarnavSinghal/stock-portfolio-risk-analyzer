"""Set equal-weighted portfolio holdings for all tickers currently in the DB."""
import os
from pathlib import Path

import mysql.connector
from dotenv import load_dotenv

load_dotenv()


def get_connection():
    return mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
    )


def main():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT id, symbol FROM tickers ORDER BY id")
    tickers = cursor.fetchall()

    if not tickers:
        print("No tickers found. Run load_prices.py first.")
        return

    weight = round(1.0 / len(tickers), 4)

    cursor.execute("DELETE FROM portfolio_holdings")
    for ticker_id, symbol in tickers:
        cursor.execute(
            "INSERT INTO portfolio_holdings (ticker_id, weight) VALUES (%s, %s)",
            (ticker_id, weight),
        )
        print(f"{symbol}: weight={weight}")

    conn.commit()
    cursor.close()
    conn.close()
    print("Done.")


if __name__ == "__main__":
    main()