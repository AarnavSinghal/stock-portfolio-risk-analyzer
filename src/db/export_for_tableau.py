"""Export MySQL tables to CSV for Tableau."""
import os
from pathlib import Path

import mysql.connector
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "tableau_exports"
OUT_DIR.mkdir(parents=True, exist_ok=True)


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

    queries = {
        "prices": """
            SELECT t.symbol, dp.price_date, dp.open_price, dp.high_price,
                   dp.low_price, dp.close_price, dp.volume
            FROM daily_prices dp
            JOIN tickers t ON t.id = dp.ticker_id
            ORDER BY dp.price_date
        """,
        "risk_metrics": "SELECT * FROM risk_metrics ORDER BY calc_date",
        "volatility_forecasts": """
            SELECT t.symbol, vf.forecast_date, vf.predicted_volatility, vf.horizon_days
            FROM volatility_forecasts vf
            JOIN tickers t ON t.id = vf.ticker_id
            ORDER BY vf.forecast_date
        """,
        "portfolio_holdings": """
            SELECT t.symbol, ph.weight
            FROM portfolio_holdings ph
            JOIN tickers t ON t.id = ph.ticker_id
        """,
    }

    for name, query in queries.items():
        df = pd.read_sql(query, conn)
        out_path = OUT_DIR / f"{name}.csv"
        df.to_csv(out_path, index=False)
        print(f"{name}: {len(df)} rows -> {out_path}")

    conn.close()
    print("\nDone.")


if __name__ == "__main__":
    main()