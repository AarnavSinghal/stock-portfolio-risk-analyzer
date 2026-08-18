"""Fetch historical OHLCV data for a portfolio of tickers via yfinance."""
import yfinance as yf
import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


def fetch_ticker_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download daily OHLCV data for a single ticker."""
    df = yf.download(ticker, start=start, end=end, auto_adjust=True)
    df["ticker"] = ticker
    return df


def fetch_portfolio(tickers: list[str], start: str, end: str) -> None:
    """Fetch and save data for each ticker in the portfolio to data/raw/."""
    for ticker in tickers:
        df = fetch_ticker_data(ticker, start, end)
        out_path = RAW_DIR / f"{ticker}.csv"
        df.to_csv(out_path)
        print(f"Saved {ticker}: {len(df)} rows -> {out_path}")


if __name__ == "__main__":
    portfolio = ["AAPL", "MSFT", "GOOGL", "AMZN"]
    fetch_portfolio(portfolio, start="2019-01-01", end="2026-08-18")