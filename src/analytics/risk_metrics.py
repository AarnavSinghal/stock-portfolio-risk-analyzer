"""Compute portfolio returns, VaR, and Sharpe ratio from MySQL price data."""
import os
from pathlib import Path

import mysql.connector
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

TRADING_DAYS = 252
RISK_FREE_RATE = 0.04  # annualized, adjust as needed


def get_connection():
    return mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
    )


def load_price_matrix(conn) -> pd.DataFrame:
    """Return a DataFrame of close prices, tickers as columns, dates as index."""
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


def load_weights(conn) -> pd.Series:
    """Return ticker symbol -> weight."""
    query = """
        SELECT t.symbol, ph.weight
        FROM portfolio_holdings ph
        JOIN tickers t ON t.id = ph.ticker_id
    """
    df = pd.read_sql(query, conn)
    return df.set_index("symbol")["weight"].astype(float)


def compute_log_returns(price_matrix: pd.DataFrame) -> pd.DataFrame:
    return np.log(price_matrix / price_matrix.shift(1)).dropna()


def compute_portfolio_returns(returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    weights = weights.reindex(returns.columns).fillna(0)
    return returns.dot(weights)


def historical_var(portfolio_returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical simulation VaR — a negative number represents expected loss."""
    return np.percentile(portfolio_returns, (1 - confidence) * 100)


def parametric_var(portfolio_returns: pd.Series, confidence: float = 0.95) -> float:
    """Variance-covariance (parametric) VaR assuming normal distribution."""
    from scipy.stats import norm
    mu = portfolio_returns.mean()
    sigma = portfolio_returns.std()
    z = norm.ppf(1 - confidence)
    return mu + z * sigma


def sharpe_ratio(portfolio_returns: pd.Series, risk_free_rate: float = RISK_FREE_RATE) -> float:
    """Annualized Sharpe ratio from daily log returns."""
    daily_rf = risk_free_rate / TRADING_DAYS
    excess_returns = portfolio_returns - daily_rf
    return (excess_returns.mean() / excess_returns.std()) * np.sqrt(TRADING_DAYS)


def rolling_sharpe(portfolio_returns: pd.Series, window: int = 60) -> pd.Series:
    daily_rf = RISK_FREE_RATE / TRADING_DAYS
    excess = portfolio_returns - daily_rf
    return (excess.rolling(window).mean() / excess.rolling(window).std()) * np.sqrt(TRADING_DAYS)


def main():
    conn = get_connection()
    prices = load_price_matrix(conn)
    weights = load_weights(conn)

    print("Weights:\n", weights, "\n")

    returns = compute_log_returns(prices)
    port_returns = compute_portfolio_returns(returns, weights)

    var_95_hist = historical_var(port_returns, 0.95)
    var_99_hist = historical_var(port_returns, 0.99)
    var_95_param = parametric_var(port_returns, 0.95)
    sharpe = sharpe_ratio(port_returns)

    print(f"Historical VaR (95%): {var_95_hist:.4%}")
    print(f"Historical VaR (99%): {var_99_hist:.4%}")
    print(f"Parametric VaR (95%): {var_95_param:.4%}")
    print(f"Annualized Sharpe Ratio: {sharpe:.4f}")

    # Persist to risk_metrics table
    cursor = conn.cursor()
    cursor.execute(
        """INSERT INTO risk_metrics (calc_date, var_95, var_99, sharpe_ratio)
           VALUES (CURDATE(), %s, %s, %s)""",
        (float(var_95_hist), float(var_99_hist), float(sharpe)),
    )
    conn.commit()
    cursor.close()
    conn.close()
    print("\nSaved to risk_metrics table.")


if __name__ == "__main__":
    main()