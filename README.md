# Stock Portfolio Risk Analyzer

A stock portfolio risk analysis tool combining classical financial risk metrics (VaR, Sharpe ratio) with LSTM-based short-term volatility forecasting, built on historical price data from Yahoo Finance.

## Overview

This project pulls daily OHLCV data for a portfolio of tickers, persists it in MySQL, computes portfolio-level risk metrics, and trains a separate LSTM model per ticker to forecast next-day volatility — helping surface risk-aware signals beyond simple historical statistics.

## Architecture

```
yfinance (ingestion) → MySQL (Docker) → Risk Metrics (VaR, Sharpe)
                                      → LSTM (per-ticker volatility forecast)
                                      → Tableau Dashboard
```

- **Ingestion**: `yfinance` pulls historical daily prices for AAPL, MSFT, GOOGL, AMZN
- **Storage**: MySQL (containerized via Docker Compose) — 5 tables: `tickers`, `daily_prices`, `portfolio_holdings`, `risk_metrics`, `volatility_forecasts`
- **Risk metrics**: Historical & parametric Value-at-Risk (95%/99%), annualized Sharpe ratio
- **Forecasting**: A separate LSTM per ticker (not a shared model — see *Design Notes* below), using 20-day lookback windows of returns, rolling volatility, and normalized trading volume
- **Pipeline**: Fully Dockerized — a single `docker-compose up` runs ingestion → persistence → analytics → forecasting end-to-end
- **Visualization**: Tableau dashboard with 4 panels — price history, portfolio allocation, volatility forecasts, and risk metrics summary

## Tech Stack

**Languages/Libraries**: Python, yfinance, TensorFlow/Keras, NumPy, Pandas, SciPy
**Infrastructure**: MySQL, Docker, Docker Compose
**Visualization**: Tableau Desktop

## Project Structure

```
stock-portfolio-risk-analyzer/
├── data/
│   ├── raw/                  # yfinance CSV output (gitignored)
│   ├── processed/            # LSTM training data + trained models (gitignored)
│   └── tableau_exports/      # CSVs exported for Tableau (gitignored)
├── src/
│   ├── ingestion/            # yfinance price fetching
│   ├── db/                   # MySQL loading, portfolio weights, Tableau export
│   ├── analytics/            # VaR, Sharpe ratio calculations
│   ├── models/                # LSTM data prep, training, inference (per-ticker)
│   └── pipeline.py           # Runs the full pipeline end-to-end
├── docker/
│   ├── Dockerfile.app
│   └── init.sql              # MySQL schema
├── docker-compose.yml
└── requirements.txt
```

## Running the Project

**Full pipeline (Docker):**
```bash
docker compose up --build
```

**Local development:**
```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
docker compose up -d mysql

python src/ingestion/fetch_prices.py
python src/db/load_prices.py
python src/db/set_weights.py
python src/analytics/risk_metrics.py
python src/models/prepare_lstm_data.py
python src/models/train_lstm.py
python src/models/forecast_volatility.py
python src/db/export_for_tableau.py
```

## Results

- **Risk metrics** (equal-weighted portfolio, AAPL/MSFT/GOOGL/AMZN): Historical VaR₉₅ ≈ -2.66%, VaR₉₉ ≈ -4.37%, Sharpe ratio ≈ 0.70
- **LSTM volatility forecasts** (per-ticker, test MAE): AAPL 0.0083, AMZN 0.0100, GOOGL 0.0105, MSFT 0.0097

## Design Notes

**Why per-ticker LSTM models instead of one shared model?**
An earlier version trained a single LSTM across all 4 tickers (with a one-hot ticker flag, later upgraded to a learned embedding). Both approaches converged to near-identical volatility predictions across tickers — the model was picking up shared market-wide volatility patterns rather than ticker-specific behavior. Switching to fully independent per-ticker models fixed this, producing genuinely distinct forecasts driven only by each stock's own history.

**Why CSV export instead of a live Tableau–MySQL connection?**
Tableau's MySQL ODBC driver failed to install on macOS due to a code-signing/dependency issue in Oracle's installer. Rather than compromise system security to force it through, the pipeline exports MySQL query results to CSV, which Tableau reads directly — a standard pattern for periodic-refresh dashboards.

## Dashboard

![alt text](<Screenshot 2026-08-22 at 04.02.11.png>)

## Future Work

- AWS deployment (S3 for model/artifact storage, Lambda for scheduled serverless forecasting runs)
- Configurable portfolio weights (currently equal-weighted)
- Additional model features (macro indicators, sector-relative volatility)