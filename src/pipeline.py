"""Run the full portfolio risk pipeline end-to-end."""
import subprocess
import sys


STEPS = [
    ("Fetching prices", "src/ingestion/fetch_prices.py"),
    ("Loading into MySQL", "src/db/load_prices.py"),
    ("Setting portfolio weights", "src/db/set_weights.py"),
    ("Computing risk metrics", "src/analytics/risk_metrics.py"),
    ("Preparing LSTM data", "src/models/prepare_lstm_data.py"),
    ("Forecasting volatility", "src/models/forecast_volatility.py"),
]


def run_step(label: str, script: str):
    print(f"\n{'='*50}\n{label}\n{'='*50}")
    result = subprocess.run([sys.executable, script])
    if result.returncode != 0:
        print(f"Step failed: {script}")
        sys.exit(1)


def main():
    for label, script in STEPS:
        run_step(label, script)
    print("\nPipeline complete.")


if __name__ == "__main__":
    main()