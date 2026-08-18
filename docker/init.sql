CREATE TABLE IF NOT EXISTS tickers (
    id INT AUTO_INCREMENT PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL UNIQUE,
    name VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS daily_prices (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    ticker_id INT NOT NULL,
    price_date DATE NOT NULL,
    open_price DECIMAL(12,4),
    high_price DECIMAL(12,4),
    low_price DECIMAL(12,4),
    close_price DECIMAL(12,4),
    volume BIGINT,
    FOREIGN KEY (ticker_id) REFERENCES tickers(id),
    UNIQUE KEY unique_ticker_date (ticker_id, price_date)
);

CREATE TABLE IF NOT EXISTS portfolio_holdings (
    id INT AUTO_INCREMENT PRIMARY KEY,
    ticker_id INT NOT NULL,
    weight DECIMAL(5,4) NOT NULL,
    FOREIGN KEY (ticker_id) REFERENCES tickers(id)
);

CREATE TABLE IF NOT EXISTS risk_metrics (
    id INT AUTO_INCREMENT PRIMARY KEY,
    calc_date DATE NOT NULL,
    var_95 DECIMAL(10,6),
    var_99 DECIMAL(10,6),
    sharpe_ratio DECIMAL(10,6),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS volatility_forecasts (
    id INT AUTO_INCREMENT PRIMARY KEY,
    ticker_id INT NOT NULL,
    forecast_date DATE NOT NULL,
    predicted_volatility DECIMAL(10,6),
    horizon_days INT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (ticker_id) REFERENCES tickers(id)
);