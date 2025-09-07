CREATE TABLE IF NOT EXISTS binance_klines (
    id BIGSERIAL PRIMARY KEY,
    open_time TIMESTAMP NOT NULL,
    symbol TEXT NOT NULL,
    open NUMERIC,
    high NUMERIC,
    low NUMERIC,
    close NUMERIC,
    volume NUMERIC,
    taker_buy_base_volume NUMERIC,
    taker_buy_quote_volume NUMERIC,
    status TEXT DEFAULT 'pending',
    UNIQUE (open_time, symbol)
);
