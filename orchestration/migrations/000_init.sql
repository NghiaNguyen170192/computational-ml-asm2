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

CREATE INDEX idx_binance_klines_open_time
    ON binance_klines (open_time DESC);

CREATE TABLE IF NOT EXISTS  crypto_news (
    id SERIAL PRIMARY KEY,
    date TIMESTAMP NOT NULL,
    sentiment JSONB,
    source VARCHAR(150),
    subject TEXT,
    text TEXT,
    title TEXT,
    url TEXT
);


CREATE INDEX idx_crypto_news_date
    ON crypto_news (date DESC);

CREATE INDEX idx_crypto_news_source
    ON crypto_news (source);

CREATE INDEX idx_crypto_news_sentiment
    ON crypto_news
    USING gin (sentiment);

CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE INDEX idx_crypto_news_title_trgm
    ON crypto_news
    USING gin (title gin_trgm_ops);

CREATE INDEX idx_crypto_news_text_trgm
    ON crypto_news
    USING gin (text gin_trgm_ops);