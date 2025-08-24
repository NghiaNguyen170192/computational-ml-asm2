import requests
import psycopg2
from airflow.plugins_manager import AirflowPlugin
from airflow.models import BaseOperator
from airflow.hooks.base import BaseHook
from airflow.utils.decorators import apply_defaults
import json


# -------- Binance Hook -------- #
class BinanceHook(BaseHook):
    def __init__(self, conn_id="binance_default"):
        super().__init__()
        self.conn_id = conn_id

    def get_order_book(self, symbol="BTCUSDT", limit=5):
        url = f"https://api.binance.com/api/v3/depth?symbol={symbol}&limit={limit}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            self.log.error(f"Failed to fetch order book for {symbol}: {e}")
            raise


# -------- Operator -------- #
class BinanceOrderBookToPostgresOperator(BaseOperator):
    @apply_defaults
    def __init__(self, symbol="BTCUSDT", limit=5, pg_conn_id="postgres_default", **kwargs):
        super().__init__(**kwargs)
        self.symbol = symbol
        self.limit = limit
        self.pg_conn_id = pg_conn_id

    def execute(self, context):
        # Step 1: Get order book
        binance_hook = BinanceHook()
        order_book = binance_hook.get_order_book(
            symbol=self.symbol, limit=self.limit)

        # Step 2: Connect to Postgres
        conn = BaseHook.get_connection(self.pg_conn_id)
        with psycopg2.connect(
            host=conn.host,
            database=conn.schema,
            user=conn.login,
            password=conn.password,
            port=conn.port
        ) as pg_conn:
            with pg_conn.cursor() as cur:
                # Step 3: Insert data
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS binance_orderbook (
                        id SERIAL PRIMARY KEY,
                        symbol TEXT,
                        bids JSONB,
                        asks JSONB,
                        ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                cur.execute("""
                    INSERT INTO binance_orderbook (symbol, bids, asks)
                    VALUES (%s, %s, %s)
                """, (
                    self.symbol,
                    json.dumps(order_book["bids"]),
                    json.dumps(order_book["asks"])
                ))

                pg_conn.commit()

        self.log.info(f"Inserted order book for {self.symbol} into Postgres")


# -------- Plugin Registration -------- #
class BinancePlugin(AirflowPlugin):
    name = "binance_plugin"
    hooks = [BinanceHook]
    operators = [BinanceOrderBookToPostgresOperator]
