import os
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime, timedelta

load_dotenv()

api_key = os.getenv("ALPACA_API_KEY")
secret_key = os.getenv("ALPACA_SECRET_KEY")

# Connect to trading account
trading = TradingClient(api_key, secret_key, paper=True)
account = trading.get_account()
print(f"Account Status: {account.status}")
print(f"Cash: ${float(account.cash):,.2f}")
print(f"Buying Power: ${float(account.buying_power):,.2f}")

# Fetch QQQ data (NASDAQ 100 ETF — same index as NQ futures)
data_client = StockHistoricalDataClient(api_key, secret_key)
request = StockBarsRequest(
    symbol_or_symbols="QQQ",
    timeframe=TimeFrame.Minute,
    start=datetime.now() - timedelta(days=3),
    end=datetime.now(),
)
bars = data_client.get_stock_bars(request)
df = bars.df
print(f"\nRetrieved {len(df)} bars of QQQ 1-min data")
print(df.tail(5))
print("\nConnection successful!")
