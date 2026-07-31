import os
import logging
import pandas as pd
from tvDatafeed import TvDatafeed, Interval
import ta

logger = logging.getLogger(__name__)

SYMBOLS = ["NQ1!", "ES1!"]

TIMEFRAME_MAP = {
    "1m":  Interval.in_1_minute,
    "3m":  Interval.in_3_minute,
    "5m":  Interval.in_5_minute,
    "15m": Interval.in_15_minute,
    "1h":  Interval.in_1_hour,
    "4h":  Interval.in_4_hour,
    "1d":  Interval.in_daily,
    "1w":  Interval.in_weekly,
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

_tv = None

def get_tv():
    global _tv
    if _tv is None:
        _tv = TvDatafeed()
    return _tv


def fetch_ohlcv(symbol: str, interval: Interval, n_bars: int = 100) -> pd.DataFrame:
    tv = get_tv()
    df = tv.get_hist(symbol=symbol, exchange="CME_MINI", interval=interval, n_bars=n_bars)
    if df is None or df.empty:
        raise ValueError(f"No data returned for {symbol}")
    df = df[["open", "high", "low", "close", "volume"]].copy()
    df.index.name = "timestamp"
    df.index = pd.to_datetime(df.index)
    return df


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    ind = pd.DataFrame(index=df.index)
    ind.index.name = "timestamp"

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    ind["ema9"]   = ta.trend.ema_indicator(close, window=9)
    ind["ema21"]  = ta.trend.ema_indicator(close, window=21)
    ind["ema50"]  = ta.trend.ema_indicator(close, window=50)
    ind["ema100"] = ta.trend.ema_indicator(close, window=100)
    ind["ema200"] = ta.trend.ema_indicator(close, window=200)

    ind["rsi14"] = ta.momentum.rsi(close, window=14)

    atr = ta.volatility.AverageTrueRange(high, low, close, window=14)
    ind["atr14"] = atr.average_true_range()

    macd = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
    ind["macd"]        = macd.macd()
    ind["macd_signal"] = macd.macd_signal()
    ind["macd_hist"]   = macd.macd_diff()

    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    ind["bb_upper"] = bb.bollinger_hband()
    ind["bb_mid"]   = bb.bollinger_mavg()
    ind["bb_lower"] = bb.bollinger_lband()

    ind["vol_ma20"] = volume.rolling(window=20).mean()

    # VWAP (rolling daily approximation using typical price * volume)
    typical = (high + low + close) / 3
    ind["vwap"] = (typical * volume).cumsum() / volume.cumsum()

    return ind


def append_new_bars(filepath: str, new_df: pd.DataFrame):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if os.path.exists(filepath):
        existing = pd.read_csv(filepath, index_col="timestamp", parse_dates=True)
        combined = pd.concat([existing, new_df])
        combined = combined[~combined.index.duplicated(keep="last")]
        combined.sort_index(inplace=True)
    else:
        combined = new_df
    combined.to_csv(filepath)
    return len(combined)


def collect_all(timeframe: str, n_bars: int = 5):
    interval = TIMEFRAME_MAP[timeframe]
    for symbol in SYMBOLS:
        try:
            df = fetch_ohlcv(symbol, interval, n_bars=max(n_bars, 250))
            ind = calculate_indicators(df)

            ohlcv_path = os.path.join(DATA_DIR, "live", symbol, "ohlcv", f"{timeframe}.csv")
            ind_path   = os.path.join(DATA_DIR, "live", symbol, "indicators", f"{timeframe}.csv")

            rows = append_new_bars(ohlcv_path, df)
            append_new_bars(ind_path, ind)

            logger.info(f"[{symbol}][{timeframe}] collected — {rows} total rows")
        except Exception as e:
            logger.error(f"[{symbol}][{timeframe}] ERROR: {e}")
