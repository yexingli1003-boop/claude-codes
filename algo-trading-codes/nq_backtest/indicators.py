# indicators.py — Vectorized indicator calculations on a pandas DataFrame

import numpy as np
import pandas as pd

from config import ATR_PERIOD, RSI_PERIOD, EMA_FAST, EMA_SLOW, AVG_BODY_PERIOD, AVG_RANGE_PERIOD


def add_candle_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df["body"] = df["close"] - df["open"]
    df["body_abs"] = df["body"].abs()
    df["range"] = df["high"] - df["low"]
    df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
    df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]
    df["wick_total"] = df["upper_wick"] + df["lower_wick"]
    df["is_bullish"] = df["close"] > df["open"]
    df["is_bearish"] = df["close"] < df["open"]
    return df


def add_ema(df: pd.DataFrame, period: int) -> pd.DataFrame:
    col = f"ema_{period}"
    df[col] = df["close"].ewm(span=period, adjust=False).mean()
    return df


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    alpha = 1.0 / period
    avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.fillna(100.0)  # avg_loss == 0 → RSI = 100
    df["rsi"] = rsi
    return df


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["atr"] = tr.rolling(period, min_periods=period).mean()
    return df


def add_avg_body(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    df["avg_body_20"] = df["body_abs"].rolling(
        period, min_periods=period).mean()
    return df


def add_avg_range(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    df["avg_range_20"] = df["range"].rolling(period, min_periods=period).mean()
    return df


def add_delta(df: pd.DataFrame) -> pd.DataFrame:
    buy_pct = np.where(df["range"] > 0, (df["close"] -
                       df["low"]) / df["range"], 0.5)
    df["estimated_delta"] = df["volume"] * (2 * buy_pct - 1)
    # Daily cumulative delta
    if "datetime" in df.columns:
        df["date_group"] = pd.to_datetime(df["datetime"]).dt.date
    else:
        df["date_group"] = 0
    df["cum_delta"] = df.groupby("date_group")["estimated_delta"].cumsum()
    df.drop(columns=["date_group"], inplace=True)
    return df


def build_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = add_candle_metrics(df)
    df = add_ema(df, EMA_FAST)
    df = add_ema(df, EMA_SLOW)
    df = add_rsi(df, RSI_PERIOD)
    df = add_atr(df, ATR_PERIOD)
    df = add_avg_body(df, AVG_BODY_PERIOD)
    df = add_avg_range(df, AVG_RANGE_PERIOD)
    # Delta estimation (from close position within candle)
    df["buy_pct"] = np.where(
        df["range"] > 0, (df["close"] - df["low"]) / df["range"], 0.5)
    df["estimated_delta"] = df["volume"] * (2 * df["buy_pct"] - 1)
    df["cum_delta"] = df["estimated_delta"].cumsum()
    df = add_delta(df)
    return df
