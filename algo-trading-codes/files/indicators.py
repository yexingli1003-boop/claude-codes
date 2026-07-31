"""
Technical Indicators
=====================
Calculates common technical indicators from OHLCV data.
All functions operate on pandas DataFrames and return Series or values.
"""

import numpy as np
import pandas as pd


class Indicators:
    """Technical indicator calculations for trading strategies."""

    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        """
        Exponential Moving Average.
        Reacts faster to recent prices than SMA.
        """
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def sma(series: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average."""
        return series.rolling(window=period).mean()

    @staticmethod
    def rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """
        Relative Strength Index (0-100).
        Below 30 = oversold (potential buy), above 70 = overbought (potential sell).
        Uses the standard Wilder smoothing method.
        """
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        # Wilder's smoothing (equivalent to EMA with alpha=1/period)
        avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)  # Default to neutral when insufficient data

    @staticmethod
    def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Average True Range — measures volatility.
        Used for dynamic stop-loss and take-profit levels.
        """
        high = df["high"]
        low = df["low"]
        close = df["close"].shift(1)

        tr1 = high - low
        tr2 = (high - close).abs()
        tr3 = (low - close).abs()

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return true_range.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    @staticmethod
    def vwap(df: pd.DataFrame) -> pd.Series:
        """
        Volume Weighted Average Price (intraday).
        Resets each trading day.
        Above VWAP = bullish bias, below = bearish bias.
        """
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        volume = df["volume"].replace(0, 1)  # Avoid division by zero

        # Group by date for intraday reset
        if hasattr(df.index, "date"):
            groups = df.index.date
        else:
            groups = pd.Series(0, index=df.index)  # No reset if no datetime index

        cumulative_tp_vol = (typical_price * volume).groupby(groups).cumsum()
        cumulative_vol = volume.groupby(groups).cumsum()

        return cumulative_tp_vol / cumulative_vol

    @staticmethod
    def bollinger_bands(series: pd.Series, period: int = 20,
                        std_dev: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Bollinger Bands: middle (SMA), upper, lower.
        Returns (upper, middle, lower).
        """
        middle = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return upper, middle, lower

    @staticmethod
    def macd(series: pd.Series, fast: int = 12, slow: int = 26,
             signal: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        MACD: Moving Average Convergence Divergence.
        Returns (macd_line, signal_line, histogram).
        """
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    @classmethod
    def add_all_indicators(cls, df: pd.DataFrame,
                           ema_fast: int = 9, ema_slow: int = 21,
                           rsi_period: int = 14, atr_period: int = 14) -> pd.DataFrame:
        """
        Convenience method: adds all common indicators as new columns.
        Modifies the DataFrame in-place and returns it.
        """
        df = df.copy()

        # EMAs
        df["ema_fast"] = cls.ema(df["close"], ema_fast)
        df["ema_slow"] = cls.ema(df["close"], ema_slow)
        df["ema_trend"] = np.where(df["ema_fast"] > df["ema_slow"], 1, -1)

        # RSI
        df["rsi"] = cls.rsi(df["close"], rsi_period)

        # ATR
        df["atr"] = cls.atr(df, atr_period)

        # VWAP
        if "volume" in df.columns:
            df["vwap"] = cls.vwap(df)

        # MACD
        macd_line, signal_line, hist = cls.macd(df["close"])
        df["macd"] = macd_line
        df["macd_signal"] = signal_line
        df["macd_hist"] = hist

        return df
