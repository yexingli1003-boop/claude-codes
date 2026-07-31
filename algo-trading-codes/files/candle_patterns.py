"""
Candlestick Pattern Detection
==============================
Detects common candlestick patterns from OHLCV data.
Each detector returns a signal: 1 (bullish), -1 (bearish), or 0 (no pattern).
"""

import logging
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class Signal(Enum):
    BULLISH = 1
    BEARISH = -1
    NEUTRAL = 0


@dataclass
class PatternResult:
    """Result of a candlestick pattern scan."""
    name: str
    signal: Signal
    strength: float  # 0.0 to 1.0, how "textbook" the pattern is
    description: str


class CandlePatternDetector:
    """
    Detects candlestick patterns from OHLCV DataFrames.

    Expects a DataFrame with columns: open, high, low, close, volume
    Index should be datetime.
    """

    def __init__(self, min_body_ratio: float = 0.6):
        self.min_body_ratio = min_body_ratio

    def scan_all(self, df: pd.DataFrame) -> list[PatternResult]:
        """
        Run ALL pattern detectors on the most recent candles.
        Returns a list of detected patterns (may be empty).
        """
        if len(df) < 3:
            return []

        patterns = []
        detectors = [
            self.detect_engulfing,
            self.detect_hammer,
            self.detect_shooting_star,
            self.detect_doji,
            self.detect_morning_star,
            self.detect_evening_star,
            self.detect_three_white_soldiers,
            self.detect_three_black_crows,
        ]

        for detector in detectors:
            result = detector(df)
            if result and result.signal != Signal.NEUTRAL:
                patterns.append(result)

        if patterns:
            names = [f"{p.name}({'↑' if p.signal == Signal.BULLISH else '↓'})"
                     for p in patterns]
            logger.info(f"🕯️ Patterns detected: {', '.join(names)}")

        return patterns

    # ─── HELPER METHODS ──────────────────────────────────────────

    @staticmethod
    def _body(row) -> float:
        """Absolute candle body size."""
        return abs(row["close"] - row["open"])

    @staticmethod
    def _range(row) -> float:
        """Full candle range (high - low)."""
        return row["high"] - row["low"]

    @staticmethod
    def _is_bullish(row) -> bool:
        return row["close"] > row["open"]

    @staticmethod
    def _is_bearish(row) -> bool:
        return row["close"] < row["open"]

    @staticmethod
    def _upper_shadow(row) -> float:
        return row["high"] - max(row["open"], row["close"])

    @staticmethod
    def _lower_shadow(row) -> float:
        return min(row["open"], row["close"]) - row["low"]

    def _body_ratio(self, row) -> float:
        """Body as a fraction of total range."""
        r = self._range(row)
        return self._body(row) / r if r > 0 else 0

    # ─── PATTERN DETECTORS ───────────────────────────────────────

    def detect_engulfing(self, df: pd.DataFrame) -> PatternResult | None:
        """
        Bullish Engulfing: Previous bearish candle fully engulfed by current bullish candle.
        Bearish Engulfing: Previous bullish candle fully engulfed by current bearish candle.
        """
        prev = df.iloc[-2]
        curr = df.iloc[-1]

        # Bullish Engulfing
        if (self._is_bearish(prev) and self._is_bullish(curr)
                and curr["open"] <= prev["close"]
                and curr["close"] >= prev["open"]
                and self._body(curr) > self._body(prev)):
            strength = min(self._body(curr) / (self._body(prev) + 1e-10), 2.0) / 2.0
            return PatternResult("Bullish Engulfing", Signal.BULLISH, strength,
                                 "Current bullish candle fully engulfs previous bearish candle")

        # Bearish Engulfing
        if (self._is_bullish(prev) and self._is_bearish(curr)
                and curr["open"] >= prev["close"]
                and curr["close"] <= prev["open"]
                and self._body(curr) > self._body(prev)):
            strength = min(self._body(curr) / (self._body(prev) + 1e-10), 2.0) / 2.0
            return PatternResult("Bearish Engulfing", Signal.BEARISH, strength,
                                 "Current bearish candle fully engulfs previous bullish candle")

        return None

    def detect_hammer(self, df: pd.DataFrame) -> PatternResult | None:
        """
        Hammer: Small body at the top, long lower shadow (≥2x body).
        Bullish reversal signal at the bottom of a downtrend.
        """
        curr = df.iloc[-1]
        body = self._body(curr)
        lower = self._lower_shadow(curr)
        upper = self._upper_shadow(curr)
        rng = self._range(curr)

        if rng == 0:
            return None

        if (lower >= 2 * body
                and upper <= body * 0.5
                and body / rng >= 0.15):
            strength = min(lower / (body + 1e-10), 4.0) / 4.0
            return PatternResult("Hammer", Signal.BULLISH, strength,
                                 "Small body with long lower shadow — potential bullish reversal")
        return None

    def detect_shooting_star(self, df: pd.DataFrame) -> PatternResult | None:
        """
        Shooting Star: Small body at the bottom, long upper shadow (≥2x body).
        Bearish reversal signal at the top of an uptrend.
        """
        curr = df.iloc[-1]
        body = self._body(curr)
        upper = self._upper_shadow(curr)
        lower = self._lower_shadow(curr)
        rng = self._range(curr)

        if rng == 0:
            return None

        if (upper >= 2 * body
                and lower <= body * 0.5
                and body / rng >= 0.15):
            strength = min(upper / (body + 1e-10), 4.0) / 4.0
            return PatternResult("Shooting Star", Signal.BEARISH, strength,
                                 "Small body with long upper shadow — potential bearish reversal")
        return None

    def detect_doji(self, df: pd.DataFrame) -> PatternResult | None:
        """
        Doji: Very small body relative to total range.
        Indicates indecision — direction depends on context.
        """
        curr = df.iloc[-1]
        rng = self._range(curr)

        if rng == 0:
            return None

        body_ratio = self._body_ratio(curr)
        if body_ratio < 0.1:  # Body is <10% of total range
            return PatternResult("Doji", Signal.NEUTRAL, 1.0 - body_ratio,
                                 "Very small body — market indecision")
        return None

    def detect_morning_star(self, df: pd.DataFrame) -> PatternResult | None:
        """
        Morning Star (3-candle bullish reversal):
        1. Large bearish candle
        2. Small-bodied candle (gaps down)
        3. Large bullish candle closing above midpoint of candle 1
        """
        if len(df) < 3:
            return None

        c1, c2, c3 = df.iloc[-3], df.iloc[-2], df.iloc[-1]

        if (self._is_bearish(c1) and self._body_ratio(c1) >= self.min_body_ratio
                and self._body_ratio(c2) < 0.3  # Small body
                and self._is_bullish(c3) and self._body_ratio(c3) >= self.min_body_ratio
                and c3["close"] > (c1["open"] + c1["close"]) / 2):
            return PatternResult("Morning Star", Signal.BULLISH, 0.8,
                                 "Three-candle bullish reversal pattern")
        return None

    def detect_evening_star(self, df: pd.DataFrame) -> PatternResult | None:
        """
        Evening Star (3-candle bearish reversal):
        1. Large bullish candle
        2. Small-bodied candle (gaps up)
        3. Large bearish candle closing below midpoint of candle 1
        """
        if len(df) < 3:
            return None

        c1, c2, c3 = df.iloc[-3], df.iloc[-2], df.iloc[-1]

        if (self._is_bullish(c1) and self._body_ratio(c1) >= self.min_body_ratio
                and self._body_ratio(c2) < 0.3
                and self._is_bearish(c3) and self._body_ratio(c3) >= self.min_body_ratio
                and c3["close"] < (c1["open"] + c1["close"]) / 2):
            return PatternResult("Evening Star", Signal.BEARISH, 0.8,
                                 "Three-candle bearish reversal pattern")
        return None

    def detect_three_white_soldiers(self, df: pd.DataFrame) -> PatternResult | None:
        """Three consecutive bullish candles, each closing higher."""
        if len(df) < 3:
            return None

        c1, c2, c3 = df.iloc[-3], df.iloc[-2], df.iloc[-1]

        if (self._is_bullish(c1) and self._is_bullish(c2) and self._is_bullish(c3)
                and c2["close"] > c1["close"] and c3["close"] > c2["close"]
                and self._body_ratio(c1) >= 0.5
                and self._body_ratio(c2) >= 0.5
                and self._body_ratio(c3) >= 0.5):
            return PatternResult("Three White Soldiers", Signal.BULLISH, 0.85,
                                 "Three strong consecutive bullish candles")
        return None

    def detect_three_black_crows(self, df: pd.DataFrame) -> PatternResult | None:
        """Three consecutive bearish candles, each closing lower."""
        if len(df) < 3:
            return None

        c1, c2, c3 = df.iloc[-3], df.iloc[-2], df.iloc[-1]

        if (self._is_bearish(c1) and self._is_bearish(c2) and self._is_bearish(c3)
                and c2["close"] < c1["close"] and c3["close"] < c2["close"]
                and self._body_ratio(c1) >= 0.5
                and self._body_ratio(c2) >= 0.5
                and self._body_ratio(c3) >= 0.5):
            return PatternResult("Three Black Crows", Signal.BEARISH, 0.85,
                                 "Three strong consecutive bearish candles")
        return None
