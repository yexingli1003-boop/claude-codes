"""
Hybrid Trading Strategy
========================
Combines candlestick pattern detection with technical indicator confirmation
to generate high-confidence trade signals for NQ futures.

Signal Logic:
    1. Detect a candlestick pattern (entry trigger)
    2. Confirm with indicators (RSI, EMA trend, VWAP position)
    3. Calculate dynamic stop-loss and take-profit using ATR
    4. Output a TradeSignal with entry, stop, and target levels
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import pandas as pd

from strategy.candle_patterns import CandlePatternDetector, Signal as CandleSignal
from strategy.indicators import Indicators
from config.config import StrategyConfig

logger = logging.getLogger(__name__)


class TradeDirection(Enum):
    LONG = "long"
    SHORT = "short"


@dataclass
class TradeSignal:
    """A complete trade signal with entry, exit, and metadata."""
    direction: TradeDirection
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float         # 0.0 to 1.0
    pattern_name: str         # Which candle pattern triggered it
    confirmations: list[str]  # Which indicators confirmed
    reason: str               # Human-readable explanation


class HybridStrategy:
    """
    Hybrid strategy combining candlestick patterns + indicator confirmation.

    Flow:
        on_new_candle(df)
            → detect candle patterns
            → check indicator confirmations
            → if enough confirmations, generate TradeSignal
            → return signal (or None if no setup)
    """

    def __init__(self, config: StrategyConfig):
        self.config = config
        self.pattern_detector = CandlePatternDetector(
            min_body_ratio=config.min_candle_body_ratio
        )
        self.indicators = Indicators()

        # Minimum confirmations needed to trigger a trade
        self.min_confirmations = 2  # At least 2 indicators must agree

    def analyze(self, df: pd.DataFrame) -> Optional[TradeSignal]:
        """
        Main analysis method. Call this on each new candle.

        Args:
            df: DataFrame with columns [open, high, low, close, volume]
                Must have at least 50 rows for reliable indicators.

        Returns:
            TradeSignal if a valid setup is found, None otherwise.
        """
        if len(df) < 50:
            logger.debug("Not enough data for analysis (need ≥50 candles)")
            return None

        # Step 1: Add indicators to the DataFrame
        df = self.indicators.add_all_indicators(
            df,
            ema_fast=self.config.ema_fast,
            ema_slow=self.config.ema_slow,
            rsi_period=self.config.rsi_period,
            atr_period=self.config.atr_period,
        )

        # Step 2: Detect candlestick patterns
        patterns = self.pattern_detector.scan_all(df)

        if not patterns:
            return None  # No pattern = no trade

        # Step 3: For each pattern, check indicator confirmations
        for pattern in patterns:
            signal = self._evaluate_pattern(df, pattern)
            if signal:
                return signal

        return None

    def _evaluate_pattern(self, df: pd.DataFrame, pattern) -> Optional[TradeSignal]:
        """
        Check if indicators confirm a candlestick pattern.
        Returns a TradeSignal if enough confirmations exist.
        """
        curr = df.iloc[-1]
        confirmations = []
        is_bullish = pattern.signal == CandleSignal.BULLISH

        # ─── Confirmation 1: EMA Trend ─────────────────────────
        ema_bullish = curr["ema_fast"] > curr["ema_slow"]
        if is_bullish and ema_bullish:
            confirmations.append(f"EMA trend bullish (fast {curr['ema_fast']:.1f} > "
                                 f"slow {curr['ema_slow']:.1f})")
        elif not is_bullish and not ema_bullish:
            confirmations.append(f"EMA trend bearish (fast {curr['ema_fast']:.1f} < "
                                 f"slow {curr['ema_slow']:.1f})")

        # ─── Confirmation 2: RSI ───────────────────────────────
        rsi_val = curr["rsi"]
        if is_bullish and rsi_val < self.config.rsi_overbought:
            # Not overbought = room to run up
            confirmations.append(f"RSI at {rsi_val:.1f} (not overbought)")
            if rsi_val < self.config.rsi_oversold:
                # Extra strength: actually oversold
                confirmations.append(f"RSI oversold at {rsi_val:.1f}")
        elif not is_bullish and rsi_val > self.config.rsi_oversold:
            confirmations.append(f"RSI at {rsi_val:.1f} (not oversold)")
            if rsi_val > self.config.rsi_overbought:
                confirmations.append(f"RSI overbought at {rsi_val:.1f}")

        # ─── Confirmation 3: VWAP Position ─────────────────────
        if self.config.use_vwap_filter and "vwap" in curr.index:
            vwap_val = curr["vwap"]
            price = curr["close"]
            if is_bullish and price > vwap_val:
                confirmations.append(
                    f"Price above VWAP ({price:.1f} > {vwap_val:.1f})")
            elif not is_bullish and price < vwap_val:
                confirmations.append(
                    f"Price below VWAP ({price:.1f} < {vwap_val:.1f})")

        # ─── Confirmation 4: MACD Histogram ────────────────────
        if "macd_hist" in curr.index:
            hist = curr["macd_hist"]
            prev_hist = df.iloc[-2]["macd_hist"] if len(df) > 1 else 0
            if is_bullish and hist > prev_hist:
                confirmations.append("MACD histogram rising")
            elif not is_bullish and hist < prev_hist:
                confirmations.append("MACD histogram falling")

        # ─── Check if we have enough confirmations ─────────────
        if len(confirmations) < self.min_confirmations:
            logger.debug(f"Pattern {pattern.name} found but only "
                         f"{len(confirmations)} confirmations (need {self.min_confirmations})")
            return None

        # ─── Calculate entry, stop, and target ─────────────────
        entry_price = curr["close"]
        atr = curr["atr"]

        if is_bullish:
            stop_loss = entry_price - (atr * self.config.atr_stop_multiplier)
            take_profit = entry_price + \
                (atr * self.config.atr_target_multiplier)
            direction = TradeDirection.LONG
        else:
            stop_loss = entry_price + (atr * self.config.atr_stop_multiplier)
            take_profit = entry_price - \
                (atr * self.config.atr_target_multiplier)
            direction = TradeDirection.SHORT

        # Confidence based on pattern strength + number of confirmations
        confidence = min(
            (pattern.strength * 0.4) + (len(confirmations) / 5 * 0.6),
            1.0
        )

        reason = (f"{pattern.name} detected | "
                  f"{len(confirmations)} confirmations: "
                  f"{', '.join(confirmations)}")

        signal = TradeSignal(
            direction=direction,
            entry_price=round(entry_price, 2),
            stop_loss=round(stop_loss, 2),
            take_profit=round(take_profit, 2),
            confidence=round(confidence, 3),
            pattern_name=pattern.name,
            confirmations=confirmations,
            reason=reason,
        )

        logger.info(f"🎯 SIGNAL: {direction.value.upper()} @ {entry_price:.2f} | "
                    f"SL: {stop_loss:.2f} | TP: {take_profit:.2f} | "
                    f"Confidence: {confidence:.0%} | {pattern.name}")

        return signal
