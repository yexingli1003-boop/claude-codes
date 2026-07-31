# strategy_ema_cross_3min.py — 3-Minute EMA Cross Pullback Strategy
#
# This is a STANDALONE strategy for 3-min MNQ trading on NASDAQ.
# It is NOT part of the main STRATEGIES dict - runs via live_trader_3min.py
#
# LONG CONDITIONS (all must be met):
# 1. Setup: A strong bullish candle crosses above both EMA9 and EMA21 with good volume
# 2. EMA trend: EMA9 is above EMA21 OR approaching (narrowing gap, within 5 pts)
# 3. Cross candles: Last 3-5 candles that crossed the EMAs must be majority green
# 4. RSI: Between 50-85 AND increasing (RSI[now] > RSI[3 bars ago])
# 5. Volume momentum: Last 3 candles show increasing/strong volume (majority green vol)
# 6. Pullback: Price pulls back into the zone between EMA9 and EMA21, or touches EMA21
# 7. Entry candle is one of:
#    a) Bullish candle with lower wick >= 40% of body (shows rejection from below)
#    b) Bearish candle with lower wick >= 40% of body (strong wick rejection)
#    c) Red hammer pattern (small red body, long lower wick)
#    d) Engulfing: prev candle undesired + current solid green engulfing it
#
# SHORT CONDITIONS: Mirror of LONG
#
# TRADE MANAGEMENT:
# - Position: 3 MNQ contracts ($2/pt each)
# - TP: Fixed 60 points
# - Initial SL: 10 pts below/above EMA21
# - At +15 pts favorable: move SL to breakeven (entry price)
# - After breakeven: trail SL along EMA21

import pandas as pd
import numpy as np


# ==================================================
# EMA TREND DETECTION
# ==================================================

def ema9_trending_above_21(df, i, lookback=3):
    """
    Check if EMA9 is above EMA21 OR trending toward crossing above.
    Returns True if:
    - EMA9 is already >= EMA21, OR
    - Gap is narrowing over last `lookback` bars AND EMA9 within 5 pts of EMA21
    """
    if i < lookback + 1:
        return False

    row = df.iloc[i]
    e9_now = row["ema_9"]
    e21_now = row["ema_21"]

    if pd.isna(e9_now) or pd.isna(e21_now):
        return False

    # Already above
    if e9_now >= e21_now:
        return True

    # Check if approaching: gap narrowing and close
    gap_now = e21_now - e9_now
    gap_prev = df.iloc[i - lookback]["ema_21"] - df.iloc[i - lookback]["ema_9"]

    if pd.isna(gap_prev):
        return False

    # Gap is narrowing AND EMA9 within 5 pts of EMA21
    return gap_now < gap_prev and gap_now <= 5.0


def ema9_trending_below_21(df, i, lookback=3):
    """Mirror of above for shorts."""
    if i < lookback + 1:
        return False

    row = df.iloc[i]
    e9_now = row["ema_9"]
    e21_now = row["ema_21"]

    if pd.isna(e9_now) or pd.isna(e21_now):
        return False

    if e9_now <= e21_now:
        return True

    gap_now = e9_now - e21_now
    gap_prev = df.iloc[i - lookback]["ema_9"] - df.iloc[i - lookback]["ema_21"]

    if pd.isna(gap_prev):
        return False

    return gap_now < gap_prev and gap_now <= 5.0


# ==================================================
# CROSS DETECTION
# ==================================================

def find_recent_cross_up(df, i, window=15):
    """
    Look backward from index i to find the most recent time price crossed above
    both EMA9 and EMA21. Returns the index of the setup candle, or None.
    """
    if i < window:
        return None

    for k in range(i - 1, max(i - window, 0), -1):
        r = df.iloc[k]
        prev = df.iloc[k - 1] if k > 0 else None

        if prev is None or pd.isna(r["ema_9"]) or pd.isna(r["ema_21"]):
            continue

        # Candle must close above both EMAs
        if r["close"] <= r["ema_9"] or r["close"] <= r["ema_21"]:
            continue

        # Previous candle was below or near EMAs
        if prev["close"] > r["ema_21"]:
            continue

        # Must be a bullish candle with decent body
        if not r["is_bullish"]:
            continue

        avg_body = r["avg_body_20"] if "avg_body_20" in r else None
        if avg_body is None or pd.isna(avg_body):
            continue

        if r["body_abs"] < avg_body * 0.8:
            continue

        return k

    return None


def find_recent_cross_down(df, i, window=15):
    """Mirror: find recent bearish cross below both EMAs."""
    if i < window:
        return None

    for k in range(i - 1, max(i - window, 0), -1):
        r = df.iloc[k]
        prev = df.iloc[k - 1] if k > 0 else None

        if prev is None or pd.isna(r["ema_9"]) or pd.isna(r["ema_21"]):
            continue

        if r["close"] >= r["ema_9"] or r["close"] >= r["ema_21"]:
            continue

        if prev["close"] < r["ema_21"]:
            continue

        if not r["is_bearish"]:
            continue

        avg_body = r["avg_body_20"] if "avg_body_20" in r else None
        if avg_body is None or pd.isna(avg_body):
            continue

        if r["body_abs"] < avg_body * 0.8:
            continue

        return k

    return None


# ==================================================
# VOLUME MOMENTUM
# ==================================================

def volume_momentum_bullish(df, cross_idx, current_idx):
    """
    Check that from the cross candle onward (last 3-5 candles), volume shows
    momentum AND the majority of those candles are green.
    """
    if cross_idx is None or current_idx - cross_idx < 1:
        return False

    # Look at candles from cross_idx to current_idx (inclusive)
    start = max(cross_idx, current_idx - 5)
    end = current_idx + 1

    window = df.iloc[start:end]
    if len(window) < 3:
        return False

    # Majority of cross candles must be green
    num_green = int(window["is_bullish"].sum())
    if num_green <= len(window) / 2:
        return False

    # Volume must show momentum (not declining)
    # Check last 3 candles volume trend — at least one recent uptick
    volumes = window["volume"].tolist()
    if len(volumes) < 3:
        return False

    # Average of last half > average of first half (momentum)
    mid = len(volumes) // 2
    avg_first = np.mean(volumes[:mid]) if mid > 0 else 0
    avg_last = np.mean(volumes[mid:])

    return avg_last >= avg_first * 0.9  # allow slight dip but not decline


def volume_momentum_bearish(df, cross_idx, current_idx):
    """Mirror: majority red candles with non-declining volume."""
    if cross_idx is None or current_idx - cross_idx < 1:
        return False

    start = max(cross_idx, current_idx - 5)
    end = current_idx + 1

    window = df.iloc[start:end]
    if len(window) < 3:
        return False

    num_red = int(window["is_bearish"].sum())
    if num_red <= len(window) / 2:
        return False

    volumes = window["volume"].tolist()
    if len(volumes) < 3:
        return False

    mid = len(volumes) // 2
    avg_first = np.mean(volumes[:mid]) if mid > 0 else 0
    avg_last = np.mean(volumes[mid:])

    return avg_last >= avg_first * 0.9


# ==================================================
# RSI MOMENTUM
# ==================================================

def rsi_bullish_momentum(df, i, lookback=3):
    """RSI between 50 and 85, and increasing over last `lookback` bars."""
    if i < lookback:
        return False

    row = df.iloc[i]
    rsi_now = row["rsi"]

    if pd.isna(rsi_now):
        return False

    if rsi_now < 50 or rsi_now > 85:
        return False

    rsi_past = df.iloc[i - lookback]["rsi"]
    if pd.isna(rsi_past):
        return False

    return rsi_now > rsi_past


def rsi_bearish_momentum(df, i, lookback=3):
    """RSI between 25 and 50, and decreasing."""
    if i < lookback:
        return False

    row = df.iloc[i]
    rsi_now = row["rsi"]

    if pd.isna(rsi_now):
        return False

    if rsi_now < 25 or rsi_now > 50:
        return False

    rsi_past = df.iloc[i - lookback]["rsi"]
    if pd.isna(rsi_past):
        return False

    return rsi_now < rsi_past


# ==================================================
# PULLBACK ZONE DETECTION
# ==================================================

def in_pullback_zone_long(row):
    """
    For LONG: price must have pulled into the zone between EMA9 and EMA21,
    or touched/wicked near EMA21.
    The wick can poke below EMA21, but the body must stay within reasonable range.
    """
    if pd.isna(row["ema_9"]) or pd.isna(row["ema_21"]):
        return False

    # For an uptrend setup, EMA9 > EMA21 (or near). Pullback zone is between them.
    ema_low = min(row["ema_9"], row["ema_21"])
    ema_high = max(row["ema_9"], row["ema_21"])

    # Candle low must have touched or gone below EMA9 (shallow)
    # or near/below EMA21 (deeper), but body should close above EMA21 - 5pts
    touched_zone = row["low"] <= ema_high
    body_ok = min(row["open"], row["close"]) >= ema_low - \
        10  # wick can poke, body stays

    return touched_zone and body_ok


def in_pullback_zone_short(row):
    """Mirror: price pulled UP into EMA zone."""
    if pd.isna(row["ema_9"]) or pd.isna(row["ema_21"]):
        return False

    ema_low = min(row["ema_9"], row["ema_21"])
    ema_high = max(row["ema_9"], row["ema_21"])

    touched_zone = row["high"] >= ema_low
    body_ok = max(row["open"], row["close"]) <= ema_high + 10

    return touched_zone and body_ok


# ==================================================
# CANDLE PATTERN DETECTION
# ==================================================

def is_bullish_reversal_candle(row):
    """
    Valid pullback entry candle for LONG:
    a) Bullish candle with lower wick >= 40% of total range (not body)
    b) Bearish candle with lower wick >= 40% of total range (rejection)
    c) Red hammer: small body, long lower wick (>= 50% of range)
    """
    rng = row["range"]
    if rng <= 0:
        return False, "no_range"

    lw_pct = row["lower_wick"] / rng
    body_pct = row["body_abs"] / rng

    # Pattern A: bullish with significant lower wick
    if row["is_bullish"] and lw_pct >= 0.4:
        return True, "bullish_with_wick"

    # Pattern B: bearish with strong lower wick (rejection from below)
    if row["is_bearish"] and lw_pct >= 0.4:
        return True, "bearish_rejection_wick"

    # Pattern C: red hammer (small body, very long lower wick, short upper)
    if row["is_bearish"] and body_pct <= 0.35 and lw_pct >= 0.5:
        return True, "red_hammer"

    return False, None


def is_bearish_reversal_candle(row):
    """Mirror for shorts — looking for upper wick rejection."""
    rng = row["range"]
    if rng <= 0:
        return False, "no_range"

    uw_pct = row["upper_wick"] / rng
    body_pct = row["body_abs"] / rng

    # Pattern A: bearish with significant upper wick
    if row["is_bearish"] and uw_pct >= 0.4:
        return True, "bearish_with_wick"

    # Pattern B: bullish with strong upper wick (rejection from above)
    if row["is_bullish"] and uw_pct >= 0.4:
        return True, "bullish_rejection_wick"

    # Pattern C: green shooting star (small body, long upper wick)
    if row["is_bullish"] and body_pct <= 0.35 and uw_pct >= 0.5:
        return True, "green_shooting_star"

    return False, None


def is_bullish_engulfing(df, i):
    """
    Fallback scenario for LONG:
    Previous candle was NOT a valid pattern (solid candle)
    Current candle is solid green AND engulfs the previous candle's body.
    """
    if i < 1:
        return False

    prev = df.iloc[i - 1]
    curr = df.iloc[i]

    # Current must be solid green (body >= 60% of range)
    if not curr["is_bullish"]:
        return False
    if curr["range"] <= 0:
        return False
    if curr["body_abs"] / curr["range"] < 0.6:
        return False

    # Must engulf previous body
    prev_body_top = max(prev["open"], prev["close"])
    prev_body_bot = min(prev["open"], prev["close"])

    curr_body_top = max(curr["open"], curr["close"])
    curr_body_bot = min(curr["open"], curr["close"])

    if curr_body_top < prev_body_top or curr_body_bot > prev_body_bot:
        return False

    # Previous was in the pullback zone
    if not in_pullback_zone_long(prev):
        return False

    return True


def is_bearish_engulfing(df, i):
    """Mirror for shorts."""
    if i < 1:
        return False

    prev = df.iloc[i - 1]
    curr = df.iloc[i]

    if not curr["is_bearish"]:
        return False
    if curr["range"] <= 0:
        return False
    if curr["body_abs"] / curr["range"] < 0.6:
        return False

    prev_body_top = max(prev["open"], prev["close"])
    prev_body_bot = min(prev["open"], prev["close"])

    curr_body_top = max(curr["open"], curr["close"])
    curr_body_bot = min(curr["open"], curr["close"])

    if curr_body_top < prev_body_top or curr_body_bot > prev_body_bot:
        return False

    if not in_pullback_zone_short(prev):
        return False

    return True


# ==================================================
# MAIN SIGNAL FUNCTIONS
# ==================================================

def signal_ema_cross_long(df, i):
    """
    S_EMA_LONG: 3-min bullish EMA cross + pullback entry.
    Returns True if all conditions met on bar i.
    """
    if i < 30:
        return False

    row = df.iloc[i]

    # 1. EMA trend (9 above or approaching 21)
    if not ema9_trending_above_21(df, i):
        return False

    # 2. Find recent cross
    cross_idx = find_recent_cross_up(df, i, window=15)
    if cross_idx is None:
        return False

    # Must be at least 1 bar after cross (the pullback)
    if i - cross_idx < 1:
        return False

    # 3. Volume momentum with majority green
    if not volume_momentum_bullish(df, cross_idx, i):
        return False

    # 4. RSI 50-85 increasing
    if not rsi_bullish_momentum(df, i):
        return False

    # 5. In pullback zone
    if not in_pullback_zone_long(row):
        return False

    # 6. Valid reversal candle OR engulfing fallback
    is_pattern, _ = is_bullish_reversal_candle(row)
    if is_pattern:
        return True

    # Fallback: engulfing
    if is_bullish_engulfing(df, i):
        return True

    return False


def signal_ema_cross_short(df, i):
    """Mirror for shorts."""
    if i < 30:
        return False

    row = df.iloc[i]

    if not ema9_trending_below_21(df, i):
        return False

    cross_idx = find_recent_cross_down(df, i, window=15)
    if cross_idx is None:
        return False

    if i - cross_idx < 1:
        return False

    if not volume_momentum_bearish(df, cross_idx, i):
        return False

    if not rsi_bearish_momentum(df, i):
        return False

    if not in_pullback_zone_short(row):
        return False

    is_pattern, _ = is_bearish_reversal_candle(row)
    if is_pattern:
        return True

    if is_bearish_engulfing(df, i):
        return True

    return False


# ==================================================
# CONFIG (for use in live_trader_3min.py)
# ==================================================

STRATEGY_CONFIG = {
    "S_EMA_LONG":  {"direction": "LONG",  "tp": 60, "sl_type": "ema21_trail"},
    "S_EMA_SHORT": {"direction": "SHORT", "tp": 60, "sl_type": "ema21_trail"},
}

SIGNAL_FUNCTIONS = {
    "S_EMA_LONG":  signal_ema_cross_long,
    "S_EMA_SHORT": signal_ema_cross_short,
}
