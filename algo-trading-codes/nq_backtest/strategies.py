# strategies.py — 5 strategy signal detector functions

import math
import pandas as pd

from config import STRATEGY_CONFIGS


def _check_bull_flag_base(df: pd.DataFrame, i: int, min_pullback_pct: float, max_pullback_pct: float) -> bool:
    """
    Shared bull flag logic used by S1 and S2.
    Looks back 3, 4, or 5 candles for an impulse, then validates consolidation and pullback depth.
    Returns True if a valid pattern is found.
    """
    for lookback in (3, 4, 5):
        impulse_idx = i - lookback
        if impulse_idx < 0:
            continue

        impulse = df.iloc[impulse_idx]

        # Condition 1: Impulse candle must be bullish and large
        avg_body = impulse["avg_body_20"]
        if pd.isna(avg_body) or avg_body == 0:
            continue
        if not impulse["is_bullish"]:
            continue
        if impulse["body_abs"] <= 1.3 * avg_body:
            continue

        # Consolidation candles: df.iloc[impulse_idx+1 : i]  (excludes current candle i)
        consol = df.iloc[impulse_idx + 1: i]
        if len(consol) < 2:
            continue

        # Condition 2: Each consolidation candle has small body
        consol_avg_body = consol["avg_body_20"]
        if consol_avg_body.isna().any():
            continue
        if not (consol["body_abs"] <= consol_avg_body).all():
            continue

        # At least half of consolidation candles are wicky
        wicky = (consol["wick_total"] >= 0.8 * consol["body_abs"])
        if wicky.sum() < math.ceil(len(consol) / 2):
            continue

        # Condition 3: Pullback depth check
        impulse_move = impulse["close"] - impulse["open"]
        if abs(impulse_move) < 1e-9:
            continue
        last_consol_close = consol.iloc[-1]["close"]
        pullback_pct = abs(last_consol_close -
                           impulse["close"]) / impulse_move * 100

        if pullback_pct < min_pullback_pct or pullback_pct >= max_pullback_pct:
            continue

        return True

    return False


def signal_s1_bull_flag_ema(df: pd.DataFrame, i: int) -> bool:
    """Strategy 1: Bull Flag + EMA filter (LONG)."""
    row = df.iloc[i]

    # Conditions 5 & 6: EMA trend filter
    if pd.isna(row["ema_9"]) or pd.isna(row["ema_21"]):
        return False
    if row["ema_9"] <= row["ema_21"]:
        return False
    if row["close"] <= row["ema_9"]:
        return False

    # Condition 4: Current candle must be bullish
    if not row["is_bullish"]:
        return False

    # Conditions 1-3: Bull flag base with pullback 20-45%
    return _check_bull_flag_base(df, i, min_pullback_pct=20.0, max_pullback_pct=45.0)


def signal_s2_bull_flag_basic(df: pd.DataFrame, i: int) -> bool:
    """Strategy 2: Bull Flag basic, no EMA conditions, pullback < 75% only."""
    row = df.iloc[i]

    # Condition 4: Current candle must be bullish
    if not row["is_bullish"]:
        return False

    # Conditions 1-3: Bull flag base, no minimum pullback, max 75%
    return _check_bull_flag_base(df, i, min_pullback_pct=0.0, max_pullback_pct=75.0)


def signal_s3_hammer(df: pd.DataFrame, i: int) -> bool:
    """Strategy 3: Hammer candle pattern (LONG)."""
    row = df.iloc[i]
    rng = row["range"]

    if rng <= 0:
        return False
    if row["body_abs"] / rng >= 0.35:
        return False
    if row["lower_wick"] / rng <= 0.55:
        return False
    if row["upper_wick"] / rng >= 0.15:
        return False

    return True


def signal_s4_death_cross(df: pd.DataFrame, i: int) -> bool:
    """Strategy 4: EMA 9/21 death cross (SHORT). Requires fresh crossover on current candle."""
    if i < 1:
        return False

    row = df.iloc[i]
    prev = df.iloc[i - 1]

    if pd.isna(row["ema_9"]) or pd.isna(row["ema_21"]):
        return False
    if pd.isna(prev["ema_9"]) or pd.isna(prev["ema_21"]):
        return False

    # Fresh death cross: was >= on previous, now <
    if not (prev["ema_9"] >= prev["ema_21"]):
        return False
    if not (row["ema_9"] < row["ema_21"]):
        return False

    # Current candle must be bearish
    if not row["is_bearish"]:
        return False

    return True


def signal_s5_rsi_oversold(df: pd.DataFrame, i: int) -> bool:
    """Strategy 5: RSI extreme oversold + bullish candle (LONG)."""
    row = df.iloc[i]

    if pd.isna(row["rsi"]):
        return False
    if row["rsi"] >= 25.0:
        return False
    if not row["is_bullish"]:
        return False

    return True


def signal_s6_delta_flip(df: pd.DataFrame, i: int) -> bool:
    """Strategy 6: Cumulative delta flips negative to positive (LONG)."""
    if i < 10:
        return False
    row = df.iloc[i]

    if "cum_delta" not in df.columns or "estimated_delta" not in df.columns:
        return False
    if pd.isna(row.get("cum_delta")) or pd.isna(row.get("estimated_delta")):
        return False

    if not row["is_bullish"]:
        return False
    if row["estimated_delta"] <= 0:
        return False
    if row["cum_delta"] <= 0:
        return False

    recent = df.iloc[i - 5:i]
    if recent["cum_delta"].mean() > 0:
        return False

    return True


def signal_s7_bull_div_ema(df: pd.DataFrame, i: int) -> bool:
    """Strategy 7: Bullish divergence + EMA stack (LONG)."""
    if i < 25:
        return False
    row = df.iloc[i]

    if not row["is_bullish"]:
        return False

    for col in ["ema_9", "ema_21", "cum_delta"]:
        if col not in df.columns or pd.isna(row.get(col)):
            return False

    if not (row["ema_9"] > row["ema_21"]):
        return False

    current_window = df.iloc[i - 4:i + 1]
    curr_price_low = current_window["low"].min()
    curr_delta_low = current_window["cum_delta"].min()

    search = df.iloc[i - 25:i - 5]
    if len(search) < 5:
        return False

    prev_low_idx = search["low"].idxmin()
    prev_price_low = search["low"].min()

    delta_around = df.iloc[max(0, prev_low_idx - 2)
                               :min(len(df), prev_low_idx + 3)]
    prev_delta_low = delta_around["cum_delta"].min()

    if curr_price_low < prev_price_low and curr_delta_low > prev_delta_low:
        return True

    return False


def signal_s6_delta_flip(df: pd.DataFrame, i: int) -> bool:
    """Strategy 6: Cumulative delta flips negative to positive (LONG)."""
    if i < 10:
        return False
    row = df.iloc[i]

    if "cum_delta" not in df.columns or "estimated_delta" not in df.columns:
        return False
    if pd.isna(row.get("cum_delta")) or pd.isna(row.get("estimated_delta")):
        return False

    # Current candle must be bullish with positive delta
    if not row["is_bullish"]:
        return False
    if row["estimated_delta"] <= 0:
        return False
    if row["cum_delta"] <= 0:
        return False

    # Previous 5 candles had negative average cum_delta
    recent = df.iloc[i - 5:i]
    if recent["cum_delta"].mean() > 0:
        return False

    return True


def signal_s7_bull_div_ema(df: pd.DataFrame, i: int) -> bool:
    """Strategy 7: Bullish divergence + EMA stack (LONG)."""
    if i < 25:
        return False
    row = df.iloc[i]

    if not row["is_bullish"]:
        return False

    # EMA stack: 9 > 21 > 50 (need ema_50)
    for col in ["ema_9", "ema_21", "cum_delta"]:
        if col not in df.columns or pd.isna(row.get(col)):
            return False

    # Check if ema_50 exists, if not skip that filter
    if "ema_50" in df.columns and not pd.isna(row.get("ema_50")):
        if not (row["ema_9"] > row["ema_21"] > row["ema_50"]):
            return False
    else:
        if not (row["ema_9"] > row["ema_21"]):
            return False

    # Bullish divergence: price lower low, delta higher low
    current_window = df.iloc[i - 4:i + 1]
    curr_price_low = current_window["low"].min()
    curr_delta_low = current_window["cum_delta"].min()

    search = df.iloc[i - 25:i - 5]
    if len(search) < 5:
        return False

    prev_low_idx = search["low"].idxmin()
    prev_price_low = search["low"].min()

    delta_around = df.iloc[max(0, prev_low_idx - 2):min(len(df), prev_low_idx + 3)]
    prev_delta_low = delta_around["cum_delta"].min()

    # Lower low in price but higher low in delta
    if curr_price_low < prev_price_low and curr_delta_low > prev_delta_low:
        return True

    return False


def signal_s8_bull_div_rsi(df: pd.DataFrame, i: int) -> bool:
    """
    Strategy 8: Bullish RSI divergence at oversold (LONG) — 5-min only.
    """
    if i < 30:
        return False

    row = df.iloc[i]

    if pd.isna(row["rsi"]):
        return False

    first_half = df.iloc[i - 20:i - 10]
    second_half = df.iloc[i - 10:i]

    if len(first_half) < 5 or len(second_half) < 5:
        return False

    if not (second_half["low"].min() < first_half["low"].min()):
        return False
    if not (second_half["high"].max() < first_half["high"].max()):
        return False

    rsi_window = df.iloc[i - 15:i + 1]
    if rsi_window["rsi"].min() >= 25:
        return False

    recent_low_idx = df.iloc[i - 5:i + 1]["low"].idxmin()
    prior_window = df.iloc[i - 20:i - 5]
    if len(prior_window) < 3:
        return False
    prior_low = prior_window["low"].min()
    recent_low = df.iloc[i - 5:i + 1]["low"].min()
    if recent_low >= prior_low:
        return False

    prior_low_idx = prior_window["low"].idxmin()
    rsi_at_prior_low = df.iloc[prior_low_idx]["rsi"]
    rsi_at_recent_low = df.iloc[recent_low_idx]["rsi"]

    if pd.isna(rsi_at_prior_low) or pd.isna(rsi_at_recent_low):
        return False
    if rsi_at_recent_low <= rsi_at_prior_low:
        return False

    if not row["is_bullish"]:
        if row["range"] > 0:
            if row["lower_wick"] / row["range"] < 0.5:
                return False
        else:
            return False

    return True


# Strategy registry consumed by backtest.py
STRATEGIES: dict = {
    name: {
        "signal_fn": fn,
        **STRATEGY_CONFIGS[name],
    }
    for name, fn in [
        ("S1_BullFlagEMA",   signal_s1_bull_flag_ema),
        ("S2_BullFlagBasic", signal_s2_bull_flag_basic),
        ("S3_Hammer",        signal_s3_hammer),
        ("S4_DeathCross",    signal_s4_death_cross),
        ("S5_RSIOversold",   signal_s5_rsi_oversold),
        ("S6_DeltaFlip",     signal_s6_delta_flip),
        ("S7_BullDivEMA",    signal_s7_bull_div_ema),
        ("S8_BullDivRSI",    signal_s8_bull_div_rsi),
    ]
}
