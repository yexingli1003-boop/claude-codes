# simulator.py — Universal trade simulation engine

import pandas as pd

from config import COMMISSION, TIMEOUT_CANDLES, TRAIL_ACTIVATE_MULTIPLIER, TRAIL_OFFSET_MULTIPLIER


def simulate_trade(
    df: pd.DataFrame,
    entry_bar_index: int,
    direction: str,          # "LONG" or "SHORT"
    tp_points: float,
    sl_points: float,
    atr_at_entry: float,
    commission: float = COMMISSION,
    timeout: int = TIMEOUT_CANDLES,
) -> dict:
    """
    Simulates a single trade starting at df.iloc[entry_bar_index].open.

    Per-bar evaluation order (per spec):
      1. SL/trail check first (worst-case)
      2. TP check
      3. Update trailing stop

    Returns dict with keys:
      entry_price, exit_price, pnl, exit_reason, bars_held, direction
    """
    entry_price = df.iloc[entry_bar_index]["open"]

    if direction == "LONG":
        tp_price = entry_price + tp_points
        sl_price = entry_price - sl_points
    else:
        tp_price = entry_price - tp_points
        sl_price = entry_price + sl_points

    trail_level = sl_price
    best_price = entry_price
    total_bars = len(df)

    for bar_offset in range(1, timeout + 1):
        bar_idx = entry_bar_index + bar_offset
        if bar_idx >= total_bars:
            # Out of data — timeout at last available close
            exit_bar = total_bars - 1
            exit_price = df.iloc[exit_bar]["close"]
            pnl = (exit_price - entry_price if direction == "LONG" else entry_price - exit_price) - commission
            return {
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl": pnl,
                "exit_reason": "TIMEOUT",
                "bars_held": bar_offset,
                "direction": direction,
            }

        candle = df.iloc[bar_idx]
        candle_low = candle["low"]
        candle_high = candle["high"]

        if direction == "LONG":
            # Step 1: SL/trail check first
            if candle_low <= trail_level:
                exit_price = trail_level
                pnl = (exit_price - entry_price) - commission
                reason = "SL" if trail_level == sl_price else "TRAIL"
                return {
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl": pnl,
                    "exit_reason": reason,
                    "bars_held": bar_offset,
                    "direction": direction,
                }

            # Step 2: TP check
            if candle_high >= tp_price:
                exit_price = tp_price
                pnl = (exit_price - entry_price) - commission
                return {
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl": pnl,
                    "exit_reason": "TP",
                    "bars_held": bar_offset,
                    "direction": direction,
                }

            # Step 3: Update trailing stop
            if candle_high > best_price:
                best_price = candle_high
            favorable_move = best_price - entry_price
            if favorable_move > atr_at_entry * TRAIL_ACTIVATE_MULTIPLIER:
                new_trail = best_price - (TRAIL_OFFSET_MULTIPLIER * atr_at_entry)
                if new_trail > trail_level:
                    trail_level = new_trail

        else:  # SHORT
            # Step 1: SL/trail check first
            if candle_high >= trail_level:
                exit_price = trail_level
                pnl = (entry_price - exit_price) - commission
                reason = "SL" if trail_level == sl_price else "TRAIL"
                return {
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl": pnl,
                    "exit_reason": reason,
                    "bars_held": bar_offset,
                    "direction": direction,
                }

            # Step 2: TP check
            if candle_low <= tp_price:
                exit_price = tp_price
                pnl = (entry_price - exit_price) - commission
                return {
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl": pnl,
                    "exit_reason": "TP",
                    "bars_held": bar_offset,
                    "direction": direction,
                }

            # Step 3: Update trailing stop
            if candle_low < best_price:
                best_price = candle_low
            favorable_move = entry_price - best_price
            if favorable_move > atr_at_entry * TRAIL_ACTIVATE_MULTIPLIER:
                new_trail = best_price + (TRAIL_OFFSET_MULTIPLIER * atr_at_entry)
                if new_trail < trail_level:
                    trail_level = new_trail

    # Timeout: exit at close of last bar in window
    exit_bar_idx = entry_bar_index + timeout
    if exit_bar_idx >= total_bars:
        exit_bar_idx = total_bars - 1
    exit_price = df.iloc[exit_bar_idx]["close"]
    pnl = (exit_price - entry_price if direction == "LONG" else entry_price - exit_price) - commission
    return {
        "entry_price": entry_price,
        "exit_price": exit_price,
        "pnl": pnl,
        "exit_reason": "TIMEOUT",
        "bars_held": timeout,
        "direction": direction,
    }
