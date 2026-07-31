# config.py — All constants and strategy configurations

from ctypes.wintypes import LONG


COMMISSION = 0.9          # points per round-trip trade
WARMUP_CANDLES = 50       # candles to skip before evaluating signals
TIMEOUT_CANDLES = 40      # max candles to hold a position

# Indicator periods
ATR_PERIOD = 14
RSI_PERIOD = 14
EMA_FAST = 9
EMA_SLOW = 21
AVG_BODY_PERIOD = 20
AVG_RANGE_PERIOD = 20

# ATR trailing stop multipliers
TRAIL_ACTIVATE_MULTIPLIER = 1.0   # activate when favorable_move > 1x ATR
TRAIL_OFFSET_MULTIPLIER = 1.5     # trail = best_price +/- 1.5x ATR

# Per-strategy configs: name -> {direction, tp (points), sl (points)}
STRATEGY_CONFIGS = {
    "S1_BullFlagEMA":   {"direction": "LONG",  "tp": 60, "sl": 40},
    "S2_BullFlagBasic": {"direction": "LONG",  "tp": 60, "sl": 40},
    "S3_Hammer":        {"direction": "LONG",  "tp": 60, "sl": 40},
    "S4_DeathCross":    {"direction": "SHORT", "tp": 50, "sl": 35},
    "S5_RSIOversold":   {"direction": "LONG",  "tp": 50, "sl": 35},
    "S6_DeltaFlip":     {"direction": "LONG",  "tp": 60, "sl": 40},
    "S7_BullDivEMA":    {"direction": "LONG",  "tp": 60, "sl": 40},
    "S8_BullDivRSI":    {"direction": "LONG",  "tp": 40, "sl": 20},
}
