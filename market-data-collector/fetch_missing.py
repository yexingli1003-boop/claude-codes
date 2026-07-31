"""
Fetch missing market data from April 22nd onward for ES1! and NQ1!
Updates both historical and live (ohlcv + indicators) CSV files.
"""
import os
import sys
import pandas as pd
from tvDatafeed import TvDatafeed, Interval
from collector import calculate_indicators, append_new_bars

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

SYMBOLS = ["NQ1!", "ES1!"]

TIMEFRAMES = {
    "1m":  (Interval.in_1_minute,  5000),
    "3m":  (Interval.in_3_minute,  2500),
    "5m":  (Interval.in_5_minute,  1500),
}

def main():
    tv = TvDatafeed()

    for symbol in SYMBOLS:
        for tf_label, (interval, n_bars) in TIMEFRAMES.items():
            print(f"\n[{symbol}][{tf_label}] Fetching {n_bars} bars...", flush=True)
            try:
                df = tv.get_hist(
                    symbol=symbol,
                    exchange="CME_MINI",
                    interval=interval,
                    n_bars=n_bars,
                )
                if df is None or df.empty:
                    print(f"  ERROR: No data returned", flush=True)
                    continue

                df = df[["open", "high", "low", "close", "volume"]].copy()
                df.index.name = "timestamp"
                df.index = pd.to_datetime(df.index)
                print(f"  Got {len(df)} bars: {df.index[0]} -> {df.index[-1]}", flush=True)

                # ── historical ohlcv ──────────────────────────────────────────
                hist_path = os.path.join(DATA_DIR, "historical", symbol, f"{tf_label}.csv")
                rows = append_new_bars(hist_path, df)
                print(f"  historical ohlcv: {rows} total rows", flush=True)

                # ── historical indicators ─────────────────────────────────────
                hist_ind_path = os.path.join(DATA_DIR, "historical", symbol, f"{tf_label}_indicators.csv")
                ind = calculate_indicators(df)
                rows_ind = append_new_bars(hist_ind_path, ind)
                print(f"  historical indicators: {rows_ind} total rows", flush=True)

                # ── live ohlcv ────────────────────────────────────────────────
                live_ohlcv_path = os.path.join(DATA_DIR, "live", symbol, "ohlcv", f"{tf_label}.csv")
                rows_live = append_new_bars(live_ohlcv_path, df)
                print(f"  live ohlcv: {rows_live} total rows", flush=True)

                # ── live indicators ───────────────────────────────────────────
                live_ind_path = os.path.join(DATA_DIR, "live", symbol, "indicators", f"{tf_label}.csv")
                rows_live_ind = append_new_bars(live_ind_path, ind)
                print(f"  live indicators: {rows_live_ind} total rows", flush=True)

            except Exception as e:
                print(f"  ERROR: {e}", flush=True)

    print("\nDone.", flush=True)

if __name__ == "__main__":
    main()
