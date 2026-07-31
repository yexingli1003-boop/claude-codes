"""
Run once to backfill historical intraday data:
    python historical.py

Fetches up to 5000 bars for 1m, 3m, 5m, 15m for NQ1!, ES1!, YM1!.
"""
import os
import logging
from collector import fetch_ohlcv, calculate_indicators, append_new_bars, SYMBOLS, TIMEFRAME_MAP, DATA_DIR

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(LOG_DIR, "collection.log")),
    ],
)
logger = logging.getLogger(__name__)

HISTORICAL_TIMEFRAMES = ["1m", "3m", "5m", "15m"]
N_BARS = 5000


def run():
    for symbol in SYMBOLS:
        for tf in HISTORICAL_TIMEFRAMES:
            logger.info(f"[HISTORICAL] Fetching {symbol} {tf} ({N_BARS} bars)...")
            try:
                df = fetch_ohlcv(symbol, TIMEFRAME_MAP[tf], n_bars=N_BARS)
                ind = calculate_indicators(df)

                ohlcv_path = os.path.join(DATA_DIR, "historical", symbol, f"{tf}.csv")
                ind_path   = os.path.join(DATA_DIR, "historical", symbol, f"{tf}_indicators.csv")

                os.makedirs(os.path.dirname(ohlcv_path), exist_ok=True)
                rows = append_new_bars(ohlcv_path, df)
                append_new_bars(ind_path, ind)

                start = df.index[0].strftime("%Y-%m-%d %H:%M")
                end   = df.index[-1].strftime("%Y-%m-%d %H:%M")
                logger.info(f"[HISTORICAL] {symbol} {tf} -- {rows} rows | {start} -> {end}")
            except Exception as e:
                logger.error(f"[HISTORICAL] {symbol} {tf} FAILED: {e}")

    logger.info("Historical backfill complete.")


if __name__ == "__main__":
    run()
