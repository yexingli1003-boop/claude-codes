"""
fetch_topstep_history.py
────────────────────────────────────────────────────────────
Pulls NQ/MNQ historical bar data from your TopstepX account
and saves it as a CSV ready for the Market Replay app.

USAGE:
    python fetch_topstep_history.py                  # defaults: MNQ, 5-min, 30 days
    python fetch_topstep_history.py --tf 15          # 15-min bars
    python fetch_topstep_history.py --days 60        # 60 days of history
    python fetch_topstep_history.py --tf 5 --days 14 # 5-min, 2 weeks
    python fetch_topstep_history.py --symbol NQ      # full-size NQ

OUTPUT:
    nq_5min_YYYYMMDD_YYYYMMDD.csv   (or nq_15min_...)
    Columns: datetime, open, high, low, close, volume

REQUIREMENTS:
    pip install project-x-py python-dotenv pandas
    .env file with PROJECT_X_API_KEY and PROJECT_X_USERNAME
"""

import asyncio
import argparse
import os
import sys
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import pandas as pd
from dotenv import load_dotenv

# ── Same auth as your live_trader.py ─────────────────────────────────────────
try:
    from project_x_py import ProjectX
except ImportError:
    print("ERROR: project-x-py not installed.")
    print("Run: pip install project-x-py")
    sys.exit(1)

# ── CONFIG ────────────────────────────────────────────────────────────────────
CT = ZoneInfo("America/Chicago")

# TopstepX caps a single retrieveBars call at ~500 bars.
# We chunk requests to get months of data.
BARS_PER_REQUEST = 500


# ── MAIN FETCH ────────────────────────────────────────────────────────────────
async def fetch_history(symbol: str, tf_minutes: int, days: int) -> pd.DataFrame:
    print(f"\nConnecting to TopstepX...")
    async with ProjectX.from_env() as client:
        await client.authenticate()
        account = client.account_info
        print(f"  Account : {account.name}")
        print(f"  Balance : ${account.balance:,.2f}")

        # ── Find contract ────────────────────────────────────────────────────
        print(f"\nSearching for {symbol} contract...")
        instruments = await client.search_instruments(symbol)
        if not instruments:
            raise RuntimeError(f"No contract found for symbol: {symbol}")
        contract = instruments[0]
        print(f"  Found   : {contract.name} (ID: {contract.id})")

        # ── Chunk strategy ───────────────────────────────────────────────────
        # Each chunk covers BARS_PER_REQUEST bars worth of calendar time
        mins_per_chunk = tf_minutes * BARS_PER_REQUEST
        end_dt = datetime.now(CT)
        start_dt = end_dt - timedelta(days=days)

        all_frames = []
        chunk_end = end_dt
        chunk_start = max(start_dt, chunk_end -
                          timedelta(minutes=mins_per_chunk))

        print(f"\nFetching {tf_minutes}-min bars from "
              f"{start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')}")
        print(f"  Chunking in {BARS_PER_REQUEST}-bar blocks...\n")

        chunk_num = 0
        while chunk_end > start_dt:
            chunk_num += 1
            print(f"  Chunk {chunk_num:02d}: "
                  f"{chunk_start.strftime('%Y-%m-%d %H:%M')} → "
                  f"{chunk_end.strftime('%Y-%m-%d %H:%M')} ", end="", flush=True)
            try:
                # project_x_py get_bars with explicit date range
                # Falls back to days= parameter if range not supported by SDK version
                try:
                    data = await client.get_bars(
                        symbol,
                        start=chunk_start,
                        end=chunk_end,
                        interval=tf_minutes,
                    )
                except TypeError:
                    # Older SDK: only supports days=
                    # Calculate how many days this chunk covers
                    chunk_days = max(
                        1, int((chunk_end - chunk_start).total_seconds() / 86400) + 1)
                    data = await client.get_bars(symbol, days=chunk_days, interval=tf_minutes)

                # Normalise to DataFrame
                if hasattr(data, "to_pandas"):
                    df = data.to_pandas()
                elif isinstance(data, pd.DataFrame):
                    df = data.copy()
                else:
                    df = pd.DataFrame(data)

                if df is None or len(df) == 0:
                    print("(empty)")
                else:
                    print(f"({len(df)} bars)")
                    all_frames.append(df)

            except Exception as e:
                print(f"(ERROR: {e})")

            # Step back one chunk
            chunk_end = chunk_start
            chunk_start = max(start_dt, chunk_end -
                              timedelta(minutes=mins_per_chunk))
            if chunk_end <= start_dt:
                break

        if not all_frames:
            raise RuntimeError(
                "No data returned from any chunk. Check credentials and symbol.")

        # ── Combine & clean ──────────────────────────────────────────────────
        print(f"\nMerging {chunk_num} chunks...")
        combined = pd.concat(all_frames, ignore_index=True)
        combined.columns = combined.columns.str.lower().str.strip()

        # Find and parse the datetime column
        dt_col = None
        for col in ("datetime", "timestamp", "date", "time", "t"):
            if col in combined.columns:
                dt_col = col
                break

        if dt_col is None:
            # Try to use index if it looks like a datetime
            if hasattr(combined.index, "to_pydatetime"):
                combined = combined.reset_index()
                combined.rename(columns={"index": "datetime"}, inplace=True)
                dt_col = "datetime"
            else:
                raise RuntimeError(
                    f"Cannot find datetime column. Columns found: {list(combined.columns)}"
                )

        combined[dt_col] = pd.to_datetime(
            combined[dt_col], utc=True, errors="coerce")
        combined.dropna(subset=[dt_col], inplace=True)
        combined.rename(columns={dt_col: "datetime"}, inplace=True)

        # Keep only OHLCV
        needed = ["datetime", "open", "high", "low", "close", "volume"]
        available = [c for c in needed if c in combined.columns]
        combined = combined[available].copy()

        for col in ("open", "high", "low", "close"):
            combined[col] = pd.to_numeric(combined[col], errors="coerce")
        if "volume" in combined.columns:
            combined["volume"] = pd.to_numeric(
                combined["volume"], errors="coerce").fillna(0)
        else:
            combined["volume"] = 0

        combined.dropna(subset=["open", "high", "low", "close"], inplace=True)

        # Deduplicate and sort
        combined.drop_duplicates(subset=["datetime"], inplace=True)
        combined.sort_values("datetime", inplace=True)
        combined.reset_index(drop=True, inplace=True)

        # Format datetime as string for CSV (replay app accepts both ISO and space-separated)
        combined["datetime"] = combined["datetime"].dt.strftime(
            "%Y-%m-%d %H:%M:%S")

        return combined


# ── SAVE ──────────────────────────────────────────────────────────────────────
def save_csv(df: pd.DataFrame, symbol: str, tf: int) -> str:
    start = df["datetime"].iloc[0].replace(" ", "T")[:10].replace("-", "")
    end = df["datetime"].iloc[-1].replace(" ", "T")[:10].replace("-", "")
    fname = f"nq_{tf}min_{start}_{end}.csv"
    df.to_csv(fname, index=False)
    return fname


# ── ENTRY POINT ───────────────────────────────────────────────────────────────
def main():
    load_dotenv()

    # Quick credential check
    if not os.getenv("PROJECT_X_API_KEY") or not os.getenv("PROJECT_X_USERNAME"):
        print("ERROR: Missing PROJECT_X_API_KEY or PROJECT_X_USERNAME in .env")
        print("Your .env file should have:")
        print("  PROJECT_X_API_KEY=your_key_here")
        print("  PROJECT_X_USERNAME=your_username_here")
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="Fetch TopstepX NQ history for market replay")
    parser.add_argument("--symbol", default="MNQ",
                        help="Contract symbol (default: MNQ)")
    parser.add_argument("--tf",     default=5,  type=int,
                        help="Timeframe in minutes (default: 5)")
    parser.add_argument("--days",   default=30, type=int,
                        help="Days of history to fetch (default: 30)")
    args = parser.parse_args()

    print("=" * 55)
    print("  TopstepX → Market Replay CSV Exporter")
    print("=" * 55)
    print(f"  Symbol    : {args.symbol}")
    print(f"  Timeframe : {args.tf}-min bars")
    print(f"  History   : last {args.days} days")
    print("=" * 55)

    try:
        df = asyncio.run(fetch_history(args.symbol, args.tf, args.days))
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        sys.exit(1)

    fname = save_csv(df, args.symbol, args.tf)

    print(f"\n{'=' * 55}")
    print(f"  Done!")
    print(f"  Bars fetched : {len(df):,}")
    print(
        f"  Date range   : {df['datetime'].iloc[0]} → {df['datetime'].iloc[-1]}")
    print(f"  Saved to     : {fname}")
    print(f"{'=' * 55}")
    print(f"\nNext step: Upload '{fname}' to the Market Replay app.")


if __name__ == "__main__":
    main()
