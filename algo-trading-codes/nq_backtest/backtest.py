# backtest.py — Main orchestration: load data, run all strategies, report results

import sys
import os
import pandas as pd

from config import COMMISSION, WARMUP_CANDLES
from indicators import build_indicators
from strategies import STRATEGIES
from simulator import simulate_trade


def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.lower().str.strip()

    # Try to find and parse a datetime column
    for col in ("datetime", "date", "time", "timestamp"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
            df = df.set_index(col)
            break

    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    df = df[["open", "high", "low", "close", "volume"]].copy()
    df = df.sort_index()
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"])
    return df


def compute_metrics(trades: list) -> dict:
    if not trades:
        return {
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "gross_profit": 0.0,
            "gross_loss": 0.0,
            "profit_factor": 0.0,
            "total_pnl": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
        }

    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = gross_profit / \
        gross_loss if gross_loss > 0 else float("inf")

    return {
        "total_trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": len(wins) / len(trades) * 100,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "profit_factor": profit_factor,
        "total_pnl": sum(pnls),
        "avg_win": sum(wins) / len(wins) if wins else 0.0,
        "avg_loss": sum(losses) / len(losses) if losses else 0.0,
    }


def print_report(all_trades: dict) -> None:
    strategy_labels = {
        "S1_BullFlagEMA":   "S1 Bull Flag + EMA",
        "S2_BullFlagBasic": "S2 Bull Flag Basic",
        "S3_Hammer":        "S3 Hammer",
        "S4_DeathCross":    "S4 Death Cross",
        "S5_RSIOversold":   "S5 RSI Oversold",
        "S6_DeltaFlip":     "S6 Delta Flip",
        "S7_BullDivEMA":    "S7 Bull Div + EMA",
    }

    print("\n" + "=" * 100)
    print("NQ E-MINI FUTURES — BACKTEST RESULTS (15-Min, All Strategies)")
    print("=" * 100)

    header = f"{'Strategy':<26} {'Trades':>7} {'Wins':>6} {'Losses':>7} {'WR%':>7} {'Avg Win':>9} {'Avg Loss':>10} {'PF':>6} {'Total PnL':>11}"
    print(header)
    print("-" * 100)

    all_pnls = []
    for name, trades in all_trades.items():
        m = compute_metrics(trades)
        all_pnls.extend(t["pnl"] for t in trades)
        label = strategy_labels.get(name, name)
        pf_str = f"{m['profit_factor']:.2f}" if m["profit_factor"] != float(
            "inf") else "  inf"
        print(
            f"{label:<26} {m['total_trades']:>7} {m['wins']:>6} {m['losses']:>7} "
            f"{m['win_rate']:>6.1f}% {m['avg_win']:>9.2f} {m['avg_loss']:>9.2f}  "
            f"{pf_str:>5} {m['total_pnl']:>10.1f}"
        )

    print("-" * 100)
    total_trades = sum(len(t) for t in all_trades.values())
    total_pnl = sum(all_pnls)
    total_wins = sum(1 for p in all_pnls if p > 0)
    wr = total_wins / total_trades * 100 if total_trades else 0
    gross_p = sum(p for p in all_pnls if p > 0)
    gross_l = abs(sum(p for p in all_pnls if p <= 0))
    pf = gross_p / gross_l if gross_l > 0 else float("inf")
    pf_str = f"{pf:.2f}" if pf != float("inf") else "  inf"
    print(
        f"{'TOTAL (all strategies)':<26} {total_trades:>7} {total_wins:>6} {total_trades - total_wins:>7} "
        f"{wr:>6.1f}% {'':>9} {'':>9}  {pf_str:>5} {total_pnl:>10.1f}"
    )
    print("=" * 100)
    print(
        f"\nNote: PnL in NQ points. 1 point = $20 per contract. Commission = {COMMISSION} pts/trade.")
    print(f"      Total PnL in dollars (1 contract): ${total_pnl * 20:,.0f}\n")

    # Per-strategy exit reason breakdown
    print("Exit Reason Breakdown:")
    print(f"{'Strategy':<26} {'TP':>6} {'SL':>6} {'TRAIL':>7} {'TIMEOUT':>9}")
    print("-" * 60)
    for name, trades in all_trades.items():
        label = strategy_labels.get(name, name)
        tp = sum(1 for t in trades if t["exit_reason"] == "TP")
        sl = sum(1 for t in trades if t["exit_reason"] == "SL")
        trail = sum(1 for t in trades if t["exit_reason"] == "TRAIL")
        timeout = sum(1 for t in trades if t["exit_reason"] == "TIMEOUT")
        print(f"{label:<26} {tp:>6} {sl:>6} {trail:>7} {timeout:>9}")
    print()


def run_backtest(filepath: str) -> None:
    print(f"Loading data from: {filepath}")
    df = load_data(filepath)
    print(f"Loaded {len(df)} candles. Range: {df.index[0]} → {df.index[-1]}")

    print("Computing indicators...")
    df = build_indicators(df)

    # Reset integer index for iloc-based access in strategies/simulator
    df = df.reset_index(drop=False)
    df.columns = [c if c != "index" else "datetime" for c in df.columns]

    print("Running backtest...\n")

    all_trades: dict = {name: [] for name in STRATEGIES}
    active_trade_end_bar: dict = {name: -1 for name in STRATEGIES}

    n = len(df)
    for i in range(WARMUP_CANDLES, n - 1):
        for name, cfg in STRATEGIES.items():
            # Skip if this strategy has an active trade
            if i <= active_trade_end_bar[name]:
                continue

            if not cfg["signal_fn"](df, i):
                continue

            entry_bar = i + 1
            if entry_bar >= n:
                continue

            # ATR from candle before entry
            atr_idx = entry_bar - 1
            atr_val = df.iloc[atr_idx]["atr"]
            if pd.isna(atr_val) or atr_val <= 0:
                continue

            result = simulate_trade(
                df=df,
                entry_bar_index=entry_bar,
                direction=cfg["direction"],
                tp_points=cfg["tp"],
                sl_points=cfg["sl"],
                atr_at_entry=atr_val,
            )

            all_trades[name].append(result)
            active_trade_end_bar[name] = entry_bar + result["bars_held"]

    print_report(all_trades)


if __name__ == "__main__":
    default_path = os.path.join(
        os.path.dirname(__file__), "data", "nq_15m.csv")
    filepath = sys.argv[1] if len(sys.argv) > 1 else default_path
    run_backtest(filepath)
