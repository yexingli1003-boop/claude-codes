# live_trader_ema_3min.py — 3-Minute EMA Cross Pullback Live Trader
#
# STANDALONE live trader for the EMA Cross strategy ONLY.
# Runs independently from live_trader.py (15-min) and live_trader_5min.py.
#
# USAGE:
#   python live_trader_ema_3min.py              (live mode)
#   python live_trader_ema_3min.py --dry-run    (signals only, no orders)
#
# REQUIREMENTS:
#   - .env file with PROJECT_X_API_KEY and PROJECT_X_USERNAME
#   - project-x-py SDK installed
#   - strategy_ema_cross_3min.py in same folder
#   - indicators.py in same folder (reuses your existing indicator code)

import asyncio
import sys
import os
import logging
from datetime import datetime, time as dtime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
from dotenv import load_dotenv
from project_x_py import ProjectX

# Reuse your existing indicators
from indicators import build_indicators

# Import the strategy
from strategy_ema_cross_3min import (
    signal_ema_cross_long,
    signal_ema_cross_short,
    STRATEGY_CONFIG,
)


# ==================================================
# CONFIG
# ==================================================

ACCOUNT_SIZE = 50_000
DAILY_LOSS_LIMIT_DOLLARS = 1_000
DAILY_LOSS_LIMIT_POINTS = DAILY_LOSS_LIMIT_DOLLARS / 2   # MNQ = $2/pt -> 500 pts
SAFETY_BUFFER_POINTS = 50
CONTRACTS = 3

TIMEZONE = ZoneInfo("America/Chicago")
HARD_CUTOFF_TIME = dtime(15, 5)      # 3:05 PM CT
NO_NEW_TRADES_TIME = dtime(14, 45)   # 2:45 PM CT
MARKET_OPEN_TIME = dtime(8, 30)
CANDLE_INTERVAL_MINUTES = 3

SYMBOL = "MNQ"
HISTORY_BARS = 200

TP_POINTS = 60
INITIAL_SL_BUFFER = 10       # SL starts 10 pts beyond EMA21
TRAIL_TRIGGER_POINTS = 15    # Move to breakeven at +15 pts
TRAIL_BUFFER = 2             # Trail SL this many pts beyond EMA21 after breakeven


# ==================================================
# LOGGING
# ==================================================

def setup_logging(dry_run=False):
    mode = "DRY-RUN" if dry_run else "LIVE"
    log_filename = f"ema_trades_{datetime.now().strftime('%Y%m%d')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format=f"[EMA3M-{mode}] %(asctime)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_filename, mode="a"),
        ],
    )
    return logging.getLogger("EMATrader")


# ==================================================
# ACTIVE TRADE WITH CUSTOM SL
# ==================================================

class EMAActiveTrade:
    def __init__(self, strategy, direction, entry_price, ema21_at_entry, tp, order_id=None):
        self.strategy = strategy
        self.direction = direction
        self.entry_price = entry_price
        self.tp_price = (
            entry_price + tp) if direction == "LONG" else (entry_price - tp)
        self.ema21_at_entry = ema21_at_entry
        self.order_id = order_id
        self.bars_held = 0
        self.moved_to_breakeven = False

        # Initial SL: 10 pts beyond EMA21
        if direction == "LONG":
            self.sl_price = ema21_at_entry - INITIAL_SL_BUFFER
            # Ensure SL is below entry
            if self.sl_price >= entry_price:
                self.sl_price = entry_price - INITIAL_SL_BUFFER
        else:
            self.sl_price = ema21_at_entry + INITIAL_SL_BUFFER
            if self.sl_price <= entry_price:
                self.sl_price = entry_price + INITIAL_SL_BUFFER

    def update(self, current_price, current_ema21):
        """Called each new bar. Updates SL according to strategy rules."""
        self.bars_held += 1

        if self.direction == "LONG":
            favorable = current_price - self.entry_price
        else:
            favorable = self.entry_price - current_price

        # Phase 1: Move to breakeven at +15 pts
        if not self.moved_to_breakeven and favorable >= TRAIL_TRIGGER_POINTS:
            if self.direction == "LONG":
                new_sl = self.entry_price
                if new_sl > self.sl_price:
                    self.sl_price = new_sl
                    self.moved_to_breakeven = True
                    return "moved_to_breakeven"
            else:
                new_sl = self.entry_price
                if new_sl < self.sl_price:
                    self.sl_price = new_sl
                    self.moved_to_breakeven = True
                    return "moved_to_breakeven"

        # Phase 2: After breakeven, trail along EMA21
        if self.moved_to_breakeven and not pd.isna(current_ema21):
            if self.direction == "LONG":
                trailing_sl = current_ema21 - TRAIL_BUFFER
                # Only ratchet up, never down
                if trailing_sl > self.sl_price:
                    self.sl_price = trailing_sl
                    return "trailed"
            else:
                trailing_sl = current_ema21 + TRAIL_BUFFER
                if trailing_sl < self.sl_price:
                    self.sl_price = trailing_sl
                    return "trailed"

        return None

    def check_exit(self, high, low):
        """Return (exit_reason, exit_price) if hit, else (None, None)."""
        if self.direction == "LONG":
            if low <= self.sl_price:
                reason = "TRAIL/BE" if self.moved_to_breakeven else "SL"
                return reason, self.sl_price
            if high >= self.tp_price:
                return "TP", self.tp_price
        else:
            if high >= self.sl_price:
                reason = "TRAIL/BE" if self.moved_to_breakeven else "SL"
                return reason, self.sl_price
            if low <= self.tp_price:
                return "TP", self.tp_price
        return None, None


# ==================================================
# MAIN TRADER
# ==================================================

class EMATrader:
    def __init__(self, dry_run=False):
        self.dry_run = dry_run
        self.log = setup_logging(dry_run)

        load_dotenv()
        self.client = None
        self.contract_id = None

        self.active_trades = {name: None for name in STRATEGY_CONFIG}
        self.daily_pnl_points = 0.0
        self.daily_trades = []
        self.trading_halted = False
        self.last_candle_time = None

    async def connect(self):
        self.client = ProjectX.from_env()
        await self.client.__aenter__()
        await self.client.authenticate()

        acc = self.client.account_info
        self.log.info(f"Account: {acc.name}")
        self.log.info(f"Balance: ${acc.balance:,.2f}")

        # Find MNQ contract
        contracts = await self.client.search_contracts(SYMBOL)
        if not contracts:
            raise ValueError(f"No contracts found for {SYMBOL}")

        # Filter for active front month
        active = [c for c in contracts if SYMBOL in c.name.upper()
                  and "MNQ" in c.name.upper()]
        if not active:
            active = contracts

        contract = active[0]
        self.contract_id = contract.id
        self.log.info(f"Contract: {contract.name} (ID: {self.contract_id})")

    async def disconnect(self):
        if self.client:
            await self.client.__aexit__(None, None, None)

    async def fetch_candles(self):
        """Fetch 3-min MNQ bars and compute indicators."""
        try:
            data = await self.client.get_bars(
                SYMBOL, days=2, interval=CANDLE_INTERVAL_MINUTES
            )

            if data is None or len(data) == 0:
                return None

            # Convert to pandas DataFrame if needed
            if hasattr(data, "to_pandas"):
                df = data.to_pandas()
            else:
                df = pd.DataFrame(data)

            # Normalize column names
            df.columns = [c.lower() for c in df.columns]

            # Try common datetime column names
            for dt_col in ["t", "timestamp", "datetime", "time", "date"]:
                if dt_col in df.columns:
                    df = df.rename(columns={dt_col: "datetime"})
                    break

            if "datetime" not in df.columns:
                # Use index as datetime
                df["datetime"] = df.index

            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.sort_values("datetime").reset_index(drop=True)

            # Ensure required columns exist
            required = ["open", "high", "low", "close", "volume"]
            for col in required:
                if col not in df.columns:
                    self.log.error(f"Missing column: {col}")
                    return None

            # Build indicators (EMA9, EMA21, RSI, ATR, candle metrics)
            df = build_indicators(df)

            return df

        except Exception as e:
            self.log.error(f"Error fetching candles: {e}")
            return None

    def evaluate_signals(self, df):
        """Check each strategy for a signal on the latest bar."""
        signals = []
        i = len(df) - 1

        for name, cfg in STRATEGY_CONFIG.items():
            if self.active_trades[name] is not None:
                continue

            fn = signal_ema_cross_long if name == "S_EMA_LONG" else signal_ema_cross_short

            try:
                if fn(df, i):
                    signals.append({
                        "strategy": name,
                        "direction": cfg["direction"],
                        "tp": cfg["tp"],
                    })
            except Exception as e:
                self.log.error(f"Error in {name}: {e}")

        return signals

    async def place_order(self, direction, entry_price):
        """Place market order. Returns order ID or None."""
        side = "BUY" if direction == "LONG" else "SELL"

        self.log.info(
            f"  PLACE: {side} {CONTRACTS} {SYMBOL} @ ~{entry_price:.2f}"
        )

        if self.dry_run:
            return f"DRY-{direction}-{datetime.now().strftime('%H%M%S')}"

        try:
            # Market order (syntax may vary with SDK)
            order = await self.client.place_order(
                contract_id=self.contract_id,
                side=side,
                size=CONTRACTS,
                order_type="MARKET",
            )
            self.log.info(f"  Order placed: {order.id}")
            return order.id
        except Exception as e:
            self.log.error(f"  ORDER FAILED: {e}")
            return None

    async def close_position(self, trade, reason, exit_price):
        """Close an open position."""
        close_side = "SELL" if trade.direction == "LONG" else "BUY"

        pnl_points = (exit_price - trade.entry_price) if trade.direction == "LONG" else (
            trade.entry_price - exit_price)
        pnl_dollars = pnl_points * 2 * CONTRACTS  # MNQ = $2/pt

        self.log.info(
            f"  CLOSE: {close_side} {CONTRACTS} {SYMBOL} @ {exit_price:.2f} | "
            f"{reason} | PnL: {pnl_points:+.1f}pts (${pnl_dollars:+,.0f})"
        )

        if not self.dry_run:
            try:
                await self.client.place_order(
                    contract_id=self.contract_id,
                    side=close_side,
                    size=CONTRACTS,
                    order_type="MARKET",
                )
            except Exception as e:
                self.log.error(f"  CLOSE FAILED: {e}")

        self.daily_pnl_points += pnl_points
        self.daily_trades.append({
            "strategy": trade.strategy,
            "direction": trade.direction,
            "entry": trade.entry_price,
            "exit": exit_price,
            "pnl_points": pnl_points,
            "pnl_dollars": pnl_dollars,
            "reason": reason,
            "bars": trade.bars_held,
        })

        self.log.info(f"  Daily PnL: {self.daily_pnl_points:+.1f}pts")
        self.active_trades[trade.strategy] = None

    async def manage_trades(self, df):
        """Check and update all active trades."""
        if df is None or len(df) == 0:
            return

        latest = df.iloc[-1]
        ema21_now = latest["ema_21"]

        for name, trade in list(self.active_trades.items()):
            if trade is None:
                continue

            # Update SL based on new bar
            update_msg = trade.update(float(latest["close"]), float(
                ema21_now) if not pd.isna(ema21_now) else None)
            if update_msg:
                self.log.info(
                    f"  {name}: {update_msg} | New SL: {trade.sl_price:.2f}")

            # Check for exits using current bar's high/low
            reason, exit_price = trade.check_exit(
                float(latest["high"]), float(latest["low"]))
            if reason:
                await self.close_position(trade, reason, exit_price)
            elif trade.bars_held >= 80:  # timeout after 80 * 3min = 4 hours
                await self.close_position(trade, "TIMEOUT", float(latest["close"]))

    async def flatten_all(self, reason):
        for name, trade in list(self.active_trades.items()):
            if trade is not None:
                self.log.warning(f"FLATTEN {name}: {reason}")
                # Use current market price estimate
                await self.close_position(trade, reason, trade.entry_price)

    def print_summary(self):
        self.log.info("=" * 70)
        self.log.info(f"  DAILY SUMMARY - EMA 3-MIN STRATEGY")
        self.log.info("=" * 70)
        self.log.info(f"  Trades: {len(self.daily_trades)}")
        total_dollars = self.daily_pnl_points * 2 * CONTRACTS
        self.log.info(
            f"  Total PnL: {self.daily_pnl_points:+.1f} pts (${total_dollars:+,.0f})")

        if self.daily_trades:
            wins = [t for t in self.daily_trades if t["pnl_points"] > 0]
            wr = len(wins) / len(self.daily_trades) * 100
            self.log.info(f"  Win Rate: {wr:.1f}%")

            for t in self.daily_trades:
                self.log.info(
                    f"    {t['strategy']} {t['direction']} | "
                    f"{t['entry']:.2f} -> {t['exit']:.2f} | "
                    f"{t['pnl_points']:+.1f}pts | {t['reason']}"
                )
        self.log.info("=" * 70)

    async def run(self):
        self.log.info("=" * 70)
        self.log.info("  EMA 3-MIN TRADER STARTING")
        self.log.info(f"  Mode: {'DRY-RUN' if self.dry_run else 'LIVE'}")
        self.log.info(f"  Symbol: {SYMBOL} | Contracts: {CONTRACTS}")
        self.log.info(f"  Timeframe: {CANDLE_INTERVAL_MINUTES}-min")
        self.log.info(
            f"  TP: {TP_POINTS}pts | Initial SL: EMA21 - {INITIAL_SL_BUFFER}pts")
        self.log.info(f"  Breakeven trigger: +{TRAIL_TRIGGER_POINTS}pts")
        self.log.info(
            f"  Daily loss limit: {DAILY_LOSS_LIMIT_POINTS}pts (${DAILY_LOSS_LIMIT_DOLLARS})")
        self.log.info("=" * 70)

        await self.connect()

        while True:
            try:
                now = datetime.now(TIMEZONE)
                ct = now.time()

                # Hard cutoff
                if ct >= HARD_CUTOFF_TIME:
                    active = [t for t in self.active_trades.values() if t]
                    if active:
                        self.log.warning("HARD CUTOFF - Flattening all.")
                        await self.flatten_all("CUTOFF")
                    self.print_summary()
                    self.log.info("Sleeping until next day...")
                    await asyncio.sleep(3600)
                    continue

                # Pre-market
                if ct < MARKET_OPEN_TIME:
                    self.log.info("Pre-market. Waiting for 08:30 AM CT...")
                    await asyncio.sleep(120)
                    continue

                # Daily loss check
                if abs(self.daily_pnl_points) >= (DAILY_LOSS_LIMIT_POINTS - SAFETY_BUFFER_POINTS):
                    if self.daily_pnl_points < 0 and not self.trading_halted:
                        self.log.warning(
                            f"LOSS LIMIT: {self.daily_pnl_points:+.1f}pts. Halting.")
                        self.trading_halted = True
                        await self.flatten_all("LOSS LIMIT")
                    if self.trading_halted:
                        await asyncio.sleep(60)
                        continue

                # Fetch data
                df = await self.fetch_candles()
                if df is None or len(df) < 50:
                    self.log.warning("Insufficient data. Waiting 30s...")
                    await asyncio.sleep(30)
                    continue

                # Dedup candles
                latest_time = str(df.iloc[-1]["datetime"])
                if latest_time != self.last_candle_time:
                    self.last_candle_time = latest_time
                    self.log.info(
                        f"New candle: {latest_time} | Close: {df.iloc[-1]['close']:.2f}")

                    # Manage existing trades first
                    await self.manage_trades(df)

                    # No new trades near close
                    if ct >= NO_NEW_TRADES_TIME:
                        self.log.info("Near close - exits only.")
                    else:
                        # Evaluate new signals
                        signals = self.evaluate_signals(df)
                        for s in signals:
                            self.log.info(
                                f"SIGNAL: {s['strategy']} | {s['direction']} | "
                                f"Price: {df.iloc[-1]['close']:.2f} | "
                                f"EMA21: {df.iloc[-1]['ema_21']:.2f}"
                            )

                            entry_price = float(df.iloc[-1]["close"])
                            ema21_entry = float(df.iloc[-1]["ema_21"])

                            order_id = await self.place_order(s["direction"], entry_price)
                            if order_id:
                                trade = EMAActiveTrade(
                                    strategy=s["strategy"],
                                    direction=s["direction"],
                                    entry_price=entry_price,
                                    ema21_at_entry=ema21_entry,
                                    tp=s["tp"],
                                    order_id=order_id,
                                )
                                self.active_trades[s["strategy"]] = trade
                                self.log.info(
                                    f"  Trade opened: Entry={trade.entry_price:.2f} "
                                    f"TP={trade.tp_price:.2f} SL={trade.sl_price:.2f}"
                                )

                # Poll every 20s (3-min candles = check frequently)
                await asyncio.sleep(20)

            except KeyboardInterrupt:
                self.log.info("Shutdown by user.")
                await self.flatten_all("USER SHUTDOWN")
                break
            except Exception as e:
                self.log.error(f"Loop error: {e}")
                await asyncio.sleep(30)

        await self.disconnect()


# ==================================================
# ENTRY POINT
# ==================================================

async def main_async():
    dry_run = "--dry-run" in sys.argv

    if not dry_run:
        print("\n" + "!" * 60)
        print("  LIVE TRADING MODE - Real orders on TopstepX!")
        print("  Use --dry-run flag to test without placing orders.")
        print("!" * 60)
        confirm = input("\nType 'YES' to confirm live trading: ")
        if confirm.strip() != "YES":
            print("Aborted.")
            return

    trader = EMATrader(dry_run=dry_run)
    await trader.run()


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
