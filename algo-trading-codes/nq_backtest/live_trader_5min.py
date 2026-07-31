# live_trader_5min.py — 5-Minute NQ Algo Trading Engine for TopstepX
#
# This script connects to your TopstepX account via the ProjectX API,
# fetches 5-minute MNQ candle data, evaluates ONLY the 5-min strategy (S8),
# and places bracket orders with TP/SL/trailing stop management.
#
# USAGE:
#   python live_trader_5min.py              (live mode)
#   python live_trader_5min.py --dry-run    (paper mode — signals only, no orders)
#
# Runs independently of live_trader.py — you can run both at the same time
# in separate terminals for 15-min + 5-min combined coverage.

import asyncio
import sys
import os
import logging
from datetime import datetime, time as dtime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
from dotenv import load_dotenv
from project_x_py import ProjectX

# Your existing strategy code — no changes needed
from config import COMMISSION
from indicators import build_indicators
from strategies import STRATEGIES as ALL_STRATEGIES

# Filter: only run S8 on 5-min timeframe
STRATEGIES = {name: cfg for name, cfg in ALL_STRATEGIES.items()
              if name in ("S8_BullDivRSI",)}

# ============================================================
# CONFIGURATION
# ============================================================

# Account settings
ACCOUNT_SIZE = 50_000          # Topstep account size in dollars
DAILY_LOSS_LIMIT_DOLLARS = 1_000  # $50K account daily loss limit
DAILY_LOSS_LIMIT_POINTS = DAILY_LOSS_LIMIT_DOLLARS / 2  # MNQ = $2/pt, so 500 pts
SAFETY_BUFFER_POINTS = 50     # Stop trading when within 50 pts of limit
CONTRACTS = 3                  # Number of MNQ contracts per trade

# Timing (Central Time — Chicago)
TIMEZONE = ZoneInfo("America/Chicago")
HARD_CUTOFF_TIME = dtime(15, 5)   # Flatten everything at 3:05 PM CT
NO_NEW_TRADES_TIME = dtime(14, 45)  # No new trades after 2:45 PM CT
MARKET_OPEN_TIME = dtime(8, 30)    # Only trade during active hours
CANDLE_INTERVAL_MINUTES = 5        # 5-minute bars

# Instrument
SYMBOL = "MNQ"   # Micro E-Mini NASDAQ 100 Futures

# Trailing stop
TRAIL_ACTIVATE_MULTIPLIER = 1.0   # Activate when move > 1x ATR
TRAIL_OFFSET_MULTIPLIER = 1.5     # Trail at best - 1.5x ATR

# How many candles of history to fetch for indicator calculation
HISTORY_BARS = 200

# ============================================================
# LOGGING SETUP
# ============================================================


def setup_logging(dry_run: bool = False):
    mode = "DRY-RUN" if dry_run else "LIVE"
    log_filename = f"trades_{datetime.now().strftime('%Y%m%d')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format=f"[{mode}] %(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_filename, mode="a"),
        ],
    )
    return logging.getLogger("LiveTrader")


# ============================================================
# ACTIVE TRADE TRACKER
# ============================================================

class ActiveTrade:
    """Tracks a single active trade with its TP/SL/trailing stop state."""

    def __init__(self, strategy_name, direction, entry_price, tp_points, sl_points, atr_at_entry, order_id=None):
        self.strategy_name = strategy_name
        self.direction = direction
        self.entry_price = entry_price
        self.tp_points = tp_points
        self.sl_points = sl_points
        self.atr_at_entry = atr_at_entry
        self.order_id = order_id
        self.entry_time = datetime.now(TIMEZONE)

        # TP/SL levels
        if direction == "LONG":
            self.tp_price = entry_price + tp_points
            self.sl_price = entry_price - sl_points
        else:
            self.tp_price = entry_price - tp_points
            self.sl_price = entry_price + sl_points

        # Trailing stop state
        self.trail_level = self.sl_price
        self.best_price = entry_price
        self.trail_activated = False
        self.bars_held = 0

    def update_trailing_stop(self, candle_high, candle_low):
        """Update trailing stop based on new candle data. Returns new trail level."""
        self.bars_held += 1

        if self.direction == "LONG":
            if candle_high > self.best_price:
                self.best_price = candle_high
            favorable_move = self.best_price - self.entry_price
            if favorable_move > self.atr_at_entry * TRAIL_ACTIVATE_MULTIPLIER:
                new_trail = self.best_price - \
                    (TRAIL_OFFSET_MULTIPLIER * self.atr_at_entry)
                if new_trail > self.trail_level:
                    self.trail_level = new_trail
                    self.trail_activated = True
        else:  # SHORT
            if candle_low < self.best_price:
                self.best_price = candle_low
            favorable_move = self.entry_price - self.best_price
            if favorable_move > self.atr_at_entry * TRAIL_ACTIVATE_MULTIPLIER:
                new_trail = self.best_price + \
                    (TRAIL_OFFSET_MULTIPLIER * self.atr_at_entry)
                if new_trail < self.trail_level:
                    self.trail_level = new_trail
                    self.trail_activated = True

        return self.trail_level

    def check_exit(self, candle_high, candle_low):
        """Check if TP or SL/trail was hit. Returns (exit_triggered, exit_reason, exit_price) or (False, None, None)."""
        if self.direction == "LONG":
            # SL/trail first (worst case)
            if candle_low <= self.trail_level:
                reason = "TRAIL" if self.trail_activated else "SL"
                return True, reason, self.trail_level
            if candle_high >= self.tp_price:
                return True, "TP", self.tp_price
        else:  # SHORT
            if candle_high >= self.trail_level:
                reason = "TRAIL" if self.trail_activated else "SL"
                return True, reason, self.trail_level
            if candle_low <= self.tp_price:
                return True, "TP", self.tp_price

        return False, None, None

    def calc_pnl(self, exit_price):
        if self.direction == "LONG":
            return (exit_price - self.entry_price) - COMMISSION
        else:
            return (self.entry_price - exit_price) - COMMISSION


# ============================================================
# MAIN LIVE TRADER
# ============================================================

class LiveTrader:
    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.log = setup_logging(dry_run)
        self.client = None
        self.contract_id = None

        # Active trades per strategy (only 1 at a time per strategy)
        self.active_trades: dict[str, ActiveTrade | None] = {
            name: None for name in STRATEGIES}

        # Daily PnL tracking
        self.daily_pnl = 0.0
        self.daily_trades = []
        self.trading_halted = False

        # Track last processed candle to avoid duplicate signals
        self.last_candle_time = None

    async def connect(self):
        """Authenticate and find NQ contract."""
        self.log.info("Connecting to TopstepX...")
        self.client = await ProjectX.from_env().__aenter__()
        await self.client.authenticate()

        account = self.client.account_info
        self.log.info(f"Account: {account.name}")
        self.log.info(f"Balance: ${account.balance:,.2f}")

        # Find NQ contract
        instruments = await self.client.search_instruments(SYMBOL)
        if instruments:
            self.contract_id = instruments[0].id
            self.log.info(
                f"Found contract: {instruments[0].name} (ID: {self.contract_id})")
        else:
            raise RuntimeError(f"Could not find {SYMBOL} contract")

    async def fetch_candles(self) -> pd.DataFrame:
        """Fetch NQ 15-min candles and return as pandas DataFrame with indicators."""
        try:
            data = await self.client.get_bars(SYMBOL, days=5, interval=CANDLE_INTERVAL_MINUTES)

            # Convert to pandas if needed (SDK may return Polars)
            if hasattr(data, 'to_pandas'):
                df = data.to_pandas()
            elif isinstance(data, pd.DataFrame):
                df = data.copy()
            else:
                df = pd.DataFrame(data)

            # Standardize column names to lowercase
            df.columns = df.columns.str.lower().str.strip()

            # Ensure required columns exist
            required = {"open", "high", "low", "close", "volume"}
            if not required.issubset(set(df.columns)):
                self.log.error(f"Missing columns. Have: {list(df.columns)}")
                return None

            # Parse datetime if present
            for col in ("datetime", "date", "time", "timestamp"):
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])
                    break

            # Sort chronologically
            df = df.sort_index() if df.index.dtype == 'datetime64[ns]' else df.sort_values(
                next((c for c in ("datetime", "date", "timestamp")
                     if c in df.columns), df.columns[0])
            )

            # Build indicators using your existing code
            df = build_indicators(df)
            df = df.reset_index(drop=True)

            self.log.info(
                f"Fetched {len(df)} candles. Latest: {df.iloc[-1].get('datetime', 'N/A')}")
            return df

        except Exception as e:
            self.log.error(f"Error fetching candles: {e}")
            return None

    def evaluate_signals(self, df: pd.DataFrame) -> list:
        """Run all 5 strategies on the latest candle. Returns list of signals."""
        signals = []
        i = len(df) - 1  # Latest completed candle

        if i < 50:  # Need warmup
            return signals

        for name, cfg in STRATEGIES.items():
            # Skip if already in a trade for this strategy
            if self.active_trades[name] is not None:
                continue

            try:
                if cfg["signal_fn"](df, i):
                    signals.append({
                        "strategy": name,
                        "direction": cfg["direction"],
                        "tp": cfg["tp"],
                        "sl": cfg["sl"],
                        "candle_time": str(df.iloc[i].get("datetime", i)),
                        "close": df.iloc[i]["close"],
                        "atr": df.iloc[i]["atr"] if not pd.isna(df.iloc[i]["atr"]) else 15.0,
                    })
            except Exception as e:
                self.log.error(f"Error evaluating {name}: {e}")

        return signals

    async def place_order(self, signal: dict) -> str | None:
        """Place a bracket order via the API. Returns order ID or None."""
        direction = signal["direction"]
        side = "Buy" if direction == "LONG" else "Sell"

        self.log.info(
            f"PLACING ORDER: {signal['strategy']} | {side} {CONTRACTS} NQ "
            f"| TP={signal['tp']} SL={signal['sl']} | Price ~{signal['close']}"
        )

        if self.dry_run:
            self.log.info("  [DRY-RUN] Order simulated, not sent to exchange.")
            return f"DRY-{signal['strategy']}-{datetime.now().strftime('%H%M%S')}"

        try:
            # Place market order with bracket (TP + SL)
            # The exact method depends on SDK version — try the TradingSuite approach
            order = await self.client.place_order(
                contract_id=self.contract_id,
                side=side,
                qty=CONTRACTS,
                order_type="Market",
            )

            order_id = getattr(order, 'id', str(order))
            self.log.info(f"  Order placed successfully. ID: {order_id}")
            return order_id

        except Exception as e:
            self.log.error(f"  ORDER FAILED: {e}")
            return None

    async def close_position(self, trade: ActiveTrade, reason: str, exit_price: float = None):
        """Close an active position."""
        side = "Sell" if trade.direction == "LONG" else "Buy"

        self.log.info(
            f"CLOSING: {trade.strategy_name} | {side} {CONTRACTS} NQ | Reason: {reason}"
        )

        if not self.dry_run:
            try:
                await self.client.place_order(
                    contract_id=self.contract_id,
                    side=side,
                    qty=CONTRACTS,
                    order_type="Market",
                )
                self.log.info(f"  Close order sent successfully.")
            except Exception as e:
                self.log.error(f"  CLOSE ORDER FAILED: {e}")

        # Calculate PnL
        if exit_price:
            pnl = trade.calc_pnl(exit_price)
        else:
            pnl = 0.0

        self.daily_pnl += pnl
        self.daily_trades.append({
            "strategy": trade.strategy_name,
            "direction": trade.direction,
            "entry": trade.entry_price,
            "exit": exit_price,
            "pnl": pnl,
            "reason": reason,
            "bars_held": trade.bars_held,
            "time": datetime.now(TIMEZONE).strftime("%H:%M:%S"),
        })

        pnl_dollars = pnl * 20
        self.log.info(
            f"  PnL: {pnl:+.1f} pts (${pnl_dollars:+,.0f}) | "
            f"Daily total: {self.daily_pnl:+.1f} pts (${self.daily_pnl * 20:+,.0f})"
        )

        # Clear the active trade
        self.active_trades[trade.strategy_name] = None

    async def manage_active_trades(self, df: pd.DataFrame):
        """Check trailing stops and TP/SL on all active trades."""
        if df is None or len(df) < 2:
            return

        latest = df.iloc[-1]

        for name, trade in self.active_trades.items():
            if trade is None:
                continue

            # Update trailing stop
            trade.update_trailing_stop(latest["high"], latest["low"])

            # Check exit conditions
            hit, reason, exit_price = trade.check_exit(
                latest["high"], latest["low"])

            if hit:
                await self.close_position(trade, reason, exit_price)
            elif trade.bars_held >= 40:
                await self.close_position(trade, "TIMEOUT", latest["close"])

    async def flatten_all(self, reason: str = "CUTOFF"):
        """Close ALL active positions immediately."""
        for name, trade in self.active_trades.items():
            if trade is not None:
                self.log.warning(f"FLATTENING {name}: {reason}")
                await self.close_position(trade, reason)

    def check_daily_limit(self) -> bool:
        """Returns True if we should stop trading due to daily loss limit."""
        limit = DAILY_LOSS_LIMIT_POINTS - SAFETY_BUFFER_POINTS
        if self.daily_pnl <= -limit:
            return True
        return False

    def get_ct_now(self) -> datetime:
        """Get current time in Central Time."""
        return datetime.now(TIMEZONE)

    async def run(self):
        """Main trading loop."""
        self.log.info("=" * 70)
        self.log.info("  NQ ALGO TRADER — STARTING UP")
        self.log.info(
            f"  Mode: {'DRY-RUN (no real orders)' if self.dry_run else 'LIVE TRADING'}")
        self.log.info(f"  Account: $50,000 | Contracts: {CONTRACTS} NQ")
        self.log.info(
            f"  Daily loss limit: {DAILY_LOSS_LIMIT_POINTS} pts (${DAILY_LOSS_LIMIT_DOLLARS})")
        self.log.info(
            f"  Hard cutoff: {HARD_CUTOFF_TIME.strftime('%I:%M %p')} CT")
        self.log.info(f"  Strategies: {', '.join(STRATEGIES.keys())}")
        self.log.info("=" * 70)

        await self.connect()

        while True:
            try:
                now = self.get_ct_now()
                current_time = now.time()

                # ── DAILY RESET (check if new trading day) ──
                # Reset daily PnL at midnight CT
                if current_time < dtime(0, 5):
                    if self.daily_pnl != 0:
                        self.log.info("NEW TRADING DAY — Resetting daily PnL")
                        self.daily_pnl = 0.0
                        self.daily_trades = []
                        self.trading_halted = False

                # ── HARD CUTOFF: Flatten at 3:05 PM CT ──
                if current_time >= HARD_CUTOFF_TIME:
                    has_active = any(
                        t is not None for t in self.active_trades.values())
                    if has_active:
                        self.log.warning(
                            "HARD CUTOFF — Flattening all positions!")
                        await self.flatten_all("3:05 PM CUTOFF")

                    # Sleep until next session
                    self.log.info(
                        "Market closed. Sleeping until next session...")
                    await asyncio.sleep(300)  # Check every 5 min
                    continue

                # ── DAILY LOSS LIMIT CHECK ──
                if self.check_daily_limit():
                    if not self.trading_halted:
                        self.log.warning(
                            f"DAILY LOSS LIMIT APPROACHING: {self.daily_pnl:+.1f} pts. "
                            f"Halting new trades for today."
                        )
                        self.trading_halted = True
                        await self.flatten_all("DAILY LOSS LIMIT")
                    await asyncio.sleep(60)
                    continue

                # ── PRE-MARKET: Wait for market open ──
                if current_time < MARKET_OPEN_TIME:
                    self.log.info(
                        f"Pre-market. Waiting for {MARKET_OPEN_TIME.strftime('%I:%M %p')} CT...")
                    await asyncio.sleep(60)
                    continue

                # ── FETCH CANDLE DATA ──
                df = await self.fetch_candles()
                if df is None or len(df) < 50:
                    self.log.warning("Insufficient data. Retrying in 60s...")
                    await asyncio.sleep(60)
                    continue

                # Check if this is a new candle (avoid duplicate signals)
                latest_time = str(df.iloc[-1].get("datetime", len(df)))
                if latest_time == self.last_candle_time:
                    # Same candle, just manage existing trades
                    await self.manage_active_trades(df)
                    # Check more frequently for trail updates
                    await asyncio.sleep(30)
                    continue

                self.last_candle_time = latest_time
                self.log.info(f"New candle: {latest_time}")

                # ── MANAGE EXISTING TRADES ──
                await self.manage_active_trades(df)

                # ── NO NEW TRADES NEAR CLOSE ──
                if current_time >= NO_NEW_TRADES_TIME:
                    self.log.info(
                        "Past 2:45 PM CT — no new trades, managing exits only.")
                    await asyncio.sleep(30)
                    continue

                # ── EVALUATE SIGNALS ──
                signals = self.evaluate_signals(df)

                for signal in signals:
                    self.log.info(
                        f"SIGNAL: {signal['strategy']} | {signal['direction']} "
                        f"| Price: {signal['close']:.2f} | ATR: {signal['atr']:.1f}"
                    )

                    # Place the order
                    order_id = await self.place_order(signal)

                    if order_id:
                        # Track the active trade
                        # Approximate; real fill may differ
                        entry_price = signal["close"]
                        trade = ActiveTrade(
                            strategy_name=signal["strategy"],
                            direction=signal["direction"],
                            entry_price=entry_price,
                            tp_points=signal["tp"],
                            sl_points=signal["sl"],
                            atr_at_entry=signal["atr"],
                            order_id=order_id,
                        )
                        self.active_trades[signal["strategy"]] = trade

                        self.log.info(
                            f"  Trade opened: Entry={entry_price:.2f} "
                            f"TP={trade.tp_price:.2f} SL={trade.sl_price:.2f}"
                        )

                # ── WAIT FOR NEXT CANDLE ──
                # Sleep ~1 minute, then re-check. The candle dedup logic
                # prevents duplicate signals even if we check frequently.
                await asyncio.sleep(60)

            except KeyboardInterrupt:
                self.log.info("Shutdown requested by user (Ctrl+C)")
                await self.flatten_all("USER SHUTDOWN")
                break

            except Exception as e:
                self.log.error(
                    f"Unexpected error in main loop: {e}", exc_info=True)
                await asyncio.sleep(30)  # Brief pause before retry

        # ── SHUTDOWN ──
        self.print_daily_summary()
        if self.client:
            await self.client.__aexit__(None, None, None)

    def print_daily_summary(self):
        """Print end-of-day trade summary."""
        self.log.info("\n" + "=" * 70)
        self.log.info("  DAILY TRADE SUMMARY")
        self.log.info("=" * 70)
        self.log.info(f"  Total trades: {len(self.daily_trades)}")
        self.log.info(
            f"  Total PnL: {self.daily_pnl:+.1f} pts (${self.daily_pnl * 20:+,.0f})")

        if self.daily_trades:
            wins = [t for t in self.daily_trades if t["pnl"] > 0]
            losses = [t for t in self.daily_trades if t["pnl"] <= 0]
            wr = len(wins) / len(self.daily_trades) * 100
            self.log.info(
                f"  Win rate: {wr:.1f}% ({len(wins)}W / {len(losses)}L)")
            self.log.info("")
            self.log.info(
                f"  {'Strategy':<22} {'Dir':<6} {'Entry':>9} {'Exit':>9} {'PnL':>8} {'Reason':<8} {'Time'}")
            self.log.info(f"  {'-'*80}")
            for t in self.daily_trades:
                entry = f"{t['entry']:.2f}" if t['entry'] else "N/A"
                exit_p = f"{t['exit']:.2f}" if t['exit'] else "N/A"
                self.log.info(
                    f"  {t['strategy']:<22} {t['direction']:<6} {entry:>9} {exit_p:>9} "
                    f"{t['pnl']:>+7.1f} {t['reason']:<8} {t['time']}"
                )

        self.log.info("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

def main():
    load_dotenv()

    dry_run = "--dry-run" in sys.argv

    if not dry_run:
        print("\n" + "!" * 60)
        print("  WARNING: LIVE TRADING MODE")
        print("  Real orders will be placed on your TopstepX account!")
        print("  Use --dry-run flag to test without placing orders.")
        print("!" * 60)
        confirm = input("\nType 'YES' to confirm live trading: ")
        if confirm.strip() != "YES":
            print("Aborted. Use: python live_trader.py --dry-run")
            return

    trader = LiveTrader(dry_run=dry_run)
    asyncio.run(trader.run())


if __name__ == "__main__":
    main()
