# live_trader_alpaca.py — QQQ Paper Trading Bot for Alpaca
#
# Runs your 5 NQ strategies on QQQ (NASDAQ 100 ETF) with scaled TP/SL.
# QQQ tracks the same NASDAQ 100 index as NQ futures.
#
# USAGE:
#   python live_trader_alpaca.py              (paper trading — places orders on Alpaca paper)
#   python live_trader_alpaca.py --dry-run    (signals only, no orders)
#
# SCALING LOGIC:
#   NQ ~25,000 pts → 60pt TP = 0.24% move
#   QQQ ~$607     → 0.24% = ~$1.46 TP per share
#   We trade 100 shares of QQQ ≈ similar dollar exposure to 1 NQ contract

import asyncio
import sys
import os
import time
import logging
from datetime import datetime, time as dtime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import numpy as np
from dotenv import load_dotenv

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# Your existing strategy code
from config import COMMISSION
from indicators import build_indicators
from strategies import STRATEGIES

# ============================================================
# CONFIGURATION
# ============================================================

# Alpaca paper trading
SYMBOL = "QQQ"
SHARES = 100  # ~$60,700 notional ≈ 1 NQ contract exposure

# Scaling: NQ points → QQQ dollars
# NQ is ~25,000, QQQ is ~$607. Ratio ≈ 607/25000 = 0.02428
# So NQ 60pts = QQQ $1.46, NQ 40pts = QQQ $0.97
# We calculate this dynamically from current QQQ price
NQ_REFERENCE_PRICE = 25000  # approximate NQ level for scaling

# Timing (Eastern Time for US stocks)
TIMEZONE = ZoneInfo("America/New_York")
MARKET_OPEN = dtime(9, 30)
MARKET_CLOSE = dtime(16, 0)
NO_NEW_TRADES_TIME = dtime(15, 45)  # Stop new trades 15 min before close
HARD_CUTOFF_TIME = dtime(15, 55)    # Flatten 5 min before close
CANDLE_INTERVAL = TimeFrame.Minute  # We'll aggregate to 15-min ourselves

# Risk management
DAILY_LOSS_LIMIT_DOLLARS = 1000  # Stop trading after $1K loss
SAFETY_BUFFER = 200              # Stop when within $200 of limit

# ATR trailing stop
TRAIL_ACTIVATE_MULT = 1.0
TRAIL_OFFSET_MULT = 1.5

# History
HISTORY_DAYS = 5

# ============================================================
# LOGGING
# ============================================================


def setup_logging(dry_run=False):
    mode = "DRY-RUN" if dry_run else "PAPER"
    log_file = f"alpaca_trades_{datetime.now().strftime('%Y%m%d')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format=f"[{mode}] %(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, mode="a"),
        ],
    )
    return logging.getLogger("AlpacaTrader")


# ============================================================
# PRICE SCALING
# ============================================================

def nq_points_to_qqq_dollars(nq_points, qqq_price):
    """Convert NQ point values to QQQ dollar equivalents."""
    ratio = qqq_price / NQ_REFERENCE_PRICE
    return round(nq_points * ratio, 2)


# ============================================================
# ACTIVE TRADE TRACKER
# ============================================================

class ActiveTrade:
    def __init__(self, strategy_name, direction, entry_price, tp_dollars, sl_dollars,
                 atr_value, order_id=None):
        self.strategy_name = strategy_name
        self.direction = direction
        self.entry_price = entry_price
        self.tp_dollars = tp_dollars
        self.sl_dollars = sl_dollars
        self.atr_value = atr_value
        self.order_id = order_id
        self.entry_time = datetime.now(ZoneInfo("America/New_York"))

        if direction == "LONG":
            self.tp_price = entry_price + tp_dollars
            self.sl_price = entry_price - sl_dollars
        else:
            self.tp_price = entry_price - tp_dollars
            self.sl_price = entry_price + sl_dollars

        self.trail_level = self.sl_price
        self.best_price = entry_price
        self.trail_activated = False
        self.bars_held = 0

    def update_trailing_stop(self, high, low):
        self.bars_held += 1

        if self.direction == "LONG":
            if high > self.best_price:
                self.best_price = high
            favorable = self.best_price - self.entry_price
            if favorable > self.atr_value * TRAIL_ACTIVATE_MULT:
                new_trail = self.best_price - \
                    (TRAIL_OFFSET_MULT * self.atr_value)
                if new_trail > self.trail_level:
                    self.trail_level = new_trail
                    self.trail_activated = True
        else:
            if low < self.best_price:
                self.best_price = low
            favorable = self.entry_price - self.best_price
            if favorable > self.atr_value * TRAIL_ACTIVATE_MULT:
                new_trail = self.best_price + \
                    (TRAIL_OFFSET_MULT * self.atr_value)
                if new_trail < self.trail_level:
                    self.trail_level = new_trail
                    self.trail_activated = True

    def check_exit(self, high, low):
        if self.direction == "LONG":
            if low <= self.trail_level:
                reason = "TRAIL" if self.trail_activated else "SL"
                return True, reason, self.trail_level
            if high >= self.tp_price:
                return True, "TP", self.tp_price
        else:
            if high >= self.trail_level:
                reason = "TRAIL" if self.trail_activated else "SL"
                return True, reason, self.trail_level
            if low <= self.tp_price:
                return True, "TP", self.tp_price
        return False, None, None

    def calc_pnl_dollars(self, exit_price):
        if self.direction == "LONG":
            return (exit_price - self.entry_price) * SHARES
        else:
            return (self.entry_price - exit_price) * SHARES


# ============================================================
# MAIN TRADER
# ============================================================

class AlpacaTrader:
    def __init__(self, dry_run=False):
        self.dry_run = dry_run
        self.log = setup_logging(dry_run)

        load_dotenv()
        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")

        if not api_key or not secret_key:
            raise ValueError(
                "ALPACA_API_KEY and ALPACA_SECRET_KEY must be in .env")

        self.trading_client = TradingClient(api_key, secret_key, paper=True)
        self.data_client = StockHistoricalDataClient(api_key, secret_key)

        self.active_trades = {name: None for name in STRATEGIES}
        self.daily_pnl = 0.0
        self.daily_trades = []
        self.trading_halted = False
        self.last_candle_time = None
        self.current_qqq_price = 607.0  # Will be updated from data

    def fetch_candles(self):
        """Fetch QQQ 15-min bars and compute indicators."""
        try:
            end = datetime.now(ZoneInfo("America/New_York"))
            start = end - timedelta(days=HISTORY_DAYS)

            # Fetch 1-min bars with IEX feed (free tier) and aggregate to 15-min
            request = StockBarsRequest(
                symbol_or_symbols=SYMBOL,
                timeframe=TimeFrame.Minute,
                start=start,
                end=end,
                feed='iex',
            )

            bars = self.data_client.get_stock_bars(request)
            df = bars.df.copy()

            # Flatten MultiIndex if present
            if isinstance(df.index, pd.MultiIndex):
                df = df.droplevel('symbol')

            # Aggregate 1-min bars to 15-min
            df = df.resample('15min').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum',
            }).dropna(subset=['open', 'close'])

            # Flatten MultiIndex if present
            if isinstance(df.index, pd.MultiIndex):
                df = df.droplevel('symbol')

            # Prepare for indicators
            df = df.reset_index()
            if 'timestamp' in df.columns:
                df = df.rename(columns={'timestamp': 'datetime'})
            elif 'index' in df.columns:
                df = df.rename(columns={'index': 'datetime'})

            df.columns = df.columns.str.lower().str.strip()

            # Ensure numeric
            for col in ('open', 'high', 'low', 'close', 'volume'):
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            df = df.dropna(subset=['open', 'high', 'low', 'close'])
            df = df.sort_values('datetime').reset_index(drop=True)

            # Update current price
            if len(df) > 0:
                self.current_qqq_price = float(df.iloc[-1]['close'])

            # Build indicators (same as your NQ strategies)
            df = build_indicators(df)

            self.log.info(
                f"Fetched {len(df)} candles. QQQ=${self.current_qqq_price:.2f}")
            return df

        except Exception as e:
            self.log.error(f"Error fetching data: {e}", exc_info=True)
            return None

    def evaluate_signals(self, df):
        """Run all 5 strategies on latest candle."""
        signals = []
        i = len(df) - 1

        if i < 50:
            return signals

        for name, cfg in STRATEGIES.items():
            if self.active_trades[name] is not None:
                continue

            try:
                if cfg["signal_fn"](df, i):
                    row = df.iloc[i]

                    # Scale NQ TP/SL to QQQ dollars
                    tp_qqq = nq_points_to_qqq_dollars(
                        cfg["tp"], self.current_qqq_price)
                    sl_qqq = nq_points_to_qqq_dollars(
                        cfg["sl"], self.current_qqq_price)

                    # Scale ATR similarly
                    atr_val = row["atr"] if not pd.isna(row["atr"]) else 0.5
                    atr_qqq = nq_points_to_qqq_dollars(
                        atr_val, self.current_qqq_price)

                    signals.append({
                        "strategy": name,
                        "direction": cfg["direction"],
                        "tp_nq": cfg["tp"],
                        "sl_nq": cfg["sl"],
                        "tp_qqq": tp_qqq,
                        "sl_qqq": sl_qqq,
                        "atr_qqq": atr_qqq,
                        "close": float(row["close"]),
                        "datetime": str(row.get("datetime", i)),
                    })
            except Exception as e:
                self.log.error(f"Error in {name}: {e}")

        return signals

    def place_order(self, signal):
        """Place a market order on Alpaca."""
        side = OrderSide.BUY if signal["direction"] == "LONG" else OrderSide.SELL

        self.log.info(
            f"ORDER: {signal['strategy']} | {side.value} {SHARES} {SYMBOL} "
            f"| TP=${signal['tp_qqq']:.2f} SL=${signal['sl_qqq']:.2f} "
            f"| NQ equiv: TP={signal['tp_nq']}pts SL={signal['sl_nq']}pts"
        )

        if self.dry_run:
            self.log.info("  [DRY-RUN] Order simulated.")
            return f"DRY-{signal['strategy']}-{datetime.now().strftime('%H%M%S')}"

        try:
            order_request = MarketOrderRequest(
                symbol=SYMBOL,
                qty=SHARES,
                side=side,
                time_in_force=TimeInForce.DAY,
            )
            order = self.trading_client.submit_order(order_request)
            self.log.info(f"  Order submitted. ID: {order.id}")
            return str(order.id)

        except Exception as e:
            self.log.error(f"  ORDER FAILED: {e}")
            return None

    def close_position(self, trade, reason, exit_price=None):
        """Close an active position."""
        side = OrderSide.SELL if trade.direction == "LONG" else OrderSide.BUY

        self.log.info(
            f"CLOSING: {trade.strategy_name} | {side.value} {SHARES} {SYMBOL} | {reason}")

        if not self.dry_run:
            try:
                order_request = MarketOrderRequest(
                    symbol=SYMBOL,
                    qty=SHARES,
                    side=side,
                    time_in_force=TimeInForce.DAY,
                )
                self.trading_client.submit_order(order_request)
                self.log.info("  Close order submitted.")
            except Exception as e:
                self.log.error(f"  CLOSE FAILED: {e}")

        if exit_price is None:
            exit_price = self.current_qqq_price

        pnl = trade.calc_pnl_dollars(exit_price)
        self.daily_pnl += pnl

        self.daily_trades.append({
            "strategy": trade.strategy_name,
            "direction": trade.direction,
            "entry": trade.entry_price,
            "exit": exit_price,
            "pnl": pnl,
            "reason": reason,
            "bars": trade.bars_held,
            "time": datetime.now(TIMEZONE).strftime("%H:%M:%S"),
        })

        self.log.info(
            f"  PnL: ${pnl:+,.2f} | Daily: ${self.daily_pnl:+,.2f}"
        )

        self.active_trades[trade.strategy_name] = None

    def manage_trades(self, df):
        """Check TP/SL/trailing on active trades."""
        if df is None or len(df) < 2:
            return

        latest = df.iloc[-1]

        for name, trade in self.active_trades.items():
            if trade is None:
                continue

            trade.update_trailing_stop(
                float(latest["high"]), float(latest["low"]))
            hit, reason, exit_price = trade.check_exit(
                float(latest["high"]), float(latest["low"]))

            if hit:
                self.close_position(trade, reason, exit_price)
            elif trade.bars_held >= 40:
                self.close_position(trade, "TIMEOUT", float(latest["close"]))

    def flatten_all(self, reason="CUTOFF"):
        for name, trade in self.active_trades.items():
            if trade is not None:
                self.log.warning(f"FLATTEN {name}: {reason}")
                self.close_position(trade, reason)

    def print_summary(self):
        self.log.info("\n" + "=" * 70)
        self.log.info("  DAILY SUMMARY")
        self.log.info("=" * 70)
        self.log.info(f"  Trades: {len(self.daily_trades)}")
        self.log.info(f"  Total PnL: ${self.daily_pnl:+,.2f}")

        if self.daily_trades:
            wins = [t for t in self.daily_trades if t["pnl"] > 0]
            wr = len(wins) / len(self.daily_trades) * 100
            self.log.info(f"  Win Rate: {wr:.1f}%")
            self.log.info("")
            self.log.info(
                f"  {'Strategy':<22} {'Dir':<6} {'Entry':>8} {'Exit':>8} {'PnL':>10} {'Reason':<7}")
            self.log.info(f"  {'-'*65}")
            for t in self.daily_trades:
                self.log.info(
                    f"  {t['strategy']:<22} {t['direction']:<6} "
                    f"${t['entry']:>7.2f} ${t['exit']:>7.2f} "
                    f"${t['pnl']:>+9.2f} {t['reason']:<7}"
                )
        self.log.info("=" * 70)

    def run(self):
        """Main trading loop (synchronous)."""
        self.log.info("=" * 70)
        self.log.info("  QQQ ALGO TRADER (Alpaca Paper)")
        self.log.info(
            f"  Mode: {'DRY-RUN' if self.dry_run else 'PAPER TRADING'}")
        self.log.info(f"  Symbol: {SYMBOL} | Shares: {SHARES}")
        self.log.info(f"  Daily loss limit: ${DAILY_LOSS_LIMIT_DOLLARS}")
        self.log.info(f"  Strategies: {', '.join(STRATEGIES.keys())}")
        self.log.info(
            f"  NQ→QQQ scaling: TP60pts ≈ ${nq_points_to_qqq_dollars(60, 607):.2f}/share")
        self.log.info("=" * 70)

        # Verify connection
        try:
            account = self.trading_client.get_account()
            self.log.info(f"  Account: {account.account_number}")
            self.log.info(f"  Cash: ${float(account.cash):,.2f}")
            self.log.info(f"  Status: {account.status}")
        except Exception as e:
            self.log.error(f"Connection failed: {e}")
            return

        self.log.info("=" * 70)

        while True:
            try:
                now = datetime.now(TIMEZONE)
                current_time = now.time()

                # Daily reset
                if current_time < dtime(4, 5) and self.daily_pnl != 0:
                    self.log.info("NEW DAY — Resetting PnL")
                    self.daily_pnl = 0.0
                    self.daily_trades = []
                    self.trading_halted = False

                # Hard cutoff
                if current_time >= HARD_CUTOFF_TIME and current_time < MARKET_CLOSE:
                    has_active = any(
                        t is not None for t in self.active_trades.values())
                    if has_active:
                        self.log.warning("HARD CUTOFF — Flattening!")
                        self.flatten_all("3:55 PM CUTOFF")

                # Outside market hours
                if current_time < MARKET_OPEN or current_time >= MARKET_CLOSE:
                    if current_time >= MARKET_CLOSE and current_time < dtime(20, 0):
                        self.print_summary()
                        self.log.info("Market closed. Waiting...")
                    else:
                        self.log.info(
                            f"Pre-market. Waiting for {MARKET_OPEN.strftime('%I:%M %p')} ET...")
                    time.sleep(120)
                    continue

                # Daily loss check
                if abs(self.daily_pnl) >= (DAILY_LOSS_LIMIT_DOLLARS - SAFETY_BUFFER):
                    if self.daily_pnl < 0 and not self.trading_halted:
                        self.log.warning(
                            f"DAILY LOSS LIMIT: ${self.daily_pnl:,.2f}. Halting.")
                        self.trading_halted = True
                        self.flatten_all("LOSS LIMIT")
                    if self.trading_halted:
                        time.sleep(60)
                        continue

                # Fetch data
                df = self.fetch_candles()
                if df is None or len(df) < 50:
                    self.log.warning("Insufficient data. Waiting 60s...")
                    time.sleep(60)
                    continue

                # Dedup candles
                latest_time = str(df.iloc[-1].get("datetime", len(df)))
                if latest_time == self.last_candle_time:
                    self.manage_trades(df)
                    time.sleep(30)
                    continue

                self.last_candle_time = latest_time
                self.log.info(
                    f"New candle: {latest_time} | QQQ=${self.current_qqq_price:.2f}")

                # Manage existing trades
                self.manage_trades(df)

                # No new trades near close
                if current_time >= NO_NEW_TRADES_TIME:
                    self.log.info("Near close — exits only.")
                    time.sleep(30)
                    continue

                # Evaluate signals
                signals = self.evaluate_signals(df)

                for signal in signals:
                    self.log.info(
                        f"SIGNAL: {signal['strategy']} | {signal['direction']} "
                        f"| QQQ=${signal['close']:.2f} "
                        f"| TP=${signal['tp_qqq']:.2f} SL=${signal['sl_qqq']:.2f}"
                    )

                    order_id = self.place_order(signal)

                    if order_id:
                        trade = ActiveTrade(
                            strategy_name=signal["strategy"],
                            direction=signal["direction"],
                            entry_price=signal["close"],
                            tp_dollars=signal["tp_qqq"],
                            sl_dollars=signal["sl_qqq"],
                            atr_value=signal["atr_qqq"],
                            order_id=order_id,
                        )
                        self.active_trades[signal["strategy"]] = trade
                        self.log.info(
                            f"  Trade open: ${trade.entry_price:.2f} "
                            f"→ TP=${trade.tp_price:.2f} SL=${trade.sl_price:.2f}"
                        )

                # Wait
                time.sleep(60)

            except KeyboardInterrupt:
                self.log.info("Shutdown (Ctrl+C)")
                self.flatten_all("USER SHUTDOWN")
                break

            except Exception as e:
                self.log.error(f"Error: {e}", exc_info=True)
                time.sleep(30)

        self.print_summary()


# ============================================================
# ENTRY POINT
# ============================================================

def main():
    load_dotenv()
    dry_run = "--dry-run" in sys.argv

    if not dry_run:
        print("\n" + "!" * 60)
        print("  PAPER TRADING MODE")
        print("  Orders will be placed on your Alpaca paper account.")
        print("  No real money. Use --dry-run for signals only.")
        print("!" * 60)

    trader = AlpacaTrader(dry_run=dry_run)
    trader.run()


if __name__ == "__main__":
    main()
