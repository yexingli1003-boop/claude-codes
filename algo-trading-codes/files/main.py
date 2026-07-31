"""
Algo Trading Bot — Main Entry Point
=====================================
Ties together all modules: broker, strategy, risk management, logging.

Usage:
    python main.py              # Run with default config (DEMO mode)
    python main.py --live       # Run in LIVE mode (real money!)

⚠️  ALWAYS start in DEMO mode and paper trade for weeks before going live.
"""

import asyncio
import logging
import signal
import sys
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd

from config.config import AppConfig, Environment
from brokers.tradovate import TradovateBroker
from brokers.base import Order, OrderSide, OrderType, Position
from strategy.hybrid_strategy import HybridStrategy, TradeSignal, TradeDirection
from risk.risk_manager import RiskManager
from utils.logger import setup_logging, TradeLogger

logger = logging.getLogger(__name__)
CT = ZoneInfo("America/Chicago")


class TradingBot:
    """
    Main trading bot orchestrator.

    Lifecycle:
        1. Initialize all components
        2. Connect to broker
        3. Subscribe to market data
        4. On each new candle:
            a. Run strategy analysis
            b. Check risk rules
            c. Execute trades if approved
            d. Manage open positions (stop/target monitoring)
        5. End-of-day: flatten, log, reset
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self.broker = TradovateBroker(config.tradovate, config.contract)
        self.strategy = HybridStrategy(config.strategy)
        self.risk_manager: Optional[RiskManager] = None
        self.trade_logger = TradeLogger(config.trade_log_file)

        # State
        self.candle_buffer: list[dict] = []
        self.current_position: Optional[Position] = None
        self.active_signal: Optional[TradeSignal] = None
        self.entry_time: Optional[datetime] = None
        self.running = False

    async def start(self) -> None:
        """Initialize and start the trading bot."""
        setup_logging(self.config.log_level, self.config.log_file)

        logger.info("=" * 60)
        logger.info("  🤖 ALGO TRADING BOT STARTING")
        logger.info(f"  Mode:     {self.config.tradovate.environment.value.upper()}")
        logger.info(f"  Symbol:   {self.config.contract.symbol}")
        logger.info(f"  Strategy: Hybrid (Candles + Indicators)")
        logger.info("=" * 60)

        if self.config.tradovate.environment == Environment.LIVE:
            logger.warning("⚠️  LIVE MODE — Real money is at risk!")
            logger.warning("    Press Ctrl+C within 10 seconds to abort...")
            await asyncio.sleep(10)

        # Step 1: Connect to broker
        connected = await self.broker.connect()
        if not connected:
            logger.critical("❌ Failed to connect to broker. Exiting.")
            return

        # Step 2: Get account info and initialize risk manager
        account = await self.broker.get_account_info()
        self.risk_manager = RiskManager(self.config.risk, account.balance)
        logger.info(f"💰 Account balance: ${account.balance:,.2f}")

        # Step 3: Subscribe to candle data
        await self.broker.subscribe_candles(
            self.config.contract.symbol,
            self.config.strategy.candle_timeframe,
            self._on_new_candle
        )

        # Step 4: Enter main loop
        self.running = True
        await self._main_loop()

    async def _main_loop(self) -> None:
        """Main event loop — monitors positions and health."""
        try:
            while self.running:
                # Check if it's time to flatten (end of day)
                if self.risk_manager.should_flatten_positions():
                    await self._flatten_end_of_day()

                # Heartbeat / health check
                await asyncio.sleep(self.config.heartbeat_interval)

        except asyncio.CancelledError:
            logger.info("Bot shutting down...")
        finally:
            await self.shutdown()

    async def _on_new_candle(self, candle: dict) -> None:
        """
        Called every time a new candle completes.
        This is the main decision-making entry point.
        """
        # Add candle to buffer
        self.candle_buffer.append(candle)

        # Keep buffer at a reasonable size (last 500 candles)
        if len(self.candle_buffer) > 500:
            self.candle_buffer = self.candle_buffer[-500:]

        # Convert to DataFrame for analysis
        df = pd.DataFrame(self.candle_buffer)
        if "timestamp" in df.columns:
            df.index = pd.to_datetime(df["timestamp"])

        # ─── If we have an open position, monitor it ───────────
        if self.current_position:
            await self._monitor_position(df)
            return  # Don't look for new entries while in a position

        # ─── If no position, look for new entry signals ────────
        signal = self.strategy.analyze(df)

        if signal is None:
            return  # No setup found

        # ─── Risk check ───────────────────────────────────────
        positions = await self.broker.get_positions()
        current_qty = sum(p.quantity for p in positions)
        risk_result = self.risk_manager.check_trade(signal, current_qty)

        if not risk_result.approved:
            logger.info(f"Signal rejected by risk manager: {risk_result.reason}")
            return

        # ─── Execute the trade ─────────────────────────────────
        quantity = risk_result.adjusted_quantity or 1
        await self._execute_entry(signal, quantity)

    async def _execute_entry(self, signal: TradeSignal, quantity: int) -> None:
        """Execute a trade entry based on the signal."""
        side = OrderSide.BUY if signal.direction == TradeDirection.LONG else OrderSide.SELL

        # Place market order for entry
        entry_order = Order(
            symbol=self.config.contract.symbol,
            side=side,
            order_type=OrderType.MARKET,
            quantity=quantity,
        )

        result = await self.broker.place_order(entry_order)

        if result.status.value not in ("rejected",):
            self.active_signal = signal
            self.entry_time = datetime.now(CT)

            # Place stop-loss order
            stop_side = OrderSide.SELL if side == OrderSide.BUY else OrderSide.BUY
            stop_order = Order(
                symbol=self.config.contract.symbol,
                side=stop_side,
                order_type=OrderType.STOP,
                quantity=quantity,
                stop_price=signal.stop_loss,
            )
            await self.broker.place_order(stop_order)

            logger.info(f"🚀 ENTRY: {signal.direction.value.upper()} "
                         f"{quantity}x {self.config.contract.symbol} @ MARKET | "
                         f"SL: {signal.stop_loss} | TP: {signal.take_profit}")

            # Refresh position
            positions = await self.broker.get_positions()
            if positions:
                self.current_position = positions[0]

    async def _monitor_position(self, df: pd.DataFrame) -> None:
        """Monitor an open position for exit conditions."""
        if not self.active_signal or not self.current_position:
            return

        current_price = df.iloc[-1]["close"]
        signal = self.active_signal
        is_long = signal.direction == TradeDirection.LONG

        # Check take-profit
        hit_tp = (is_long and current_price >= signal.take_profit) or \
                 (not is_long and current_price <= signal.take_profit)

        # Check stop-loss (backup — broker stop should catch this too)
        hit_sl = (is_long and current_price <= signal.stop_loss) or \
                 (not is_long and current_price >= signal.stop_loss)

        if hit_tp or hit_sl:
            await self._execute_exit(current_price, "TP" if hit_tp else "SL")

        # Update unrealized P&L for risk manager
        if is_long:
            unrealized = (current_price - self.current_position.entry_price) * \
                         self.current_position.quantity * self.config.contract.point_value
        else:
            unrealized = (self.current_position.entry_price - current_price) * \
                         self.current_position.quantity * self.config.contract.point_value

        self.risk_manager.update_unrealized_pnl(unrealized)

    async def _execute_exit(self, exit_price: float, reason: str) -> None:
        """Close the current position and log the trade."""
        if not self.current_position or not self.active_signal:
            return

        # Flatten via market order
        await self.broker.flatten_all()

        # Calculate P&L
        signal = self.active_signal
        entry = self.current_position.entry_price
        qty = self.current_position.quantity
        point_value = self.config.contract.point_value

        if signal.direction == TradeDirection.LONG:
            pnl = (exit_price - entry) * qty * point_value
        else:
            pnl = (entry - exit_price) * qty * point_value

        # Record in risk manager
        self.risk_manager.record_trade_result(pnl)

        # Log to CSV
        duration = (datetime.now(CT) - self.entry_time).total_seconds() \
            if self.entry_time else 0

        self.trade_logger.log_trade(
            direction=signal.direction.value,
            symbol=self.config.contract.symbol,
            entry_price=entry,
            exit_price=exit_price,
            quantity=qty,
            pnl=pnl,
            pattern=signal.pattern_name,
            confirmations="; ".join(signal.confirmations),
            confidence=signal.confidence,
            duration_seconds=duration,
        )

        exit_emoji = "✅" if pnl >= 0 else "❌"
        logger.info(f"{exit_emoji} EXIT ({reason}): "
                     f"{'LONG' if signal.direction == TradeDirection.LONG else 'SHORT'} "
                     f"{qty}x {self.config.contract.symbol} | "
                     f"Entry: {entry:.2f} → Exit: {exit_price:.2f} | "
                     f"P&L: ${pnl:+.2f}")

        # Print risk status
        logger.info(self.risk_manager.get_status_report())

        # Clear state
        self.current_position = None
        self.active_signal = None
        self.entry_time = None

    async def _flatten_end_of_day(self) -> None:
        """End-of-day routine: flatten positions and reset."""
        logger.info("🌅 End of day — flattening all positions")

        if self.current_position:
            positions = await self.broker.get_positions()
            if positions:
                current_price = 0  # Would come from last candle data
                await self._execute_exit(current_price, "EOD_FLATTEN")

        await self.broker.flatten_all()

        # Log daily stats
        logger.info(self.risk_manager.get_status_report())
        self.risk_manager.check_consistency_rule()
        self.risk_manager.reset_daily()

    async def shutdown(self) -> None:
        """Graceful shutdown."""
        logger.info("🛑 Shutting down trading bot...")

        # Flatten any open positions
        try:
            await self.broker.flatten_all()
        except Exception as e:
            logger.error(f"Error flattening on shutdown: {e}")

        await self.broker.disconnect()
        logger.info("Bot stopped. Final status:")
        if self.risk_manager:
            logger.info(self.risk_manager.get_status_report())


# ─── ENTRY POINT ─────────────────────────────────────────────

def main():
    """Main entry point."""
    config = AppConfig()

    # Check for --live flag
    if "--live" in sys.argv:
        config.tradovate.environment = Environment.LIVE
        print("\n⚠️  WARNING: Running in LIVE mode with real money!")
        confirm = input("Type 'YES' to confirm: ")
        if confirm != "YES":
            print("Aborted.")
            return

    bot = TradingBot(config)

    # Handle Ctrl+C gracefully
    loop = asyncio.new_event_loop()

    def handle_signal(sig, frame):
        logger.info(f"Received signal {sig}, shutting down...")
        bot.running = False

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        loop.run_until_complete(bot.start())
    finally:
        loop.close()


if __name__ == "__main__":
    main()
