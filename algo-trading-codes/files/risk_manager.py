"""
Risk Manager
=============
Enforces Topstep-specific risk rules and general risk management.

This module acts as a GATEKEEPER between your strategy signals and order execution.
Every trade signal MUST pass through the risk manager before being sent to the broker.

Topstep Rules Enforced:
    - Daily loss limit
    - Trailing max drawdown
    - Max position size
    - Consistency rule (no single day > X% of total profit)
    - Trading hours enforcement
    - Consecutive loss circuit breaker
"""

import logging
from datetime import datetime, time
from dataclasses import dataclass, field
from typing import Optional
from zoneinfo import ZoneInfo

from config.config import TopstepRiskConfig
from strategy.hybrid_strategy import TradeSignal, TradeDirection

logger = logging.getLogger(__name__)

# Topstep uses Central Time
CT = ZoneInfo("America/Chicago")


@dataclass
class DailyStats:
    """Tracks daily trading performance."""
    date: str = ""
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    num_trades: int = 0
    num_wins: int = 0
    num_losses: int = 0
    consecutive_losses: int = 0
    high_water_mark: float = 0.0  # Peak balance for trailing drawdown
    is_locked: bool = False       # True = no more trading today


@dataclass
class RiskCheckResult:
    """Result of a risk check."""
    approved: bool
    reason: str
    adjusted_quantity: Optional[int] = None  # May reduce quantity


class RiskManager:
    """
    Gatekeeper for all trade execution.
    Call check_trade() before EVERY order placement.
    """

    def __init__(self, config: TopstepRiskConfig, starting_balance: float):
        self.config = config
        self.starting_balance = starting_balance
        self.daily_stats = DailyStats()
        self.daily_stats.high_water_mark = starting_balance
        self.total_profit_all_days: float = 0.0
        self.daily_profits: list[float] = []  # Track each day's P&L
        self._last_loss_time: Optional[datetime] = None

    # ─── MAIN RISK CHECK ─────────────────────────────────────────

    def check_trade(self, signal: TradeSignal, current_position_qty: int = 0) -> RiskCheckResult:
        """
        Run ALL risk checks against a proposed trade signal.
        Returns RiskCheckResult with approval/denial and reason.
        """
        checks = [
            self._check_daily_lock(),
            self._check_trading_hours(),
            self._check_daily_loss_limit(),
            self._check_trailing_drawdown(),
            self._check_position_size(current_position_qty),
            self._check_consecutive_losses(),
            self._check_cooldown_period(),
        ]

        for result in checks:
            if not result.approved:
                logger.warning(f"⛔ RISK DENIED: {result.reason}")
                return result

        logger.info(f"✅ Risk check PASSED for {signal.direction.value} signal")
        return RiskCheckResult(approved=True, reason="All risk checks passed")

    # ─── INDIVIDUAL RISK CHECKS ──────────────────────────────────

    def _check_daily_lock(self) -> RiskCheckResult:
        """Check if trading is locked for the day."""
        if self.daily_stats.is_locked:
            return RiskCheckResult(False, "Trading is LOCKED for today. "
                                          "Daily limit was hit — no more trades allowed.")
        return RiskCheckResult(True, "Day not locked")

    def _check_trading_hours(self) -> RiskCheckResult:
        """Ensure we're within allowed trading hours (Central Time)."""
        now_ct = datetime.now(CT)
        current_time = now_ct.time()

        start = time(self.config.trading_start_hour, self.config.trading_start_minute)
        end = time(self.config.trading_end_hour, self.config.trading_end_minute)

        if current_time < start:
            return RiskCheckResult(False, f"Too early — trading starts at {start} CT "
                                          f"(current: {current_time.strftime('%H:%M')} CT)")
        if current_time > end:
            return RiskCheckResult(False, f"Too late — no new entries after {end} CT "
                                          f"(current: {current_time.strftime('%H:%M')} CT)")

        return RiskCheckResult(True, "Within trading hours")

    def _check_daily_loss_limit(self) -> RiskCheckResult:
        """Check if daily loss limit has been reached."""
        total_daily_pnl = self.daily_stats.realized_pnl + self.daily_stats.unrealized_pnl

        if total_daily_pnl <= -self.config.max_daily_loss:
            self.daily_stats.is_locked = True
            return RiskCheckResult(False,
                                   f"DAILY LOSS LIMIT HIT: ${total_daily_pnl:.2f} "
                                   f"(limit: -${self.config.max_daily_loss:.2f})")

        # Warn if approaching limit (within 75%)
        if total_daily_pnl <= -(self.config.max_daily_loss * 0.75):
            logger.warning(f"⚠️ Approaching daily loss limit: ${total_daily_pnl:.2f} "
                           f"of -${self.config.max_daily_loss:.2f}")

        return RiskCheckResult(True, "Within daily loss limit")

    def _check_trailing_drawdown(self) -> RiskCheckResult:
        """
        Check trailing max drawdown.
        Topstep's trailing drawdown follows your equity high-water mark.
        """
        current_balance = self.starting_balance + self.daily_stats.realized_pnl
        drawdown = self.daily_stats.high_water_mark - current_balance

        if drawdown >= self.config.trailing_max_drawdown:
            self.daily_stats.is_locked = True
            return RiskCheckResult(False,
                                   f"TRAILING DRAWDOWN LIMIT HIT: ${drawdown:.2f} "
                                   f"(limit: ${self.config.trailing_max_drawdown:.2f})")

        return RiskCheckResult(True, "Within trailing drawdown limit")

    def _check_position_size(self, current_qty: int) -> RiskCheckResult:
        """Ensure we don't exceed max position size."""
        if current_qty >= self.config.max_position_size:
            return RiskCheckResult(False,
                                   f"Max position size reached: {current_qty} contracts "
                                   f"(limit: {self.config.max_position_size})")

        # Calculate how many more contracts we can add
        remaining = self.config.max_position_size - current_qty
        adjusted = min(remaining, self.config.max_contracts_per_order)

        return RiskCheckResult(True, "Position size OK", adjusted_quantity=adjusted)

    def _check_consecutive_losses(self) -> RiskCheckResult:
        """Stop trading after N consecutive losses."""
        if self.daily_stats.consecutive_losses >= self.config.max_consecutive_losses:
            return RiskCheckResult(False,
                                   f"Circuit breaker: {self.daily_stats.consecutive_losses} "
                                   f"consecutive losses (limit: {self.config.max_consecutive_losses}). "
                                   f"Take a break.")
        return RiskCheckResult(True, "Consecutive loss limit not reached")

    def _check_cooldown_period(self) -> RiskCheckResult:
        """Enforce cooldown period after a loss."""
        if self._last_loss_time is None:
            return RiskCheckResult(True, "No recent losses")

        now = datetime.now(CT)
        elapsed = (now - self._last_loss_time).total_seconds() / 60

        if elapsed < self.config.cooldown_after_loss_minutes:
            remaining = self.config.cooldown_after_loss_minutes - elapsed
            return RiskCheckResult(False,
                                   f"Cooldown active: {remaining:.0f} minutes remaining "
                                   f"after last loss")

        return RiskCheckResult(True, "Cooldown period over")

    # ─── P&L TRACKING ────────────────────────────────────────────

    def record_trade_result(self, pnl: float) -> None:
        """Record the result of a closed trade."""
        self.daily_stats.realized_pnl += pnl
        self.daily_stats.num_trades += 1

        if pnl >= 0:
            self.daily_stats.num_wins += 1
            self.daily_stats.consecutive_losses = 0
        else:
            self.daily_stats.num_losses += 1
            self.daily_stats.consecutive_losses += 1
            self._last_loss_time = datetime.now(CT)

        # Update high-water mark
        current_balance = self.starting_balance + self.daily_stats.realized_pnl
        if current_balance > self.daily_stats.high_water_mark:
            self.daily_stats.high_water_mark = current_balance

        logger.info(f"📊 Trade P&L: ${pnl:+.2f} | "
                     f"Daily P&L: ${self.daily_stats.realized_pnl:+.2f} | "
                     f"W/L: {self.daily_stats.num_wins}/{self.daily_stats.num_losses}")

    def update_unrealized_pnl(self, unrealized: float) -> None:
        """Update unrealized P&L (call on each price tick while in a position)."""
        self.daily_stats.unrealized_pnl = unrealized

        # Check if unrealized + realized breaks daily limit
        total = self.daily_stats.realized_pnl + unrealized
        if total <= -self.config.max_daily_loss:
            self.daily_stats.is_locked = True
            logger.critical(f"🚨 DAILY LIMIT BREACHED WITH UNREALIZED P&L: ${total:.2f}")

    def should_flatten_positions(self) -> bool:
        """Check if it's time to flatten all positions (end of day)."""
        now_ct = datetime.now(CT)
        flatten_time = time(self.config.flatten_hour, self.config.flatten_minute)
        return now_ct.time() >= flatten_time

    def reset_daily(self) -> None:
        """Reset daily stats for a new trading day."""
        if self.daily_stats.realized_pnl != 0:
            self.daily_profits.append(self.daily_stats.realized_pnl)
            self.total_profit_all_days += self.daily_stats.realized_pnl

        self.daily_stats = DailyStats()
        self.daily_stats.high_water_mark = (
            self.starting_balance + self.total_profit_all_days
        )
        logger.info("📅 Daily stats reset for new trading day")

    def check_consistency_rule(self) -> bool:
        """
        Topstep consistency rule: No single day's profit should be
        more than X% of total profits.
        """
        if self.total_profit_all_days <= 0 or not self.daily_profits:
            return True  # Not applicable yet

        max_single_day = max(self.daily_profits)
        pct = (max_single_day / self.total_profit_all_days) * 100

        if pct > self.config.consistency_target_pct:
            logger.warning(f"⚠️ Consistency warning: Best day is {pct:.1f}% "
                           f"of total profit (target: <{self.config.consistency_target_pct}%)")
            return False
        return True

    def get_status_report(self) -> str:
        """Generate a human-readable risk status report."""
        stats = self.daily_stats
        return (
            f"\n{'='*50}\n"
            f"  📊 RISK MANAGER STATUS\n"
            f"{'='*50}\n"
            f"  Daily P&L:          ${stats.realized_pnl:+.2f}\n"
            f"  Unrealized P&L:     ${stats.unrealized_pnl:+.2f}\n"
            f"  Daily Loss Limit:   -${self.config.max_daily_loss:.2f}\n"
            f"  Remaining Room:     ${self.config.max_daily_loss + stats.realized_pnl:.2f}\n"
            f"  Trades Today:       {stats.num_trades} "
            f"(W:{stats.num_wins} / L:{stats.num_losses})\n"
            f"  Consec. Losses:     {stats.consecutive_losses} "
            f"(max: {self.config.max_consecutive_losses})\n"
            f"  Day Locked:         {'YES ⛔' if stats.is_locked else 'NO ✅'}\n"
            f"{'='*50}"
        )
