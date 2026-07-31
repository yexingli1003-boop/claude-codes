"""
Configuration for the Algo Trading Bot
=======================================
Edit these values to match your Tradovate credentials,
strategy preferences, and Topstep account rules.
"""

import os
from dataclasses import dataclass, field
from enum import Enum


class Environment(Enum):
    DEMO = "demo"
    LIVE = "live"


@dataclass
class TradovateConfig:
    """Tradovate API connection settings."""
    # --- Credentials (use environment variables in production) ---
    username: str = os.getenv("TRADOVATE_USERNAME", "")
    password: str = os.getenv("TRADOVATE_PASSWORD", "")
    api_key: str = os.getenv("TRADOVATE_API_KEY", "")
    api_secret: str = os.getenv("TRADOVATE_API_SECRET", "")
    cid: str = os.getenv("TRADOVATE_CID", "")

    # --- Environment ---
    # START WITH DEMO! Switch to LIVE only when ready
    environment: Environment = Environment.DEMO

    # --- API URLs ---
    @property
    def base_url(self) -> str:
        if self.environment == Environment.DEMO:
            return "https://demo.tradovateapi.com/v1"
        return "https://live.tradovateapi.com/v1"

    @property
    def ws_url(self) -> str:
        if self.environment == Environment.DEMO:
            return "wss://demo.tradovateapi.com/v1/websocket"
        return "wss://live.tradovateapi.com/v1/websocket"

    @property
    def md_url(self) -> str:
        """Market data WebSocket URL."""
        if self.environment == Environment.DEMO:
            return "wss://md-demo.tradovateapi.com/v1/websocket"
        return "wss://md.tradovateapi.com/v1/websocket"


@dataclass
class ContractConfig:
    """Futures contract settings."""
    symbol: str = "NQ"             # E-mini Nasdaq 100
    exchange: str = "CME"
    tick_size: float = 0.25        # NQ tick size
    tick_value: float = 5.00       # $5 per tick for NQ
    point_value: float = 20.00    # $20 per point for NQ


@dataclass
class StrategyConfig:
    """Hybrid strategy parameters."""
    # --- Candlestick Pattern Settings ---
    candle_timeframe: str = "5min"  # Primary candle timeframe
    higher_tf: str = "15min"        # Higher timeframe for trend context
    # Minimum body-to-range ratio for strong candles
    min_candle_body_ratio: float = 0.6

    # --- EMA Settings ---
    ema_fast: int = 9
    ema_slow: int = 21

    # --- RSI Settings ---
    rsi_period: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0

    # --- ATR Settings (for dynamic stop/target) ---
    atr_period: int = 14
    atr_stop_multiplier: float = 1.5   # Stop loss = ATR * this
    atr_target_multiplier: float = 2.5  # Take profit = ATR * this

    # --- VWAP ---
    use_vwap_filter: bool = True  # Only long above VWAP, short below


@dataclass
class TopstepRiskConfig:
    """
    Topstep-specific risk rules.
    ⚠️ UPDATE THESE to match your specific Topstep account rules!
    These are example values for a $50K Topstep account.
    """
    max_daily_loss: float = 1_000.00       # Max loss allowed per day ($)
    trailing_max_drawdown: float = 2_000.00  # Trailing drawdown limit ($)
    max_position_size: int = 2              # Max contracts at once
    max_contracts_per_order: int = 1        # Max contracts per single order
    consistency_target_pct: float = 30.0    # No single day > 30% of total profit

    # --- Trading Hours (CT = Central Time, CME hours) ---
    trading_start_hour: int = 8     # Start trading at 8:00 AM CT
    trading_start_minute: int = 30
    trading_end_hour: int = 14      # Stop new entries at 2:00 PM CT
    trading_end_minute: int = 55
    flatten_hour: int = 14          # Flatten all positions by 2:55 PM CT
    flatten_minute: int = 55

    # --- Circuit Breakers ---
    max_consecutive_losses: int = 3  # Stop trading after N consecutive losses
    cooldown_after_loss_minutes: int = 15  # Wait N minutes after a loss


@dataclass
class AppConfig:
    """Master configuration combining all settings."""
    tradovate: TradovateConfig = field(default_factory=TradovateConfig)
    contract: ContractConfig = field(default_factory=ContractConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    risk: TopstepRiskConfig = field(default_factory=TopstepRiskConfig)

    # --- Logging ---
    log_level: str = "INFO"
    log_file: str = "logs/trading.log"
    trade_log_file: str = "logs/trades.csv"

    # --- General ---
    heartbeat_interval: int = 30  # Seconds between health checks
