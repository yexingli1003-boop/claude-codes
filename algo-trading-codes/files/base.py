"""
Abstract Broker Interface
=========================
All broker connectors must implement this interface.
This lets your strategy code work with ANY broker without changes.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional
from datetime import datetime


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    """Represents a trading order."""
    id: Optional[str] = None
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    order_type: OrderType = OrderType.MARKET
    quantity: int = 1
    price: Optional[float] = None       # For limit orders
    stop_price: Optional[float] = None  # For stop orders
    status: OrderStatus = OrderStatus.PENDING
    filled_price: Optional[float] = None
    filled_at: Optional[datetime] = None
    created_at: Optional[datetime] = None


@dataclass
class Position:
    """Represents a current position."""
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    quantity: int = 0
    entry_price: float = 0.0
    current_price: float = 0.0
    unrealized_pnl: float = 0.0


@dataclass
class AccountInfo:
    """Represents account balance and status."""
    account_id: str = ""
    balance: float = 0.0
    realized_pnl_today: float = 0.0
    unrealized_pnl: float = 0.0
    buying_power: float = 0.0


class BaseBroker(ABC):
    """
    Abstract base class for all broker connectors.
    Implement this interface for each broker you want to support.
    """

    @abstractmethod
    async def connect(self) -> bool:
        """Authenticate and establish connection. Returns True on success."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Cleanly disconnect from the broker."""
        pass

    @abstractmethod
    async def get_account_info(self) -> AccountInfo:
        """Get current account balance and P&L info."""
        pass

    @abstractmethod
    async def get_positions(self) -> list[Position]:
        """Get all open positions."""
        pass

    @abstractmethod
    async def place_order(self, order: Order) -> Order:
        """Submit an order. Returns the order with updated status/ID."""
        pass

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order. Returns True on success."""
        pass

    @abstractmethod
    async def modify_order(self, order_id: str, price: Optional[float] = None,
                           stop_price: Optional[float] = None,
                           quantity: Optional[int] = None) -> Order:
        """Modify an existing order."""
        pass

    @abstractmethod
    async def flatten_all(self) -> bool:
        """Close ALL open positions immediately. Emergency use."""
        pass

    @abstractmethod
    async def get_historical_candles(self, symbol: str, timeframe: str,
                                      count: int) -> list[dict]:
        """
        Fetch historical OHLCV candles.
        Returns list of dicts: {timestamp, open, high, low, close, volume}
        """
        pass

    @abstractmethod
    async def subscribe_candles(self, symbol: str, timeframe: str,
                                 callback) -> None:
        """Subscribe to real-time candle updates via WebSocket."""
        pass
