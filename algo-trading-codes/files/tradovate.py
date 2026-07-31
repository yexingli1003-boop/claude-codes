"""
Tradovate Broker Connector
===========================
Implements the BaseBroker interface for Tradovate's REST + WebSocket API.
Used for connecting to Topstep accounts.

API Docs: https://api.tradovate.com/
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import Optional, Callable

import aiohttp

from brokers.base import (
    BaseBroker, Order, Position, AccountInfo,
    OrderSide, OrderType, OrderStatus
)
from config.config import TradovateConfig, ContractConfig

logger = logging.getLogger(__name__)


class TradovateBroker(BaseBroker):
    """Tradovate API connector for futures trading."""

    def __init__(self, config: TradovateConfig, contract: ContractConfig):
        self.config = config
        self.contract = contract
        self.access_token: Optional[str] = None
        self.token_expiry: float = 0
        self.account_id: Optional[int] = None
        self.contract_id: Optional[int] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._md_ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._ws_request_id: int = 0
        self._candle_callbacks: dict[str, Callable] = {}

    # ─── CONNECTION & AUTH ────────────────────────────────────────

    async def connect(self) -> bool:
        """Authenticate with Tradovate and get access token."""
        try:
            self._session = aiohttp.ClientSession()

            # Step 1: Get access token
            auth_payload = {
                "name": self.config.username,
                "password": self.config.password,
                "appId": self.config.api_key,
                "appVersion": "1.0",
                "cid": self.config.cid,
                "sec": self.config.api_secret,
            }

            async with self._session.post(
                f"{self.config.base_url}/auth/accessTokenRequest",
                json=auth_payload
            ) as resp:
                if resp.status != 200:
                    error = await resp.text()
                    logger.error(f"Auth failed ({resp.status}): {error}")
                    return False

                data = await resp.json()
                self.access_token = data.get("accessToken")
                # Token typically valid for ~80 minutes
                self.token_expiry = time.time() + data.get("expirationTime", 4800)
                logger.info("✅ Authenticated with Tradovate successfully")

            # Step 2: Get account ID
            await self._fetch_account_id()

            # Step 3: Resolve the contract ID for our symbol
            await self._resolve_contract()

            return True

        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False

    async def disconnect(self) -> None:
        """Close all connections."""
        if self._ws:
            await self._ws.close()
        if self._md_ws:
            await self._md_ws.close()
        if self._session:
            await self._session.close()
        logger.info("Disconnected from Tradovate")

    async def _ensure_token(self) -> None:
        """Refresh token if expired or about to expire."""
        if time.time() > (self.token_expiry - 60):
            logger.info("Token expiring soon, refreshing...")
            async with self._session.post(
                f"{self.config.base_url}/auth/renewAccessToken",
                headers=self._headers()
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    self.access_token = data.get("accessToken", self.access_token)
                    self.token_expiry = time.time() + data.get("expirationTime", 4800)
                    logger.info("Token refreshed")

    def _headers(self) -> dict:
        """Standard auth headers."""
        return {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
        }

    async def _fetch_account_id(self) -> None:
        """Get the primary trading account ID."""
        async with self._session.get(
            f"{self.config.base_url}/account/list",
            headers=self._headers()
        ) as resp:
            accounts = await resp.json()
            if accounts:
                self.account_id = accounts[0]["id"]
                logger.info(f"Using account ID: {self.account_id} "
                            f"({accounts[0].get('name', 'N/A')})")

    async def _resolve_contract(self) -> None:
        """Find the contract ID for our configured symbol."""
        async with self._session.get(
            f"{self.config.base_url}/contract/find",
            params={"name": self.contract.symbol},
            headers=self._headers()
        ) as resp:
            if resp.status == 200:
                contract_data = await resp.json()
                self.contract_id = contract_data.get("id")
                logger.info(f"Resolved {self.contract.symbol} → "
                            f"contract ID {self.contract_id}")

    # ─── ACCOUNT INFO ────────────────────────────────────────────

    async def get_account_info(self) -> AccountInfo:
        """Fetch current account balance and P&L."""
        await self._ensure_token()

        async with self._session.get(
            f"{self.config.base_url}/cashBalance/getCashBalanceSnapshot",
            params={"accountId": self.account_id},
            headers=self._headers()
        ) as resp:
            data = await resp.json()

        return AccountInfo(
            account_id=str(self.account_id),
            balance=data.get("totalCashValue", 0),
            realized_pnl_today=data.get("realizedPnL", 0),
            unrealized_pnl=data.get("unrealizedPnL", 0),
            buying_power=data.get("totalCashValue", 0),
        )

    # ─── POSITIONS ───────────────────────────────────────────────

    async def get_positions(self) -> list[Position]:
        """Get all open positions."""
        await self._ensure_token()

        async with self._session.get(
            f"{self.config.base_url}/position/list",
            headers=self._headers()
        ) as resp:
            positions_data = await resp.json()

        positions = []
        for p in positions_data:
            if p.get("netPos", 0) != 0:
                side = OrderSide.BUY if p["netPos"] > 0 else OrderSide.SELL
                positions.append(Position(
                    symbol=self.contract.symbol,
                    side=side,
                    quantity=abs(p["netPos"]),
                    entry_price=p.get("netPrice", 0),
                    current_price=p.get("netPrice", 0),  # Updated via WebSocket
                    unrealized_pnl=0,  # Calculated separately
                ))
        return positions

    # ─── ORDER MANAGEMENT ────────────────────────────────────────

    async def place_order(self, order: Order) -> Order:
        """Submit an order to Tradovate."""
        await self._ensure_token()

        # Map our order type to Tradovate's expected format
        action = "Buy" if order.side == OrderSide.BUY else "Sell"

        payload = {
            "accountSpec": self.config.username,
            "accountId": self.account_id,
            "action": action,
            "symbol": self.contract.symbol,
            "orderQty": order.quantity,
            "orderType": self._map_order_type(order.order_type),
            "isAutomated": True,  # Important: flag as automated order
        }

        # Add price for limit orders
        if order.order_type in (OrderType.LIMIT, OrderType.STOP_LIMIT):
            payload["price"] = order.price

        # Add stop price for stop orders
        if order.order_type in (OrderType.STOP, OrderType.STOP_LIMIT):
            payload["stopPrice"] = order.stop_price

        async with self._session.post(
            f"{self.config.base_url}/order/placeOrder",
            json=payload,
            headers=self._headers()
        ) as resp:
            data = await resp.json()

            if "orderId" in data:
                order.id = str(data["orderId"])
                order.status = OrderStatus.PENDING
                order.created_at = datetime.now(timezone.utc)
                logger.info(f"📤 Order placed: {action} {order.quantity}x "
                            f"{self.contract.symbol} | ID: {order.id}")
            else:
                order.status = OrderStatus.REJECTED
                logger.error(f"❌ Order rejected: {data}")

        return order

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order."""
        await self._ensure_token()

        async with self._session.post(
            f"{self.config.base_url}/order/cancelOrder",
            json={"orderId": int(order_id)},
            headers=self._headers()
        ) as resp:
            data = await resp.json()
            success = data.get("commandStatus") == "AtExecution"
            if success:
                logger.info(f"🚫 Order {order_id} cancelled")
            return success

    async def modify_order(self, order_id: str, price: Optional[float] = None,
                           stop_price: Optional[float] = None,
                           quantity: Optional[int] = None) -> Order:
        """Modify an existing order."""
        await self._ensure_token()

        payload = {"orderId": int(order_id)}
        if price is not None:
            payload["price"] = price
        if stop_price is not None:
            payload["stopPrice"] = stop_price
        if quantity is not None:
            payload["orderQty"] = quantity

        async with self._session.post(
            f"{self.config.base_url}/order/modifyOrder",
            json=payload,
            headers=self._headers()
        ) as resp:
            data = await resp.json()

        order = Order(id=order_id)
        order.status = OrderStatus.PENDING
        return order

    async def flatten_all(self) -> bool:
        """
        🚨 EMERGENCY: Close ALL open positions immediately.
        Used when hitting risk limits or during shutdown.
        """
        await self._ensure_token()
        logger.warning("🚨 FLATTENING ALL POSITIONS")

        positions = await self.get_positions()
        for pos in positions:
            # Place opposite market order to close
            close_side = OrderSide.SELL if pos.side == OrderSide.BUY else OrderSide.BUY
            close_order = Order(
                symbol=pos.symbol,
                side=close_side,
                order_type=OrderType.MARKET,
                quantity=pos.quantity,
            )
            await self.place_order(close_order)

        return True

    # ─── MARKET DATA ─────────────────────────────────────────────

    async def get_historical_candles(self, symbol: str, timeframe: str,
                                      count: int) -> list[dict]:
        """
        Fetch historical OHLCV candles from Tradovate.
        timeframe: '1min', '5min', '15min', '1hour', '1day'
        """
        await self._ensure_token()

        # Map timeframe string to Tradovate's elementSize/elementSizeUnit
        tf_map = {
            "1min": (1, "UnderlyingUnits"),
            "5min": (5, "UnderlyingUnits"),
            "15min": (15, "UnderlyingUnits"),
            "1hour": (60, "UnderlyingUnits"),
            "1day": (1, "Day"),
        }
        size, unit = tf_map.get(timeframe, (5, "UnderlyingUnits"))

        params = {
            "symbol": symbol,
            "chartDescription": json.dumps({
                "underlyingType": "MinuteBar",
                "elementSize": size,
                "elementSizeUnit": unit,
                "withHistogram": False,
            }),
            "timeRange": json.dumps({
                "closestTimestamp": datetime.now(timezone.utc).isoformat(),
                "asMuchAsElements": count,
            }),
        }

        # NOTE: Historical chart data in Tradovate is typically fetched via
        # WebSocket using chart subscriptions. This REST approach may vary.
        # In production, you'd subscribe via the market data WebSocket.
        # Below is a simplified version — adjust based on Tradovate's current API.

        async with self._session.get(
            f"{self.config.base_url}/contract/find",
            params={"name": symbol},
            headers=self._headers()
        ) as resp:
            contract = await resp.json()

        logger.info(f"Fetched {count} candles for {symbol} ({timeframe})")
        # Return placeholder — in production, parse actual WebSocket chart data
        return []

    async def subscribe_candles(self, symbol: str, timeframe: str,
                                 callback: Callable) -> None:
        """
        Subscribe to live candle updates via WebSocket.
        The callback receives a dict: {timestamp, open, high, low, close, volume}
        """
        self._candle_callbacks[f"{symbol}_{timeframe}"] = callback

        # Connect to market data WebSocket
        if not self._md_ws:
            self._md_ws = await self._session.ws_connect(self.config.md_url)
            # Authorize on the WebSocket
            await self._md_ws.send_str(
                f"authorize\n0\n\n{self.access_token}"
            )
            logger.info("Connected to market data WebSocket")

        # Subscribe to chart data
        self._ws_request_id += 1
        tf_map = {"1min": 1, "5min": 5, "15min": 15}
        element_size = tf_map.get(timeframe, 5)

        subscribe_msg = {
            "symbol": symbol,
            "chartDescription": {
                "underlyingType": "MinuteBar",
                "elementSize": element_size,
                "elementSizeUnit": "UnderlyingUnits",
                "withHistogram": False,
            },
            "timeRange": {
                "asMuchAsElements": 200,  # Get 200 bars of history + live
            },
        }

        await self._md_ws.send_str(
            f"md/getChart\n{self._ws_request_id}\n\n{json.dumps(subscribe_msg)}"
        )
        logger.info(f"📊 Subscribed to {symbol} {timeframe} candles")

        # Start listening loop (runs in background)
        asyncio.create_task(self._md_listener())

    async def _md_listener(self) -> None:
        """Listen for incoming market data WebSocket messages."""
        try:
            async for msg in self._md_ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    await self._handle_md_message(msg.data)
                elif msg.type in (aiohttp.WSMsgType.ERROR,
                                   aiohttp.WSMsgType.CLOSED):
                    logger.warning("Market data WebSocket closed")
                    break
        except Exception as e:
            logger.error(f"MD listener error: {e}")

    async def _handle_md_message(self, raw: str) -> None:
        """Parse and route incoming market data messages."""
        try:
            # Tradovate WebSocket messages have a specific format:
            # type\nid\n\njson_payload
            lines = raw.split("\n", 3)
            if len(lines) < 4:
                return

            msg_type = lines[0]
            payload = json.loads(lines[3]) if lines[3] else {}

            if msg_type == "chart" and "bars" in payload:
                for bar in payload["bars"]:
                    candle = {
                        "timestamp": bar.get("timestamp"),
                        "open": bar.get("open"),
                        "high": bar.get("high"),
                        "low": bar.get("low"),
                        "close": bar.get("close"),
                        "volume": bar.get("upVolume", 0) + bar.get("downVolume", 0),
                    }
                    # Trigger all registered callbacks
                    for cb in self._candle_callbacks.values():
                        await cb(candle)

        except (json.JSONDecodeError, IndexError):
            pass  # Not all messages are JSON chart data

    # ─── HELPERS ─────────────────────────────────────────────────

    @staticmethod
    def _map_order_type(ot: OrderType) -> str:
        """Map our OrderType enum to Tradovate's string."""
        mapping = {
            OrderType.MARKET: "Market",
            OrderType.LIMIT: "Limit",
            OrderType.STOP: "Stop",
            OrderType.STOP_LIMIT: "StopLimit",
        }
        return mapping.get(ot, "Market")
