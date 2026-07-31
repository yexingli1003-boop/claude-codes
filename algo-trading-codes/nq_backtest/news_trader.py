# news_trader.py — High-Impact News Scalping Bot
#
# Scrapes Forex Factory economic calendar for high-impact USD events,
# monitors for actual vs expected data, and trades the displacement.
#
# USAGE:
#   python news_trader.py              (paper trading on Alpaca)
#   python news_trader.py --dry-run    (signals only, no orders)
#
# SUPPORTED EVENTS:
#   CPI, Non-Farm Payrolls, PPI, Unemployment Claims,
#   Fed Rate Decision, Fed Chair Speech
#
# DISPLACEMENT LOGIC (per user specification):
#   After entry, monitor price every 100ms:
#   - First 1 second: if price moves 12+ NQ pts in our favor → HOLD 10 sec
#   - First 1 second: if price DOESN'T move 12pts → EXIT immediately
#   - If price moves AGAINST us:
#       → Wait 300ms for rebound
#       → If no rebound → EXIT immediately
#       → If rebound → wait another 500ms
#       → If displaces 12pts in our favor in that 500ms → HOLD 10 sec
#       → If still chopping → EXIT immediately

import os
import sys
import time
import json
import logging
import re
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo

import requests
import pandas as pd
from dotenv import load_dotenv

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame

# ============================================================
# CONFIGURATION
# ============================================================

SYMBOL = "QQQ"
SHARES = 100  # ~1 NQ contract equivalent

# NQ → QQQ scaling
NQ_REF = 25000
DISPLACEMENT_NQ_POINTS = 12  # minimum displacement in NQ points
HOLD_TIME_SECONDS = 10       # hold time after confirmed displacement
AGAINST_WAIT_MS = 300        # wait for rebound if price goes against
REBOUND_WAIT_MS = 500        # additional wait after rebound detected

TIMEZONE_ET = ZoneInfo("America/New_York")
TIMEZONE_CT = ZoneInfo("America/Chicago")

# High-impact events we care about
HIGH_IMPACT_EVENTS = [
    "Non-Farm Employment Change",
    "Nonfarm Payrolls",
    "CPI m/m",
    "CPI y/y",
    "Core CPI m/m",
    "Core CPI y/y",
    "PPI m/m",
    "PPI y/y",
    "Core PPI m/m",
    "Unemployment Claims",
    "Initial Jobless Claims",
    "Unemployment Rate",
    "Federal Funds Rate",
    "FOMC Statement",
    "FOMC Press Conference",
    "Fed Chair Powell Speaks",
    "ISM Manufacturing PMI",
    "ISM Services PMI",
    "GDP q/q",
    "Advance GDP q/q",
    "Retail Sales m/m",
    "Core Retail Sales m/m",
    "PCE Price Index m/m",
    "Core PCE Price Index m/m",
]

# Events where "higher = bullish for USD = bearish for stocks" (inverse)
# Most economic data: strong data = hawkish Fed = stocks down short-term
# But NFP/GDP can be "risk-on" bullish
# We categorize by typical NQ reaction:
BULLISH_IF_BEATS = [
    # Jobs data beating = strong economy = initially bullish NQ
    "Non-Farm Employment Change",
    "Nonfarm Payrolls",
    "GDP q/q",
    "Advance GDP q/q",
    "Retail Sales m/m",
    "Core Retail Sales m/m",
    "ISM Manufacturing PMI",
    "ISM Services PMI",
]

BEARISH_IF_BEATS = [
    # Inflation data beating = more hawkish Fed = bearish NQ
    "CPI m/m",
    "CPI y/y",
    "Core CPI m/m",
    "Core CPI y/y",
    "PPI m/m",
    "PPI y/y",
    "Core PPI m/m",
    "PCE Price Index m/m",
    "Core PCE Price Index m/m",
]

BULLISH_IF_LOWER = [
    # Lower unemployment = strong economy = bullish
    # But also: lower claims = less layoffs = bullish
    "Unemployment Rate",
    "Unemployment Claims",
    "Initial Jobless Claims",
]

FED_EVENTS = [
    "Federal Funds Rate",
    "FOMC Statement",
    "FOMC Press Conference",
    "Fed Chair Powell Speaks",
]

# ============================================================
# LOGGING
# ============================================================


def setup_logging(dry_run=False):
    mode = "DRY-RUN" if dry_run else "PAPER"
    log_file = f"news_trades_{datetime.now().strftime('%Y%m%d')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format=f"[NEWS-{mode}] %(asctime)s.%(msecs)03d | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, mode="a"),
        ],
    )
    return logging.getLogger("NewsTrader")


# ============================================================
# FOREX FACTORY CALENDAR SCRAPER
# ============================================================

def fetch_forex_factory_calendar(target_date=None):
    """
    Fetch today's economic calendar from Forex Factory.
    Returns list of events with: time, event, forecast, previous, actual, impact
    """
    if target_date is None:
        target_date = date.today()

    url = f"https://nfs.faireconomy.media/ff_calendar_thisweek.json"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        events = response.json()

        # Filter for today's USD high-impact events
        today_events = []
        for event in events:
            # Check if it's today and USD
            event_date_str = event.get("date", "")
            try:
                event_date = datetime.fromisoformat(
                    event_date_str.replace("Z", "+00:00"))
                if event_date.date() != target_date:
                    continue
            except:
                continue

            if event.get("country", "") != "USD":
                continue

            if event.get("impact", "") != "High":
                continue

            title = event.get("title", "")

            # Check if it matches our high-impact list
            is_relevant = False
            for hi_event in HIGH_IMPACT_EVENTS:
                if hi_event.lower() in title.lower() or title.lower() in hi_event.lower():
                    is_relevant = True
                    break

            if not is_relevant:
                continue

            today_events.append({
                "title": title,
                "datetime": event_date,
                "time_et": event_date.astimezone(TIMEZONE_ET).strftime("%H:%M"),
                "forecast": event.get("forecast", ""),
                "previous": event.get("previous", ""),
                "actual": event.get("actual", ""),
                "impact": event.get("impact", ""),
            })

        return today_events

    except Exception as e:
        logging.error(f"Failed to fetch Forex Factory calendar: {e}")
        return []


def fetch_forex_factory_with_retry(target_date=None, retries=3):
    """Fetch calendar with retries."""
    for attempt in range(retries):
        events = fetch_forex_factory_calendar(target_date)
        if events:
            return events
        time.sleep(2)
    return []


# ============================================================
# ECONOMIC DATA CHECKER
# ============================================================

def parse_number(value_str):
    """Parse economic data strings like '3.2%', '245K', '0.3%' into float."""
    if not value_str or value_str == "":
        return None
    try:
        cleaned = value_str.strip().replace(",", "")
        cleaned = cleaned.replace("%", "")
        cleaned = cleaned.replace("K", "")
        cleaned = cleaned.replace("M", "")
        cleaned = cleaned.replace("B", "")
        return float(cleaned)
    except (ValueError, AttributeError):
        return None


def determine_trade_direction(event_title, actual, forecast):
    """
    Determine trade direction based on actual vs forecast.
    Returns: 'LONG', 'SHORT', or 'SKIP'
    """
    if actual is None or forecast is None:
        return "SKIP", "Missing data"

    diff = actual - forecast
    tolerance = abs(forecast) * 0.001 if forecast != 0 else 0.001

    # Actual matches forecast → skip
    if abs(diff) <= tolerance:
        return "SKIP", f"Actual {actual} matches forecast {forecast}"

    beats = actual > forecast
    misses = actual < forecast

    # Check event category
    title_lower = event_title.lower()

    # Category 1: Bullish if beats (NFP, GDP, Retail Sales, ISM)
    for event_name in BULLISH_IF_BEATS:
        if event_name.lower() in title_lower:
            if beats:
                return "LONG", f"{event_title}: {actual} > {forecast} (beats → bullish NQ)"
            else:
                return "SHORT", f"{event_title}: {actual} < {forecast} (misses → bearish NQ)"

    # Category 2: Bearish if beats (CPI, PPI, PCE — inflation)
    for event_name in BEARISH_IF_BEATS:
        if event_name.lower() in title_lower:
            if beats:
                return "SHORT", f"{event_title}: {actual} > {forecast} (hot inflation → bearish NQ)"
            else:
                return "LONG", f"{event_title}: {actual} < {forecast} (cool inflation → bullish NQ)"

    # Category 3: Bullish if lower (Unemployment)
    for event_name in BULLISH_IF_LOWER:
        if event_name.lower() in title_lower:
            if misses:  # lower than forecast = good
                return "LONG", f"{event_title}: {actual} < {forecast} (lower unemployment → bullish NQ)"
            else:
                return "SHORT", f"{event_title}: {actual} > {forecast} (higher unemployment → bearish NQ)"

    # Category 4: Fed events
    for event_name in FED_EVENTS:
        if event_name.lower() in title_lower:
            # For rate decisions: lower rate = dovish = bullish NQ
            if "rate" in title_lower:
                if misses:  # lower rate than expected = dovish surprise
                    return "LONG", f"Rate lower than expected ({actual} < {forecast}) → dovish → bullish NQ"
                elif beats:  # higher rate than expected = hawkish
                    return "SHORT", f"Rate higher than expected ({actual} > {forecast}) → hawkish → bearish NQ"
            return "SKIP", "Fed event — need rate data to trade"

    # Default: skip if we can't categorize
    return "SKIP", f"Unknown event type: {event_title}"


def determine_fed_direction(actual_rate, expected_rate, market_wants_cut=True):
    """
    Special handler for Fed rate decisions.
    market_wants_cut: True if market is pricing in a cut
    """
    if actual_rate < expected_rate:
        return "LONG", "Surprise rate cut → dovish → bullish NQ"
    elif actual_rate > expected_rate:
        return "SHORT", "Surprise rate hike → hawkish → bearish NQ"
    else:
        if market_wants_cut:
            return "SKIP", "Rate as expected — no surprise"
        else:
            return "SKIP", "Rate as expected — no surprise"


# ============================================================
# DISPLACEMENT MONITOR
# ============================================================

def nq_to_qqq(nq_points, qqq_price):
    """Convert NQ points to QQQ dollar movement."""
    return nq_points * (qqq_price / NQ_REF)


class DisplacementMonitor:
    """
    Monitors price after entry to determine if displacement is confirmed.

    Logic (all NQ point references are scaled to QQQ dollars):

    Phase 1 — First 1 second after entry:
      - If price moves 12+ NQ pts in our favor → CONFIRMED, hold 10 sec
      - If price moves against us → go to Phase 2
      - If price chops (no 12pt move) after 1 sec → EXIT

    Phase 2 — Price went against us:
      - Wait 300ms for rebound
      - If no rebound after 300ms → EXIT immediately
      - If rebound detected → go to Phase 3

    Phase 3 — Rebound detected, wait 500ms more:
      - If price displaces 12pts in our favor within 500ms → CONFIRMED, hold 10 sec
      - If still chopping after 500ms → EXIT
    """

    def __init__(self, entry_price, direction, qqq_price, log):
        self.entry_price = entry_price
        self.direction = direction
        self.qqq_price = qqq_price
        self.log = log

        # Displacement threshold in QQQ dollars
        self.displacement_threshold = nq_to_qqq(
            DISPLACEMENT_NQ_POINTS, qqq_price)
        # Against threshold (any move against us)
        # 3 NQ pts against = trigger rebound check
        self.against_threshold = nq_to_qqq(3, qqq_price)

        self.log.info(
            f"  Displacement monitor started | Entry: ${entry_price:.2f} | "
            f"Direction: {direction} | "
            f"Threshold: ${self.displacement_threshold:.2f} ({DISPLACEMENT_NQ_POINTS} NQ pts)"
        )

    def check_displacement(self, current_price):
        """Check if price has displaced enough in our favor."""
        if self.direction == "LONG":
            move = current_price - self.entry_price
        else:
            move = self.entry_price - current_price
        return move >= self.displacement_threshold, move

    def check_against(self, current_price):
        """Check if price moved against us."""
        if self.direction == "LONG":
            move = self.entry_price - current_price
        else:
            move = current_price - self.entry_price
        return move > self.against_threshold, move

    def check_rebound(self, current_price):
        """Check if price is rebounding back toward our direction."""
        if self.direction == "LONG":
            return current_price > self.entry_price
        else:
            return current_price < self.entry_price


# ============================================================
# MAIN NEWS TRADER
# ============================================================

class NewsTrader:
    def __init__(self, dry_run=False):
        self.dry_run = dry_run
        self.log = setup_logging(dry_run)

        load_dotenv()
        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")

        if not api_key or not secret_key:
            raise ValueError(
                "ALPACA_API_KEY and ALPACA_SECRET_KEY required in .env")

        self.trading_client = TradingClient(api_key, secret_key, paper=True)
        self.data_client = StockHistoricalDataClient(api_key, secret_key)

        self.current_qqq_price = 607.0
        self.todays_events = []
        self.traded_events = set()  # Don't trade same event twice

    def get_current_price(self):
        """Get latest QQQ price."""
        try:
            request = StockLatestQuoteRequest(symbol_or_symbols=SYMBOL)
            quote = self.data_client.get_stock_latest_quote(request)
            if SYMBOL in quote:
                mid = (float(quote[SYMBOL].ask_price) +
                       float(quote[SYMBOL].bid_price)) / 2
                self.current_qqq_price = mid
                return mid
        except Exception as e:
            self.log.error(f"Price fetch error: {e}")

        return self.current_qqq_price

    def place_order(self, direction):
        """Place market order. Returns order ID or None."""
        side = OrderSide.BUY if direction == "LONG" else OrderSide.SELL

        self.log.info(f"  PLACING ORDER: {side.value} {SHARES} {SYMBOL}")

        if self.dry_run:
            self.log.info("  [DRY-RUN] Simulated order.")
            return f"DRY-{direction}-{int(time.time())}"

        try:
            order_request = MarketOrderRequest(
                symbol=SYMBOL,
                qty=SHARES,
                side=side,
                time_in_force=TimeInForce.DAY,
            )
            order = self.trading_client.submit_order(order_request)
            self.log.info(f"  Order filled. ID: {order.id}")
            return str(order.id)
        except Exception as e:
            self.log.error(f"  ORDER FAILED: {e}")
            return None

    def close_position(self, direction, reason):
        """Close the position."""
        close_side = OrderSide.SELL if direction == "LONG" else OrderSide.BUY

        self.log.info(
            f"  CLOSING: {close_side.value} {SHARES} {SYMBOL} | Reason: {reason}")

        if self.dry_run:
            self.log.info("  [DRY-RUN] Simulated close.")
            return

        try:
            order_request = MarketOrderRequest(
                symbol=SYMBOL,
                qty=SHARES,
                side=close_side,
                time_in_force=TimeInForce.DAY,
            )
            self.trading_client.submit_order(order_request)
            self.log.info("  Close order submitted.")
        except Exception as e:
            self.log.error(f"  CLOSE FAILED: {e}")

    def monitor_displacement(self, direction, entry_price):
        """
        Monitor price movement after entry.
        Implements the full displacement logic.

        Returns: (pnl_dollars, hold_time_seconds, exit_reason)
        """
        monitor = DisplacementMonitor(
            entry_price, direction, self.current_qqq_price, self.log)

        # ══════════════════════════════════════════════
        # PHASE 1: First 1 second — check for displacement
        # ══════════════════════════════════════════════
        self.log.info("  PHASE 1: Watching for displacement (1 second)...")

        phase1_start = time.time()
        went_against = False

        while time.time() - phase1_start < 1.0:
            current = self.get_current_price()
            displaced, move = monitor.check_displacement(current)

            if displaced:
                # Displacement confirmed! Hold for 10 seconds
                self.log.info(
                    f"  DISPLACEMENT CONFIRMED in {(time.time()-phase1_start)*1000:.0f}ms! "
                    f"Move: ${move:.2f} | Holding {HOLD_TIME_SECONDS}s..."
                )
                return self._hold_and_exit(direction, entry_price, HOLD_TIME_SECONDS, phase1_start)

            # Check if price went against us
            against, against_move = monitor.check_against(current)
            if against:
                went_against = True
                self.log.info(
                    f"  Price moved against us: -${against_move:.2f}. Entering Phase 2...")
                break

            time.sleep(0.1)  # Poll every 100ms

        # If we completed 1 second without displacement and no against move
        if not went_against:
            elapsed = time.time() - phase1_start
            current = self.get_current_price()
            self.log.info(
                f"  NO DISPLACEMENT after {elapsed:.1f}s. Price: ${current:.2f}. Exiting."
            )
            self.close_position(direction, "No displacement in 1s")
            pnl = self._calc_pnl(direction, entry_price, current)
            return pnl, elapsed, "NO_DISPLACEMENT"

        # ══════════════════════════════════════════════
        # PHASE 2: Price went against us — wait 300ms for rebound
        # ══════════════════════════════════════════════
        self.log.info(f"  PHASE 2: Waiting 300ms for rebound...")

        phase2_start = time.time()
        rebound_detected = False

        while time.time() - phase2_start < (AGAINST_WAIT_MS / 1000):
            current = self.get_current_price()

            if monitor.check_rebound(current):
                rebound_detected = True
                self.log.info(
                    f"  REBOUND detected at ${current:.2f} "
                    f"({(time.time()-phase2_start)*1000:.0f}ms). Entering Phase 3..."
                )
                break

            time.sleep(0.05)  # Poll every 50ms

        if not rebound_detected:
            # No rebound after 300ms — exit immediately
            current = self.get_current_price()
            elapsed = time.time() - phase1_start
            self.log.info(
                f"  NO REBOUND after 300ms. Price: ${current:.2f}. Exiting.")
            self.close_position(direction, "No rebound after 300ms")
            pnl = self._calc_pnl(direction, entry_price, current)
            return pnl, elapsed, "NO_REBOUND"

        # ══════════════════════════════════════════════
        # PHASE 3: Rebound detected — wait 500ms more for displacement
        # ══════════════════════════════════════════════
        self.log.info(
            f"  PHASE 3: Rebound found. Watching 500ms for displacement...")

        phase3_start = time.time()

        while time.time() - phase3_start < (REBOUND_WAIT_MS / 1000):
            current = self.get_current_price()
            displaced, move = monitor.check_displacement(current)

            if displaced:
                self.log.info(
                    f"  DISPLACEMENT CONFIRMED after rebound! "
                    f"Move: ${move:.2f} | Holding {HOLD_TIME_SECONDS}s..."
                )
                return self._hold_and_exit(direction, entry_price, HOLD_TIME_SECONDS, phase1_start)

            time.sleep(0.05)  # Poll every 50ms

        # 500ms passed without displacement — exit
        current = self.get_current_price()
        elapsed = time.time() - phase1_start
        self.log.info(
            f"  No displacement after rebound. Price: ${current:.2f}. Exiting.")
        self.close_position(direction, "No displacement after rebound")
        pnl = self._calc_pnl(direction, entry_price, current)
        return pnl, elapsed, "REBOUND_NO_DISPLACEMENT"

    def _hold_and_exit(self, direction, entry_price, hold_seconds, start_time):
        """Hold position for specified seconds, monitoring for reversal, then exit."""
        hold_start = time.time()

        while time.time() - hold_start < hold_seconds:
            current = self.get_current_price()

            # Check for major reversal during hold (price crosses entry by significant amount)
            if direction == "LONG":
                reversal = entry_price - current
            else:
                reversal = current - entry_price

            # If we lose all our gains and go 6+ NQ pts against entry, cut
            reversal_threshold = nq_to_qqq(10, self.current_qqq_price)
            if reversal > reversal_threshold:
                elapsed = time.time() - start_time
                self.log.info(
                    f"  REVERSAL during hold! Lost ${reversal:.2f}. Cutting at ${current:.2f}"
                )
                self.close_position(direction, "Reversal during hold")
                pnl = self._calc_pnl(direction, entry_price, current)
                return pnl, elapsed, "REVERSAL_CUT"

            remaining = hold_seconds - (time.time() - hold_start)
            if int(remaining) != int(remaining + 0.5):
                self.log.info(
                    f"  Holding... {remaining:.1f}s left | Price: ${current:.2f}")

            time.sleep(0.5)  # Check every 500ms during hold

        # Hold complete — exit
        current = self.get_current_price()
        elapsed = time.time() - start_time
        self.log.info(
            f"  HOLD COMPLETE ({hold_seconds}s). Exiting at ${current:.2f}")
        self.close_position(direction, f"Hold {hold_seconds}s complete")
        pnl = self._calc_pnl(direction, entry_price, current)
        return pnl, elapsed, "HOLD_COMPLETE"

    def _calc_pnl(self, direction, entry, exit_price):
        if direction == "LONG":
            return (exit_price - entry) * SHARES
        else:
            return (entry - exit_price) * SHARES

    def poll_for_actual_data(self, event, timeout_seconds=120):
        """
        Poll Forex Factory for the actual data release.
        Returns actual value when it appears, or None on timeout.
        """
        self.log.info(
            f"  Polling for actual data (timeout: {timeout_seconds}s)...")
        start = time.time()

        while time.time() - start < timeout_seconds:
            try:
                events = fetch_forex_factory_calendar(date.today())
                for e in events:
                    if e["title"] == event["title"]:
                        actual = parse_number(e.get("actual", ""))
                        if actual is not None:
                            self.log.info(
                                f"  ACTUAL DATA RECEIVED: {e['actual']}")
                            return actual, e.get("actual", "")
            except:
                pass

            time.sleep(1)  # Poll every 1 second

        self.log.warning(f"  Timeout waiting for actual data.")
        return None, None

    def run(self):
        """Main loop — fetch calendar, wait for events, trade."""
        self.log.info("=" * 70)
        self.log.info("  NEWS SCALPING BOT")
        self.log.info(
            f"  Mode: {'DRY-RUN' if self.dry_run else 'PAPER TRADING'}")
        self.log.info(f"  Symbol: {SYMBOL} | Shares: {SHARES}")
        self.log.info(f"  Displacement: {DISPLACEMENT_NQ_POINTS} NQ pts")
        self.log.info(
            f"  Hold time: {HOLD_TIME_SECONDS}s after confirmed displacement")
        self.log.info("=" * 70)

        # Verify connection
        try:
            account = self.trading_client.get_account()
            self.log.info(f"  Account: {account.account_number}")
            self.log.info(f"  Cash: ${float(account.cash):,.2f}")
        except Exception as e:
            self.log.error(f"Connection failed: {e}")
            return

        # Get today's price for scaling
        self.get_current_price()
        displacement_qqq = nq_to_qqq(
            DISPLACEMENT_NQ_POINTS, self.current_qqq_price)
        self.log.info(f"  QQQ: ${self.current_qqq_price:.2f}")
        self.log.info(f"  12 NQ pts = ${displacement_qqq:.2f} QQQ")
        self.log.info("=" * 70)

        total_pnl = 0
        trades_taken = []

        while True:
            try:
                now = datetime.now(TIMEZONE_ET)

                # Fetch today's calendar
                if not self.todays_events or now.hour == 0 and now.minute < 5:
                    self.log.info("Fetching today's economic calendar...")
                    self.todays_events = fetch_forex_factory_with_retry(
                        now.date())

                    if self.todays_events:
                        self.log.info(
                            f"Found {len(self.todays_events)} high-impact USD events today:")
                        for e in self.todays_events:
                            self.log.info(
                                f"  {e['time_et']} ET | {e['title']} | "
                                f"Forecast: {e['forecast']} | Previous: {e['previous']}"
                            )
                    else:
                        self.log.info(
                            "No high-impact events today. Checking again in 1 hour...")
                        time.sleep(3600)
                        continue

                # Check each upcoming event
                for event in self.todays_events:
                    event_key = f"{event['title']}_{event['time_et']}"
                    if event_key in self.traded_events:
                        continue

                    event_time = event["datetime"].astimezone(TIMEZONE_ET)
                    time_until = (event_time - now).total_seconds()

                    # 5 minutes before event — get ready
                    if 0 < time_until <= 300:
                        self.log.info(
                            f"\n{'='*50}"
                            f"\n  UPCOMING: {event['title']} in {time_until:.0f}s"
                            f"\n  Forecast: {event['forecast']} | Previous: {event['previous']}"
                            f"\n{'='*50}"
                        )

                        # Update QQQ price
                        self.get_current_price()
                        forecast = parse_number(event.get("forecast", ""))

                        self.log.info(f"  QQQ: ${self.current_qqq_price:.2f}")
                        self.log.info(f"  Waiting for event time...")

                        # Wait until event time
                        while datetime.now(TIMEZONE_ET) < event_time:
                            time.sleep(0.1)

                        self.log.info(f"\n  EVENT TIME! Polling for data...")

                        # Poll for actual data
                        actual, actual_str = self.poll_for_actual_data(
                            event, timeout_seconds=120)

                        if actual is None:
                            self.log.info(
                                "  Could not get actual data. Skipping.")
                            self.traded_events.add(event_key)
                            continue

                        # Determine trade direction
                        direction, reason = determine_trade_direction(
                            event["title"], actual, forecast
                        )

                        self.log.info(
                            f"  Actual: {actual_str} | Forecast: {event['forecast']}")
                        self.log.info(f"  Decision: {direction} | {reason}")

                        if direction == "SKIP":
                            self.log.info("  SKIPPING — no trade.")
                            self.traded_events.add(event_key)
                            continue

                        # ── EXECUTE TRADE ──
                        entry_price = self.get_current_price()
                        order_id = self.place_order(direction)

                        if order_id:
                            self.log.info(
                                f"\n  TRADE ENTERED: {direction} {SHARES} {SYMBOL} @ ${entry_price:.2f}"
                            )

                            # Monitor displacement
                            pnl, hold_time, exit_reason = self.monitor_displacement(
                                direction, entry_price
                            )

                            total_pnl += pnl
                            trades_taken.append({
                                "event": event["title"],
                                "direction": direction,
                                "entry": entry_price,
                                "pnl": pnl,
                                "hold_time": hold_time,
                                "exit_reason": exit_reason,
                                "actual": actual_str,
                                "forecast": event["forecast"],
                            })

                            self.log.info(
                                f"\n  TRADE RESULT: ${pnl:+,.2f} | "
                                f"Hold: {hold_time:.1f}s | Reason: {exit_reason}"
                            )
                            self.log.info(f"  Daily PnL: ${total_pnl:+,.2f}")

                        self.traded_events.add(event_key)

                    # Event already passed
                    elif time_until <= 0:
                        if event_key not in self.traded_events:
                            self.traded_events.add(event_key)

                # No imminent events — sleep and check again
                next_event_seconds = float('inf')
                for event in self.todays_events:
                    event_key = f"{event['title']}_{event['time_et']}"
                    if event_key not in self.traded_events:
                        t = (event["datetime"].astimezone(
                            TIMEZONE_ET) - now).total_seconds()
                        if t > 0:
                            next_event_seconds = min(next_event_seconds, t)

                if next_event_seconds == float('inf'):
                    self.log.info(
                        "All events done for today. Waiting for tomorrow...")
                    # Print summary
                    if trades_taken:
                        self.log.info(f"\n{'='*50}")
                        self.log.info(f"  TODAY'S NEWS TRADES")
                        self.log.info(f"{'='*50}")
                        for t in trades_taken:
                            self.log.info(
                                f"  {t['event']} | {t['direction']} | "
                                f"${t['pnl']:+,.2f} | {t['exit_reason']}"
                            )
                        self.log.info(f"  TOTAL: ${total_pnl:+,.2f}")
                        self.log.info(f"{'='*50}")

                    time.sleep(3600)  # Check again in 1 hour
                elif next_event_seconds > 600:
                    self.log.info(
                        f"Next event in {next_event_seconds/60:.0f} min. "
                        f"Sleeping 5 min..."
                    )
                    time.sleep(300)
                elif next_event_seconds > 60:
                    self.log.info(
                        f"Next event in {next_event_seconds/60:.1f} min. "
                        f"Checking every 30s..."
                    )
                    time.sleep(30)
                else:
                    time.sleep(1)  # Event very close, poll fast

            except KeyboardInterrupt:
                self.log.info("Shutdown requested.")
                break

            except Exception as e:
                self.log.error(f"Error: {e}", exc_info=True)
                time.sleep(30)


# ============================================================
# ENTRY POINT
# ============================================================

def main():
    load_dotenv()
    dry_run = "--dry-run" in sys.argv

    trader = NewsTrader(dry_run=dry_run)
    trader.run()


if __name__ == "__main__":
    main()
