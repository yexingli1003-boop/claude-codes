"""
Logging Setup
==============
Configures logging for the trading bot with both console and file output.
Trade executions are also logged to a CSV for post-analysis.
"""

import csv
import logging
import os
from datetime import datetime, timezone


def setup_logging(log_level: str = "INFO", log_file: str = "logs/trading.log") -> None:
    """Configure the root logger with console + file handlers."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # Create formatters
    console_fmt = logging.Formatter(
        "%(asctime)s │ %(levelname)-7s │ %(name)-25s │ %(message)s",
        datefmt="%H:%M:%S"
    )
    file_fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    console = logging.StreamHandler()
    console.setLevel(getattr(logging, log_level))
    console.setFormatter(console_fmt)

    # File handler (rotating would be better for production)
    file_handler = logging.FileHandler(log_file, mode="a")
    file_handler.setLevel(logging.DEBUG)  # Log everything to file
    file_handler.setFormatter(file_fmt)

    # Root logger
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(console)
    root.addHandler(file_handler)


class TradeLogger:
    """Logs completed trades to a CSV file for backtesting analysis."""

    def __init__(self, filepath: str = "logs/trades.csv"):
        self.filepath = filepath
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Write header if file doesn't exist
        if not os.path.exists(filepath):
            with open(filepath, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "direction", "symbol", "entry_price",
                    "exit_price", "quantity", "pnl", "pattern",
                    "confirmations", "confidence", "duration_seconds"
                ])

    def log_trade(self, direction: str, symbol: str, entry_price: float,
                  exit_price: float, quantity: int, pnl: float,
                  pattern: str, confirmations: str, confidence: float,
                  duration_seconds: float) -> None:
        """Append a completed trade to the CSV log."""
        with open(self.filepath, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now(timezone.utc).isoformat(),
                direction, symbol, f"{entry_price:.2f}",
                f"{exit_price:.2f}", quantity, f"{pnl:.2f}",
                pattern, confirmations, f"{confidence:.3f}",
                f"{duration_seconds:.1f}"
            ])
