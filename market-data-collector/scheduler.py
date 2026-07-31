"""
Live data collector — starts automatically at market open.
Run with: python scheduler.py

Collection begins Monday 9:30 AM ET and runs Mon–Fri during session hours.
"""
import os
import logging
import pytz
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from collector import collect_all

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(LOG_DIR, "collection.log")),
    ],
)
logger = logging.getLogger(__name__)

ET = pytz.timezone("US/Eastern")

# CME Globex: Sun 6 PM – Fri 5 PM ET, daily maintenance 5–6 PM ET
GLOBEX_DAYS  = "sun-fri"
GLOBEX_HOURS = "18-23,0-16"   # skips the 17:xx maintenance window


def make_job(tf, n_bars=5):
    def job():
        collect_all(tf, n_bars=n_bars)
    job.__name__ = f"collect_{tf}"
    return job


def build_scheduler() -> BlockingScheduler:
    scheduler = BlockingScheduler(timezone=ET)

    # 1-minute bars — every minute during Globex session
    scheduler.add_job(
        make_job("1m"),
        CronTrigger(day_of_week=GLOBEX_DAYS, hour=GLOBEX_HOURS, minute="*", timezone=ET),
        id="job_1m",
    )

    # 3-minute bars
    scheduler.add_job(
        make_job("3m"),
        CronTrigger(day_of_week=GLOBEX_DAYS, hour=GLOBEX_HOURS, minute="*/3", timezone=ET),
        id="job_3m",
    )

    # 5-minute bars
    scheduler.add_job(
        make_job("5m"),
        CronTrigger(day_of_week=GLOBEX_DAYS, hour=GLOBEX_HOURS, minute="*/5", timezone=ET),
        id="job_5m",
    )

    # 15-minute bars
    scheduler.add_job(
        make_job("15m"),
        CronTrigger(day_of_week=GLOBEX_DAYS, hour=GLOBEX_HOURS, minute="*/15", timezone=ET),
        id="job_15m",
    )

    # 1-hour bars — top of each hour
    scheduler.add_job(
        make_job("1h"),
        CronTrigger(day_of_week=GLOBEX_DAYS, hour=GLOBEX_HOURS, minute="0", timezone=ET),
        id="job_1h",
    )

    # 4-hour bars — anchored to Globex open (6 PM ET)
    scheduler.add_job(
        make_job("4h"),
        CronTrigger(day_of_week=GLOBEX_DAYS, hour="18,22,2,6,10,14", minute="0", timezone=ET),
        id="job_4h",
    )

    # Daily bar — captured at 4:55 PM ET (just before close)
    scheduler.add_job(
        make_job("1d"),
        CronTrigger(day_of_week="mon-fri", hour=16, minute=55, timezone=ET),
        id="job_1d",
    )

    # Weekly bar — Friday at close
    scheduler.add_job(
        make_job("1w"),
        CronTrigger(day_of_week="fri", hour=16, minute=55, timezone=ET),
        id="job_1w",
    )

    return scheduler


if __name__ == "__main__":
    logger.info("Scheduler starting — CME Globex hours (Sun 6 PM – Fri 5 PM ET)...")
    scheduler = build_scheduler()
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped.")
