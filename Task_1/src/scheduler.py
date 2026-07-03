from __future__ import annotations

import logging

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from .config import AppConfig
from .updater import KnowledgeBaseUpdater

logger = logging.getLogger("kb_system")


def _build_trigger(schedule_cfg: dict):
    sched_type = schedule_cfg.get("type", "interval")
    if sched_type == "interval":
        return IntervalTrigger(
            minutes=schedule_cfg.get("minutes", 60),
            hours=schedule_cfg.get("hours", 0),
            seconds=schedule_cfg.get("seconds", 0),
        )
    if sched_type == "cron":
        cron_kwargs = {k: v for k, v in schedule_cfg.items() if k != "type"}
        return CronTrigger(**cron_kwargs)
    raise ValueError(f"Unknown schedule type: {sched_type}")


class KnowledgeBaseScheduler:
    """
    Registers one scheduled job per source (each with its own cadence) so
    that, e.g., a fast-moving RSS feed can refresh hourly while a static
    document folder only rescans nightly.
    """

    def __init__(self, config: AppConfig, updater: KnowledgeBaseUpdater):
        self.config = config
        self.updater = updater
        self.scheduler = BackgroundScheduler()

    def _run_and_log(self, source_cfg: dict):
        logger.info("Running scheduled update for source '%s'", source_cfg["id"])
        self.updater.update_source(source_cfg)

    def start(self, run_immediately: bool = True):
        for source_cfg in self.config.sources:
            trigger = _build_trigger(source_cfg.get("schedule", {"type": "interval", "minutes": 60}))
            self.scheduler.add_job(
                self._run_and_log,
                trigger=trigger,
                args=[source_cfg],
                id=source_cfg["id"],
                replace_existing=True,
                max_instances=1,
                coalesce=True,
            )
            logger.info("Scheduled source '%s' with trigger %s", source_cfg["id"], trigger)

        self.scheduler.start()

        if run_immediately:
            logger.info("Running initial update pass for all sources before entering scheduled mode...")
            self.updater.update_all()

    def shutdown(self):
        self.scheduler.shutdown(wait=False)
