"""Auto-fetch scheduler for periodic pipeline runs."""

from __future__ import annotations

import logging

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from ppagent.config import AppConfig
from ppagent.pipeline import PaperPipeline

logger = logging.getLogger(__name__)


class PaperScheduler:
    """Runs the paper pipeline on a configurable cron schedule."""

    def __init__(self, config: AppConfig, pipeline: PaperPipeline) -> None:
        self.config = config
        self.pipeline = pipeline

    def start(self) -> None:
        """Start the blocking scheduler."""
        scheduler = BlockingScheduler(timezone=self.config.scheduler.timezone)
        scheduler.add_job(
            self._run_pipeline,
            trigger=CronTrigger(
                hour=self.config.scheduler.cron_hour,
                minute=self.config.scheduler.cron_minute,
            ),
            id="daily-paper-run",
            name="Daily paper discovery and report generation",
            misfire_grace_time=3600,  # 1 hour grace period
        )
        logger.info(
            "Scheduler started: %02d:%02d (%s)",
            self.config.scheduler.cron_hour,
            self.config.scheduler.cron_minute,
            self.config.scheduler.timezone,
        )
        scheduler.start()

    def _run_pipeline(self) -> None:
        """Execute the full pipeline and log results."""
        logger.info("Scheduled pipeline run started")
        try:
            reports = self.pipeline.run(prompt_replace=False, prompt_publish=False)
            logger.info("Scheduled run complete: %d report(s) generated", len(reports))
            for r in reports:
                logger.info("  - %s (%s)", r.paper.title, r.paper.id)
        except Exception as exc:
            logger.exception("Scheduled pipeline run failed: %s", exc)
