"""Pre-LLM noise filter: drop alerts matching known-noise rules
(config/noise_filters.yaml) before they cost tokens or attention. Runs inside
the pipeline, after masking and before the alerts are embedded and sent to the
agents."""

from __future__ import annotations

import logging
import re
from typing import Any

from shared.config import noise_filters
from shared.environments import matches_environment
from shared.models import Alert

logger = logging.getLogger(__name__)


class Prefilter:
    def __init__(self, config: dict[str, Any] | None = None):
        cfg = config or noise_filters()
        self._drop_rules: list[tuple[str, re.Pattern]] = [
            (rule["name"], re.compile(rule["pattern"], re.IGNORECASE))
            for rule in cfg.get("drop_patterns", [])
        ]
        self._drop_projects = {p.lower() for p in cfg.get("drop_projects", [])}
        self._drop_environments = list(cfg.get("drop_environments", []))
        self._min_event_count = int(cfg.get("min_event_count", 1))

    def _drop_reason(self, alert: Alert) -> str | None:
        if alert.project.lower() in self._drop_projects:
            return f"project '{alert.project}' is in drop_projects"
        # Canonical comparison, not `.lower() in set`: real Sentry orgs carry
        # environment names like "@dev" and "Local", which an exact match
        # silently fails to drop. See shared/environments.py.
        if matches_environment(alert.environment, self._drop_environments):
            return f"environment '{alert.environment}' is in drop_environments"
        haystack = f"{alert.title} {alert.body}"
        for name, pattern in self._drop_rules:
            if pattern.search(haystack):
                return f"matched noise rule '{name}'"
        # events_24h(), never `event_count`: on the real org-issues endpoint
        # `count` is the issue's ALL-TIME total (a real one reads 206,017 next
        # to 223 events in the last 24h), so comparing it against a
        # window-relative threshold filters on the wrong number entirely.
        events = alert.events_24h()
        if events < self._min_event_count:
            return f"events_24h {events} below minimum {self._min_event_count}"
        return None

    def apply(self, alerts: list[Alert]) -> tuple[list[Alert], int]:
        """Returns (kept_alerts, dropped_count)."""
        kept: list[Alert] = []
        dropped = 0
        for alert in alerts:
            reason = self._drop_reason(alert)
            if reason:
                dropped += 1
                logger.info("prefilter dropped %s (%s): %s", alert.fingerprint(), alert.title[:80], reason)
            else:
                kept.append(alert)
        return kept, dropped
