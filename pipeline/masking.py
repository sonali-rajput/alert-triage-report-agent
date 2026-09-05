"""Security layer: scrub secrets/PII from alerts before anything is archived
to storage or sent to the LLM, and truncate huge bodies to control token cost.

Rules live in config/masking_patterns.yaml. Enrichment (stack traces,
breadcrumbs, request URLs) is masked with the same rules, so the patterns
must cover the places PII/secrets hide there (query strings, emails, tokens).
"""

from __future__ import annotations

import re
from typing import Any

from shared.config import masking_patterns
from shared.models import Alert


class Masker:
    def __init__(self, config: dict[str, Any] | None = None):
        cfg = config or masking_patterns()
        self._rules: list[tuple[str, re.Pattern, str]] = [
            (rule.get("name", "unnamed"), re.compile(rule["pattern"]), rule["replacement"])
            for rule in cfg.get("patterns", [])
        ]
        trunc = cfg.get("truncation", {})
        self._max_body = int(trunc.get("max_body_chars", 4000))
        self._keep_head = int(trunc.get("keep_head_chars", 2500))
        self._keep_tail = int(trunc.get("keep_tail_chars", 1000))

    def mask_text(self, text: str) -> tuple[str, list[str]]:
        """Returns (masked_text, names_of_rules_that_fired).

        The rule names are not bookkeeping: a `bearer-token` or `aws-access-key`
        hit means an application is logging a credential into its error
        messages. That is a real security finding, and it used to be discarded
        here at zero benefit. It is surfaced in the report's security section.
        """
        hits: list[str] = []
        for name, pattern, replacement in self._rules:
            text, count = pattern.subn(replacement, text)
            if count:
                hits.append(name)
        return text, hits

    def truncate(self, text: str) -> str:
        if len(text) <= self._max_body:
            return text
        omitted = len(text) - self._keep_head - self._keep_tail
        return (
            text[: self._keep_head]
            + f"\n... [{omitted} chars truncated] ...\n"
            + text[-self._keep_tail :]
        )

    def mask_alert(self, alert: Alert) -> Alert:
        title, title_hits = self.mask_text(alert.title)
        body, body_hits = self.mask_text(alert.body)
        # Union with any hits already on the alert: a body can pass through
        # here more than once, and an earlier finding must not be lost when a
        # later pass over a longer body happens not to reproduce it.
        hits = list(dict.fromkeys([*alert.masking_hits, *title_hits, *body_hits]))
        return alert.model_copy(
            update={
                "title": title,
                "body": self.truncate(body),
                "masking_hits": hits,
            }
        )

    def mask_alerts(self, alerts: list[Alert]) -> list[Alert]:
        return [self.mask_alert(a) for a in alerts]
