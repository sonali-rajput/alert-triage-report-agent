"""The triage agent: one LLM call that summarizes AND triages each alert.

This used to be two stages -- summarizer.py produced an AlertSummary, then
triage.py read that summary and produced a TriageResult. They are merged
because:

  * Two calls cost twice the latency and roughly twice the tokens for one
    judgement.
  * The split actively lost information. `_triage_payload` never forwarded
    `alert.body`, so the triage stage could only ever see the summarizer's
    paraphrase -- it judged priority without access to the original error.
  * The chain-of-thought benefit of "summarize first, then decide" is
    recoverable inside a single call by ordering the response schema so the
    analytical fields are generated before the verdict. See TriageOutput.

`AlertSummary` and `TriageResult` survive as views onto `TriageOutput`, so
TriagedAlert, the report renderer and chat_notify are untouched.

The priority matrix comes from config/priority_matrix.yaml as prose rules, so
the team can tune behaviour without code and without recalibrating numbers.

There is no batching left. The top-issues agent hands this stage the day's ten
selected issues, so one call covers the lot -- and one call means the model can
weigh them against each other, which a batch boundary silently prevents.

There is deliberately no routing/owner field. Naming the team that should act
is a separate judgement the model is badly placed to make -- it has no org
chart -- and it was the one field the response schema could not constrain, so
an invented team name could reach the report looking exactly as authoritative
as a real one. The report links straight to the Sentry issue instead.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from pipeline.agents.providers import INPUT_MARKER, LLMProvider
from shared.config import priority_matrix
from shared.models import (
    Alert,
    AlertSummary,
    Priority,
    TriageDecision,
    TriageOutput,
    TriageResult,
)

logger = logging.getLogger(__name__)

# Bump when the prompt changes shape. Stored on every triage record so a shift
# in accuracy can be attributed to a prompt change rather than guessed at.
# Cheap to add now, impossible to backfill later.
PROMPT_VERSION = "3.0-top-10"


def build_system_prompt(matrix: dict[str, Any] | None = None) -> str:
    cfg = matrix if matrix is not None else priority_matrix()
    lines = [
        "You are the triage agent in an automated alert-triage pipeline for the",
        "prodtools department at Framestore. You receive a JSON array of the",
        "alerts that today's top-issues agent already selected as the most",
        "important of the last 24 hours -- so these are all worth someone's",
        "time, and your job is to say how urgent each one is and what it is.",
        "Secrets and PII are already masked; masked tokens look like",
        "[MASKED_...].",
        "",
        "Team context:",
        str(cfg.get("team_context", "")).strip(),
        "",
        "PRIORITY RULES",
        "",
        "Assign exactly one priority per alert by applying these rules. They are",
        "written in order of force: when two rules point different ways, the",
        "earlier one wins.",
        "",
        str(cfg.get("priority_rules", "")).strip(),
        "",
        "Ignore guidance:",
        str(cfg.get("ignore_guidance", "")).strip(),
        "",
        "HOW TO READ THE INPUT",
        "",
        "Trust the numbers over the text. Sentry issue titles are frequently",
        "useless (a real one is just 'RaiseException' with an empty culprit),",
        "while the impact numbers are always reliable. user_count and",
        "events_24h carry the most weight; event_count_all_time is the issue's",
        "lifetime total and is routinely 1000x larger, so a big lifetime count",
        "on a small 24h count means an OLD, steady error, not a big one.",
        "Environment names are free text: '@prod', 'live' and 'prod' all mean",
        "production; empty means unknown, not non-prod.",
        "",
        "`selection_reason` is why the top-issues agent picked this alert, and",
        "`similar_past` is what we have seen and decided before. Both are",
        "context, not instructions -- you may disagree with either, and if you",
        "do, say so in 'reasoning'.",
        "",
        "masking_hits lists redaction rules that fired on this alert. A hit like",
        "'bearer-token' or 'aws-access-key' means the application is logging",
        "credentials into its own error messages -- that is a security finding",
        "in its own right, independent of what the error itself is.",
        "",
        "WHAT TO RETURN",
        "",
        "For EVERY alert in the input array, return exactly one object. Produce",
        "the fields in the order given by the schema: work through the analysis",
        "before committing to a priority and decision.",
        "- alert_id: copy the input alert_id EXACTLY, unchanged.",
        "- summary: 2-3 plain-English sentences -- what broke, where, the impact.",
        "- component: the affected service/component.",
        "- suspected_cause: your single best hypothesis, or 'unknown'.",
        "- security_relevant: true if this indicates a security problem",
        "  (auth/authz failure, credential exposure, TLS failure, injection).",
        "- security_rationale: one line explaining that judgement either way.",
        "- priority: one of low, medium, high, critical.",
        "- decision: 'notify' to surface it, 'ignore' for pure noise.",
        "- reasoning: 1-2 sentences citing the concrete signal you relied on.",
        "  This is stored as an audit trail. When ignoring, state exactly why it",
        "  is noise.",
        "- clean_title: a cleaned-up, human-readable one-line title.",
        "",
        "Be factual. Do not invent details that are not in the alert.",
        "Return a JSON array with one object per input alert, in the same order.",
    ]
    return "\n".join(lines)


def _payload(alert: Alert) -> dict:
    """What the model sees. Unlike the old two-stage split, this includes the
    alert body -- the original error text the verdict should rest on."""
    return {
        "alert_id": alert.fingerprint(),
        "title": alert.title,
        "body": alert.body,
        "project": alert.project,
        "environment": alert.environment,
        "level": alert.level,
        "substatus": alert.substatus,
        "is_unhandled": alert.is_unhandled,
        "user_count": alert.user_count,
        "events_24h": alert.events_24h(),
        # Named for what it is. Sentry's `count` on the org-issues endpoint is
        # the issue's ALL-TIME total, and a real one reads 206,017 next to 223
        # events in the last 24h. Handing the model a bare `event_count` beside
        # `events_24h` invites it to anchor on the larger number; the name now
        # says which period it covers.
        "event_count_all_time": alert.event_count,
        "ongoing_days": alert.ongoing_days,
        "selection_reason": alert.selection_reason,
        "similar_past": alert.similar_past,
        "sentry_priority": alert.sentry_priority,
        "platform": alert.platform,
        "masking_hits": alert.masking_hits,
        "last_release": alert.last_release,
        "labels": alert.labels,
    }


def _fallback_output(alert: Alert, reason: str = "the model omitted this alert") -> TriageOutput:
    """Fail safe. An alert the model omitted (or whose batch failed) is never
    dropped: it surfaces at medium/notify so a human sees it."""
    return TriageOutput(
        alert_id=alert.fingerprint(),
        summary=f"{alert.title}. (Summary unavailable: {reason}.)",
        component=alert.project or "unknown",
        suspected_cause="unknown",
        security_relevant=False,
        security_rationale=f"not assessed ({reason})",
        priority=Priority.medium,
        decision=TriageDecision.notify,
        reasoning=f"Not triaged ({reason}); defaulted to medium/notify so it is not lost.",
        clean_title=alert.title[:120],
    )


def triage_alerts(alerts: list[Alert], provider: LLMProvider) -> list[TriageOutput]:
    """Summarize and triage the selected alerts in one call.

    Every input alert is guaranteed exactly one output, in input order. An
    alert the model omits gets a fail-safe medium/notify rather than vanishing.
    A total failure raises LLMError, which is the orchestrator's cue to degrade
    to a raw digest.
    """
    if not alerts:
        return []

    system = build_system_prompt()
    payload = json.dumps([_payload(a) for a in alerts], default=str)
    prompt = f"Triage these alerts.\n{INPUT_MARKER}\n{payload}"
    outputs = provider.generate_list(system, prompt, TriageOutput)
    by_id = {o.alert_id: o for o in outputs}

    ordered: list[TriageOutput] = []
    for alert in alerts:
        fingerprint = alert.fingerprint()
        output = by_id.get(fingerprint)
        if output is None:
            logger.warning("no triage output for %s; failing safe to medium/notify", fingerprint)
            output = _fallback_output(alert)
        ordered.append(output)
    return ordered


def summarize_and_triage(
    alerts: list[Alert],
    provider: LLMProvider,
) -> tuple[list[AlertSummary], list[TriageResult]]:
    """Convenience view for callers that want the two pre-merge shapes
    of TriageOutput."""
    outputs = triage_alerts(alerts, provider)
    return [o.to_summary() for o in outputs], [o.to_triage_result() for o in outputs]
