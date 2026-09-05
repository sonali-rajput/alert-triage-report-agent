"""Google Chat notifications via an incoming webhook.

The card is a **front door, not the report**. Someone opening Chat at 08:00
wants three things in about five seconds: how bad is today, what changed since
yesterday, and what should I look at first. Everything else lives in the PDF.

That is why the rows lead with impact numbers rather than the LLM's prose: a
summary sentence is the slowest thing on a card to read, and the numbers are
what actually decide whether you click through.

Incoming webhooks are one-way by construction, so buttons here can only
`openLink`. Interactive card actions would need a Chat app with its own
identity and a public callback service; that is deliberately out of scope.
"""

from __future__ import annotations

import html
import logging

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from shared.models import Alert, Priority, RunStats, TriagedAlert, TriageDecision

logger = logging.getLogger(__name__)

_PRIORITY_EMOJI = {
    Priority.critical: "🔴",
    Priority.high: "🟠",
    Priority.medium: "🟡",
    Priority.low: "⚪",
}
_PRIORITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}
_PRIORITY_ICON = {
    Priority.critical: "https://fonts.gstatic.com/s/i/googlematerialicons/error/v14/gm_grey-24dp/1x/gm_error_gm_grey_24dp.png",
    Priority.high: "https://fonts.gstatic.com/s/i/googlematerialicons/warning/v13/gm_grey-24dp/1x/gm_warning_gm_grey_24dp.png",
}

# How many alerts the card shows. Three, and then the report.
VISIBLE_ALERTS = 3


@retry(
    retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    stop=stop_after_attempt(4),
    reraise=True,
)
def _post(webhook_url: str, payload: dict) -> None:
    resp = httpx.post(webhook_url, json=payload, timeout=30)
    resp.raise_for_status()


def _impact_line(r: TriagedAlert) -> str:
    """The bottom line of a row: what makes this worth someone's morning.

    Ordered by how quickly each fact changes a reader's mind — people affected
    first, then volume, then what is different about today.
    """
    parts: list[str] = []
    if r.user_count:
        parts.append(f"{r.user_count} user" + ("" if r.user_count == 1 else "s"))
    events = r.events_24h or r.event_count
    if events:
        parts.append(f"{events} event" + ("" if events == 1 else "s"))
    if r.substatus in ("new", "regressed", "escalating"):
        parts.append(r.substatus)
    if r.is_unhandled:
        parts.append("unhandled")
    if r.ongoing_days > 1:
        parts.append(f"ongoing {r.ongoing_days}d")
    return " · ".join(parts) or "no impact data"


def _alert_widget(r: TriagedAlert) -> dict:
    """One dense row: priority/project/env on top, title in the middle, impact
    numbers underneath, Sentry link on the right.

    This replaces the old three-line HTML blob plus a separate full-width
    button per alert, which cost roughly four lines of vertical space each.
    """
    emoji = _PRIORITY_EMOJI.get(r.triage.priority, "⚪")
    env = r.environment or "unknown"
    label = f"{emoji} {r.triage.priority.value.upper()} · {r.project} · {env}"
    # Rows follow the top-issues agent's ranking, so a LOW-priority alert can
    # appear near the top. Without this the card looks broken; with it, the
    # divergence between the two agents is the interesting part. Matches
    # section 7 of the report.
    if r.disagreement:
        label += f" · ⚠ ranked #{r.rank} today"
    # Chat renders an HTML subset in decoratedText, so Sentry titles and LLM
    # output (both of which can legitimately contain "<") must be escaped or a
    # stray angle bracket breaks the widget.
    widget: dict = {
        "decoratedText": {
            "topLabel": label,
            "text": f"<b>{html.escape(r.summary.title)}</b>",
            "bottomLabel": _impact_line(r),
            "wrapText": True,
        }
    }
    icon = _PRIORITY_ICON.get(r.triage.priority)
    if icon:
        widget["decoratedText"]["startIcon"] = {"iconUrl": icon}
    if r.url:
        widget["decoratedText"]["button"] = {
            "text": "Sentry",
            "onClick": {"openLink": {"url": r.url}},
        }
    return widget


def _delta_line(results: list[TriagedAlert]) -> str:
    """"What changed since I last looked" — free from Sentry's substatus, and
    the question someone opening this at 08:00 is actually asking."""
    counts = {
        status: sum(1 for r in results if r.substatus == status)
        for status in ("new", "regressed", "escalating")
    }
    parts = [f"{n} {status}" for status, n in counts.items() if n]
    return "▲ " + " · ".join(parts) if parts else "No new, regressed or escalating alerts"


def post_run_summary(
    webhook_url: str,
    run_date: str,
    stats: RunStats,
    results: list[TriagedAlert],
    pdf_url: str = "",
) -> None:
    if not webhook_url:
        logger.warning("CHAT_WEBHOOK_URL not set; skipping Chat notification")
        return

    notify = [r for r in results if r.triage.decision == TriageDecision.notify]
    # The top-issues agent's ranking, so the card and the report's Top 10
    # agree. Priority breaks ties, and an unranked alert (rank 0, which only
    # happens on the fail-safe path) sorts last rather than first.
    ranked = sorted(
        notify,
        key=lambda r: (r.rank if r.rank > 0 else 10**6, _PRIORITY_ORDER.get(r.triage.priority.value, 9)),
    )
    # The delta line describes the same population the rows are drawn from --
    # counting ignored alerts would let the card say "3 new" and show none.
    delta_line = _delta_line(notify)

    counts = " ".join(
        f"{_PRIORITY_EMOJI[Priority(level)]} {stats.by_priority.get(level, 0)} {level}"
        for level in ("critical", "high", "medium", "low")
        if stats.by_priority.get(level, 0)
    )
    funnel = (
        f"{stats.ingested} ingested → {stats.prefiltered} noise → "
        f"{stats.deduped} duplicates → {stats.processed} triaged"
    )

    sections: list[dict] = [
        {
            "widgets": [
                {"decoratedText": {"text": counts or "No alerts triaged today.", "wrapText": True}},
                {"decoratedText": {"text": f"<font color=\"#6c6f85\">{funnel}</font>", "wrapText": True}},
                {"decoratedText": {"text": f"<font color=\"#6c6f85\">{delta_line}</font>", "wrapText": True}},
            ]
        }
    ]

    if ranked:
        sections.append({"widgets": [_alert_widget(r) for r in ranked[:VISIBLE_ALERTS]]})
    else:
        sections.append({"widgets": [{"decoratedText": {"text": "✅ No alerts require attention today."}}]})

    if pdf_url and pdf_url.startswith("http"):
        sections.append(
            {
                "widgets": [
                    {
                        "buttonList": {
                            "buttons": [{"text": "📄 Full report (PDF)", "onClick": {"openLink": {"url": pdf_url}}}]
                        }
                    }
                ]
            }
        )

    payload = {
        "cardsV2": [
            {
                "cardId": f"triage-{run_date}",
                "card": {
                    "header": {
                        "title": "Daily Alert Triage",
                        "subtitle": f"prodtools · {run_date}",
                    },
                    "sections": sections,
                },
            }
        ]
    }
    _post(webhook_url, payload)
    logger.info("posted run summary to Google Chat")


def post_fallback_digest(webhook_url: str, run_date: str, alerts: list[Alert], error: str) -> None:
    """LLM failure path: post the raw prefiltered alert list so nothing is
    silently dropped."""
    if not webhook_url:
        logger.warning("CHAT_WEBHOOK_URL not set; cannot post fallback digest")
        return
    lines = [
        f"⚠️ *Alert triage degraded — {run_date}*",
        f"The AI triage stage failed ({error[:200]}). "
        f"Below is the raw, unfiltered digest of all {len(alerts)} collected alerts so nothing is missed:",
        "",
    ]
    for alert in alerts[:40]:
        line = f"• [{alert.source.value}/{alert.project}] {alert.title[:120]}"
        if alert.url:
            line += f" — {alert.url}"
        lines.append(line)
    if len(alerts) > 40:
        lines.append(f"… and {len(alerts) - 40} more (every one is in the BigQuery alerts table).")
    _post(webhook_url, {"text": "\n".join(lines)})
    logger.info("posted fallback digest to Google Chat")


def post_error(webhook_url: str, run_date: str, stage: str, error: str) -> None:
    """Hard-failure path (e.g. Sentry unreachable): post an error so the team
    knows the run did not complete. The endpoint also returns non-200 so Cloud
    Scheduler retries."""
    if not webhook_url:
        logger.warning("CHAT_WEBHOOK_URL not set; cannot post error notice")
        return
    _post(
        webhook_url,
        {
            "text": (
                f"🚨 *Alert triage failed for {run_date}* at stage '{stage}'. "
                f"{error[:300]} "
                "Cloud Scheduler will retry; today's alerts may not be triaged yet."
            )
        },
    )
    logger.info("posted error notice to Google Chat")
