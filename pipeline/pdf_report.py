"""Daily PDF report rendered from an HTML template via WeasyPrint.

The section data is assembled here, in Python, and the template stays purely
presentational -- Jinja is a bad place to compute anything, and every number
below is testable this way.

WeasyPrint needs native libraries (Pango/Cairo) that are present in the Linux
Docker image but usually absent on Windows dev machines, so it is imported
lazily and the renderer degrades to emitting the HTML itself.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

from shared.models import RunStats, TriagedAlert, TriageDecision

logger = logging.getLogger(__name__)

_PRIORITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}
TOP_N = 10
SPARKLINE_HEIGHT = 18
SPARKLINE_BAR_WIDTH = 3
# substatuses worth calling out as "what changed since yesterday"
CHANGED_SUBSTATUSES = ("new", "regressed", "escalating")
# An alert seen this many days running without ever being actioned is a
# standing offender. If that section grows, the triage loop is not working.
OFFENDER_DAYS = 3

# autoescape must be unconditional: select_autoescape matches on the final
# filename suffix, and "report.html.j2" ends in ".j2", so ["html"] silently
# left escaping OFF for every render. Alert titles and LLM output are
# attacker-influenced text and must never land in the HTML raw.
_env = Environment(
    loader=FileSystemLoader(Path(__file__).resolve().parent / "templates"),
    autoescape=True,
)


# ---------------------------------------------------------------------------
# Section data
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Delta:
    """A metric compared against yesterday and the trailing average.

    `previous` and `average` are None when there is no history yet -- a first
    run must render cleanly rather than showing a fabricated 0% change.
    """

    current: int
    previous: int | None = None
    average: float | None = None

    @property
    def change(self) -> int | None:
        return None if self.previous is None else self.current - self.previous

    @property
    def arrow(self) -> str:
        change = self.change
        if change is None or change == 0:
            return "="
        return "▲" if change > 0 else "▼"


@dataclass
class ReportSections:
    scorecard: dict[str, Delta] = field(default_factory=dict)
    top_alerts: list[TriagedAlert] = field(default_factory=list)
    security: list[TriagedAlert] = field(default_factory=list)
    credential_leaks: list[TriagedAlert] = field(default_factory=list)
    # Distinct projects among credential_leaks. The headline is about how many
    # *applications* are misbehaving, which is not the same as how many alerts
    # fired -- one app can leak in five different errors.
    leaking_projects: list[str] = field(default_factory=list)
    by_environment: dict[str, list[TriagedAlert]] = field(default_factory=dict)
    changed: dict[str, list[TriagedAlert]] = field(default_factory=dict)
    offenders: list[TriagedAlert] = field(default_factory=list)
    disagreements: list[TriagedAlert] = field(default_factory=list)
    appendix: list[TriagedAlert] = field(default_factory=list)


def _by_priority(results: Sequence[TriagedAlert]) -> list[TriagedAlert]:
    return sorted(results, key=lambda r: _PRIORITY_ORDER.get(r.triage.priority.value, 9))


def build_scorecard(stats: RunStats, history: Sequence[dict[str, Any]]) -> dict[str, Delta]:
    """Headline counters, each against yesterday and the trailing average.

    `history` is newest-first, as returned by StateStore.recent_runs.
    """

    def past_values(path: tuple[str, ...]) -> list[int]:
        values: list[int] = []
        for run in history:
            node: Any = run.get("stats") or {}
            for key in path:
                node = node.get(key) if isinstance(node, dict) else None
            if isinstance(node, (int, float)):
                values.append(int(node))
        return values

    def delta(path: tuple[str, ...], current: int) -> Delta:
        past = past_values(path)
        return Delta(
            current=current,
            previous=past[0] if past else None,
            average=(sum(past) / len(past)) if past else None,
        )

    return {
        "ingested": delta(("ingested",), stats.ingested),
        "triaged": delta(("processed",), stats.processed),
        "critical": delta(("by_priority", "critical"), stats.by_priority.get("critical", 0)),
        "high": delta(("by_priority", "high"), stats.by_priority.get("high", 0)),
    }


def build_sections(
    stats: RunStats,
    results: Sequence[TriagedAlert],
    history: Sequence[dict[str, Any]] = (),
) -> ReportSections:
    """Everything the template renders, computed once and testable in isolation."""
    notify = [r for r in results if r.triage.decision == TriageDecision.notify]
    ignored = [r for r in results if r.triage.decision == TriageDecision.ignore]

    # Section 2: the top-issues agent's ranking, NOT the triage agent's
    # priority. These are two separate judgements -- "how much does this
    # deserve attention today" and "how urgent is it" -- and the first is the
    # one that decides reading order. rank 0 means unranked (the fail-safe
    # path) and sorts last rather than first.
    top = sorted(results, key=lambda r: (r.rank if r.rank > 0 else 10**6, -r.events_24h))[:TOP_N]

    # Section 3: two distinct populations. A masking hit is a finding about the
    # *application* -- it is logging secrets into its own error messages --
    # rather than about the error, so it is listed separately. It is the finding
    # most likely to be actionable and most likely to be missed.
    security = [r for r in results if r.security_relevant]
    leaks = [r for r in results if r.masking_hits]

    by_env: dict[str, list[TriagedAlert]] = {}
    for r in results:
        by_env.setdefault(r.environment or "unknown", []).append(r)
    # production first, then alphabetical
    by_env = {
        env: _by_priority(rs)
        for env, rs in sorted(by_env.items(), key=lambda kv: (kv[0] != "production", kv[0]))
    }

    changed = {
        status: _by_priority([r for r in results if r.substatus == status])
        for status in CHANGED_SUBSTATUSES
    }

    offenders = sorted(
        (r for r in results if r.ongoing_days >= OFFENDER_DAYS),
        key=lambda r: r.ongoing_days,
        reverse=True,
    )

    return ReportSections(
        scorecard=build_scorecard(stats, history),
        top_alerts=top,
        security=_by_priority(security),
        credential_leaks=leaks,
        leaking_projects=sorted({r.project for r in leaks if r.project}),
        by_environment=by_env,
        changed=changed,
        offenders=offenders,
        disagreements=sorted((r for r in results if r.disagreement), key=lambda r: r.rank),
        appendix=_by_priority(notify) + _by_priority(ignored),
    )


def sparkline_bars(hourly_counts: Sequence[tuple[int, int]]) -> list[dict[str, int]]:
    """Bar heights in px for a 24h sparkline, scaled to the busiest bucket.

    Returned as data rather than markup so the template owns presentation and
    this stays testable. An all-zero day yields flat 1px bars rather than
    dividing by zero.
    """
    counts = [c for _ts, c in hourly_counts]
    if not counts:
        return []
    peak = max(counts)
    return [
        {
            "height": max(1, round(c / peak * SPARKLINE_HEIGHT)) if peak else 1,
            "count": c,
            "width": SPARKLINE_BAR_WIDTH,
        }
        for c in counts
    ]


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_report_html(
    run_id: str,
    run_date: str,
    stats: RunStats,
    results: list[TriagedAlert],
    degraded: bool = False,
    history: Sequence[dict[str, Any]] = (),
) -> str:
    sections = build_sections(stats, results, history)
    template = _env.get_template("report.html.j2")
    return template.render(
        run_id=run_id,
        run_date=run_date,
        stats=stats,
        noise_ratio=stats.noise_ratio,
        degraded=degraded,
        sections=sections,
        priorities=["critical", "high", "medium", "low"],
        sparkline=sparkline_bars,
        sparkline_height=SPARKLINE_HEIGHT,
        offender_days=OFFENDER_DAYS,
        has_history=bool(history),
    )


def render_report(
    run_id: str,
    run_date: str,
    stats: RunStats,
    results: list[TriagedAlert],
    degraded: bool = False,
    history: Sequence[dict[str, Any]] = (),
) -> tuple[bytes, str]:
    """Returns (document_bytes, extension) -- 'pdf' normally, 'html' when
    WeasyPrint's native dependencies are unavailable (local dev)."""
    html = render_report_html(run_id, run_date, stats, results, degraded, history)
    try:
        from weasyprint import HTML

        return HTML(string=html).write_pdf(), "pdf"
    except Exception as exc:
        logger.warning("WeasyPrint unavailable (%s); emitting HTML report instead", exc)
        return html.encode("utf-8"), "html"
