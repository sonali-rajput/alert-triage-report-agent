"""Flatten a Sentry event into the text an agent reads: stack trace, then
breadcrumbs.

Pure functions, no I/O. This used to live in `pipeline/enrichment.py` alongside
the logic that decided *which* alerts were worth enriching. That decision is
gone -- every issue is now fetched with its full payload in one ingestion pass,
because the agent that picks the day's top issues should not have to make that
pick from thinner evidence than it uses for everything else. What is left is
the formatting, which is what this module is.

Event payloads vary a lot by platform, so every level here is optional.
"""

from __future__ import annotations

from typing import Any

MAX_FRAMES = 12
MAX_BREADCRUMBS = 10


def _entry(event: dict, entry_type: str) -> dict | None:
    for item in event.get("entries") or []:
        if isinstance(item, dict) and item.get("type") == entry_type:
            data = item.get("data")
            return data if isinstance(data, dict) else None
    return None


def _format_frame(frame: dict) -> str:
    location = frame.get("filename") or frame.get("module") or "<unknown>"
    function = frame.get("function") or "<unknown>"
    line_no = frame.get("lineNo")
    marker = "*" if frame.get("inApp") else " "
    where = f"{location}:{line_no}" if line_no else location
    return f"  {marker} {where} in {function}"


def stack_trace_lines(event: dict) -> list[str]:
    exception = _entry(event, "exception")
    if not exception:
        return []

    lines: list[str] = []
    for value in exception.get("values") or []:
        if not isinstance(value, dict):
            continue
        header = ": ".join(p for p in (value.get("type"), value.get("value")) if p)
        if header:
            lines.append(f"Exception: {header}")

        frames = ((value.get("stacktrace") or {}).get("frames")) or []
        # Sentry orders frames oldest-first; the innermost frames are where the
        # error actually happened, so keep the tail rather than the head.
        for frame in frames[-MAX_FRAMES:]:
            if isinstance(frame, dict):
                lines.append(_format_frame(frame))
        if len(frames) > MAX_FRAMES:
            lines.append(f"  ... {len(frames) - MAX_FRAMES} outer frames omitted")
    return lines


def breadcrumb_lines(event: dict) -> list[str]:
    breadcrumbs = _entry(event, "breadcrumbs")
    if not breadcrumbs:
        return []

    values = [v for v in (breadcrumbs.get("values") or []) if isinstance(v, dict)]
    if not values:
        return []

    lines = ["Breadcrumbs (most recent last):"]
    for crumb in values[-MAX_BREADCRUMBS:]:
        category = crumb.get("category") or crumb.get("type") or "log"
        level = crumb.get("level") or "info"
        message = (crumb.get("message") or "").strip()
        if not message:
            data = crumb.get("data")
            message = str(data) if data else ""
        lines.append(f"  [{level}] {category}: {message}"[:300])
    return lines


def event_to_body_text(event: dict[str, Any]) -> str:
    """The text an event contributes to an alert body."""
    if not isinstance(event, dict):
        return ""
    sections = [stack_trace_lines(event), breadcrumb_lines(event)]
    return "\n".join("\n".join(s) for s in sections if s)
