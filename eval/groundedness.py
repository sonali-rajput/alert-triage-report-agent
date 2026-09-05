"""Did the model make anything up?

A deterministic hallucination check. No labels, no second model, no judgement
call: every figure the agent cites in its prose has to appear somewhere in the
payload it was given, and the component it names has to be a string that
actually occurs in the alert.

Why this is worth having when the priority might be right anyway: the report's
value is that an engineer can act on it without opening Sentry. A summary
saying "affecting 47 users" when the payload says 4 is worse than a wrong
priority -- a wrong priority is a judgement someone can second-guess, an
invented number is a fact they will act on.

Deliberately narrow. It does not check whether the *reasoning* is sound, only
whether it is anchored in the input. Soundness is what the golden dataset and a
human reading disagreements are for.
"""

from __future__ import annotations

import re

from eval.harness import Outcome, Report
from pipeline.agents.providers import LLMProvider
from pipeline.agents.top_issues import _payload as selection_payload
from pipeline.agents.triage import _payload as triage_payload
from pipeline.agents.triage import triage_alerts
from shared.models import Alert, TriageOutput

_NUMBER = re.compile(r"\b\d[\d,]*\b")
# Figures a model uses to write English rather than to state a fact about the
# alert: "one of the", "2-3 sentences", "the top 10". Flagging these would bury
# the real findings in noise.
_RHETORICAL = {"0", "1", "2", "3", "10"}
# Time windows are quoted from the question, not claimed about the alert.
# "223 events in the last 24 hours" asserts one fact and one framing, and
# flagging the 24 in every single summary would drown the real findings.
_TIME_WINDOW = re.compile(
    r"\b\d+\s*(?:-|\s)?\s*(?:h|hr|hrs|hour|hours|d|day|days|m|min|mins|minute|minutes|w|week|weeks|month|months)\b",
    re.IGNORECASE,
)


def _numbers_in(text: str) -> set[str]:
    return {m.group(0).replace(",", "") for m in _NUMBER.finditer(text)}


def _claimed_numbers(prose: str) -> set[str]:
    """Numbers the model is asserting about the alert, with time windows
    removed first."""
    return _numbers_in(_TIME_WINDOW.sub(" ", prose))


def _supported_numbers(alert: Alert) -> set[str]:
    """Every figure the model could legitimately have read off its input: the
    payload's numeric fields, and any number appearing in the text it was given
    (a stack frame line number, an HTTP status in the title)."""
    payload = {**selection_payload(alert), **triage_payload(alert)}
    supported: set[str] = set()
    for key, value in payload.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            supported.add(str(value))
        elif isinstance(value, list):
            supported |= _numbers_in(" ".join(str(v) for v in value))
        elif isinstance(value, str):
            supported |= _numbers_in(value)
    # Sums and simple derivations are fair game: a model that adds up the
    # hourly buckets and reports the total is doing arithmetic, not inventing.
    supported.add(str(alert.events_24h()))
    supported.add(str(sum(c for _ts, c in alert.hourly_counts)))
    return supported


def unsupported_figures(alert: Alert, output: TriageOutput) -> list[str]:
    """Figures in the agent's prose that appear nowhere in its input."""
    prose = f"{output.summary} {output.reasoning} {output.suspected_cause} {output.security_rationale}"
    supported = _supported_numbers(alert)
    return sorted(n for n in _claimed_numbers(prose) - supported if n not in _RHETORICAL)


# Words that describe a KIND of thing rather than identify one. A component
# named entirely from these is a description, not an identification -- "Frontend
# Media Player" could be any of a hundred services, and it is the confident
# naming of a thing that does not exist that sends someone to the wrong team.
_GENERIC_COMPONENT_WORDS = {
    "frontend", "backend", "front", "back", "service", "services", "server",
    "client", "api", "integration", "editor", "player", "media", "module",
    "component", "app", "application", "handler", "worker", "job", "web",
    "page", "system", "layer", "library", "framework", "endpoint", "gateway",
}


def component_is_grounded(alert: Alert, output: TriageOutput) -> bool:
    """The named component must be ANCHORED in the alert by at least one
    distinctive word.

    An earlier version required every meaningful token to appear, and the real
    model failed it four times out of eight -- correctly, three of those times.
    It answered `frontend/axios` for an axios request failure, `Salesforce
    Integration` for a `SalesforceMalformedRequest`, and `TinyMCE Editor` for a
    tinymce error. In each case the identifying word IS in the payload and the
    model added a descriptor, which is what a readable component name looks
    like. Demanding literal containment punished it for being helpful.

    The fourth was different: `Frontend Media Player` for a `play() failed`
    autoplay error, where not one of those three words appears anywhere in the
    alert. That is inference presented as identification, and it is the case
    this check exists for.

    So: at least one non-generic token must appear. 'unknown' is an honest
    answer and passes.
    """
    component = (output.component or "").strip().lower()
    if not component or component in {"unknown", "unclear", "n/a", "none"}:
        return True
    haystack = f"{alert.title} {alert.body} {alert.project} {alert.url}".lower()
    if component in haystack:
        return True
    tokens = [t for t in re.split(r"[\s\-_/.]+", component) if len(t) > 2]
    distinctive = [t for t in tokens if t not in _GENERIC_COMPONENT_WORDS]
    if not distinctive:
        # Nothing but descriptors -- it can only be grounded by appearing whole.
        return False
    return any(t in haystack for t in distinctive)


def run(alerts: list[Alert], provider: LLMProvider) -> Report:
    report = Report("GROUNDEDNESS (no labels needed)")
    if not alerts:
        report.add(Outcome("groundedness", True, "no alerts to check", skipped=True))
        return report

    outputs = triage_alerts(alerts, provider)
    by_id = {o.alert_id: o for o in outputs}

    invented: list[str] = []
    ungrounded_components: list[str] = []
    for alert in alerts:
        output = by_id.get(alert.fingerprint())
        if output is None:
            continue
        figures = unsupported_figures(alert, output)
        if figures:
            invented.append(f"{alert.short_id or alert.source_id}: {', '.join(figures)}")
        if not component_is_grounded(alert, output):
            ungrounded_components.append(f"{alert.short_id or alert.source_id}: '{output.component}'")

    report.add(Outcome(
        "no invented figures in the agent's prose",
        not invented,
        f"{len(invented)}/{len(alerts)} alerts cite a figure absent from their payload"
        + (f" — {'; '.join(invented[:5])}" if invented else ""),
        "An invented number is worse than a wrong priority: a priority is a "
        "judgement someone can second-guess, a number is a fact they will act on.",
    ))
    report.add(Outcome(
        "named components exist in the alert",
        not ungrounded_components,
        f"{len(ungrounded_components)}/{len(alerts)} name a component that appears nowhere in the payload"
        + (f" — {'; '.join(ungrounded_components[:5])}" if ungrounded_components else ""),
        "A confidently-named service that does not exist sends someone to the "
        "wrong team.",
    ))
    return report
