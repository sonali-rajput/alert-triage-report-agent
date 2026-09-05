"""Shared plumbing for the label-free evaluation checks.

The golden dataset answers "is the model right?", which is the expensive
question: it needs a human to say what right was. These checks answer cheaper
questions that catch most of the same failures:

  * invariants  -- "is the model *consistent* with the rules we gave it?"
  * groundedness -- "is the model making things up?"
  * stability   -- "does it say the same thing twice?"

None of them need a label. All of them run against any provider.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from shared.models import Alert, AlertSource


@dataclass
class Outcome:
    """One check's result. `detail` is written to be read by a human who is
    deciding whether to care, so it states what happened, not just pass/fail."""

    name: str
    passed: bool
    detail: str
    why: str = ""  # what this check protects against
    skipped: bool = False
    # The check never reached a verdict -- a rate limit, a dead key, a timeout.
    # Reporting that as FAIL is the kind of noise that gets a suite ignored:
    # "the model broke RULE 2" and "we never got to ask" are different facts and
    # lead to different actions.
    errored: bool = False

    @property
    def mark(self) -> str:
        if self.skipped:
            return "SKIP"
        if self.errored:
            return "ERR "
        return "PASS" if self.passed else "FAIL"


@dataclass
class Report:
    title: str
    outcomes: list[Outcome] = field(default_factory=list)

    def add(self, outcome: Outcome) -> Outcome:
        self.outcomes.append(outcome)
        return outcome

    @property
    def failures(self) -> list[Outcome]:
        """Checks the model actually got wrong."""
        return [o for o in self.outcomes if not o.passed and not o.skipped and not o.errored]

    @property
    def errors(self) -> list[Outcome]:
        """Checks that never reached a verdict."""
        return [o for o in self.outcomes if o.errored]

    def render(self) -> str:
        lines = [f"\n=== {self.title} ===", ""]
        for o in self.outcomes:
            lines.append(f"[{o.mark}] {o.name}")
            lines.append(f"       {o.detail}")
            if not o.passed and o.why:
                lines.append(f"       why it matters: {o.why}")
        passed = sum(1 for o in self.outcomes if o.passed and not o.skipped and not o.errored)
        total = sum(1 for o in self.outcomes if not o.skipped and not o.errored)
        lines.append("")
        summary = f"{passed}/{total} passed" if total else "nothing evaluated"
        if self.errors:
            summary += f"  ({len(self.errors)} could not run)"
        lines.append(summary)
        return "\n".join(lines)


def make_alert(source_id: str, title: str, **kwargs) -> Alert:
    """A plausible alert with everything defaulted, so each check sets only the
    one or two fields it is actually varying. That is the whole point of a
    metamorphic test: if two alerts differ in exactly one field, any difference
    in the verdict is attributable to that field."""
    base: dict = dict(
        source=AlertSource.sentry,
        source_id=source_id,
        kind="sentry_issue",
        title=title,
        body="Level: error\nCulprit: app/handler.py in process",
        project="asset-service",
        environment="production",
        level="error",
        substatus="ongoing",
        user_count=10,
        event_count=100,
        hourly_counts=[(i, 4) for i in range(24)],
        sentry_priority="medium",
    )
    base.update(kwargs)
    return Alert(**base)


def rank_of(selected: list[Alert], source_id: str) -> int | None:
    """1-based rank, or None when the alert was not selected at all."""
    for position, alert in enumerate(selected, start=1):
        if alert.source_id == source_id:
            return position
    return None
