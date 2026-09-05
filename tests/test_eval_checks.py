"""Tests for the label-free evaluation harness.

A check that cannot fail is worse than no check, so these run the suites
against two deliberate provider stand-ins: one that follows the rules in
`config/priority_matrix.yaml` and must pass, and one that violates a specific
rule and must be caught. The offline MockProvider is used as a third case
because it genuinely fails several invariants -- it is a keyword heuristic that
was never given the rules -- which is the cheapest available proof that the
invariants are not vacuous.
"""

from __future__ import annotations

import pytest

from eval import groundedness, invariants, stability
from eval.harness import Report, make_alert
from pipeline.agents.providers import MockProvider, extract_input_json
from shared.models import Priority, SelectedIssue, TriageDecision, TriageOutput


class RuleFollower:
    """A stand-in for a model that actually applies the selection rules.

    Crude on purpose -- it is not a good triager, it is a provider that honours
    the specific relations the invariants assert, so a passing suite means the
    suite can pass at all.
    """

    def _score(self, item: dict) -> float:
        score = 0.0
        score += 3.0 * float(item.get("user_count", 0))                      # RULE 1
        score += 40.0 if "prod" in str(item.get("environment", "")) else 0.0  # RULE 2
        score += 200.0 if item.get("masking_hits") else 0.0                   # RULE 3
        score += 25.0 if item.get("substatus") in ("new", "regressed", "escalating") else 0.0
        score += 25.0 if item.get("is_unhandled") else 0.0                    # RULE 5
        # 24h volume counts; the lifetime total deliberately does not.
        score += 0.05 * float(sum(item.get("hourly_events") or []))
        return score

    def _is_duplicate(self, item: dict) -> bool:
        nearest = (item.get("similar_past") or [{}])[0]
        if not nearest.get("alert_id") or float(nearest.get("distance", 1.0)) > 0.1:
            return False
        # ...unless it has clearly escalated, which is the whole point of
        # dedup_guidance.
        if item.get("substatus") in ("regressed", "escalating"):
            return False
        return int(item.get("user_count", 0)) < 100

    def generate_list(self, system, prompt, item_schema):
        items = extract_input_json(prompt)
        if item_schema is SelectedIssue:
            ranked = sorted(items, key=self._score, reverse=True)
            rank_by_id = {i["alert_id"]: n for n, i in enumerate(ranked, start=1)}
            return [
                SelectedIssue(
                    alert_id=i["alert_id"],
                    is_duplicate=self._is_duplicate(i),
                    duplicate_of=(i.get("similar_past") or [{}])[0].get("alert_id", ""),
                    reason=f"score {self._score(i):.0f}",
                    selected=not self._is_duplicate(i),
                    rank=rank_by_id[i["alert_id"]],
                )
                for i in items
            ]
        if item_schema is TriageOutput:
            out = []
            for i in items:
                users = int(i.get("user_count", 0))
                priority = (
                    Priority.critical if users >= 500 else
                    Priority.high if users >= 100 else
                    Priority.medium if users >= 5 else Priority.low
                )
                out.append(TriageOutput(
                    alert_id=i["alert_id"],
                    summary=f"{i.get('title', '')} affecting {users} users.",
                    component=i.get("project", "unknown"),
                    suspected_cause="unknown",
                    security_relevant=bool(i.get("masking_hits")),
                    security_rationale="masking hit" if i.get("masking_hits") else "none",
                    priority=priority,
                    decision=TriageDecision.notify,
                    reasoning=f"{users} users affected.",
                    clean_title=str(i.get("title", ""))[:80],
                ))
            return out
        raise AssertionError(f"unexpected schema {item_schema}")


# --------------------------------------------------------------------------
# Invariants
# --------------------------------------------------------------------------


def test_a_rule_following_provider_passes_every_invariant():
    report = invariants.run(RuleFollower())
    assert report.failures == [], [f"{o.name}: {o.detail}" for o in report.failures]


def test_the_invariants_are_not_vacuous():
    """The offline mock ranks on users and events only and dedups on a bare
    distance threshold. If the suite passed it, the suite would be measuring
    nothing."""
    report = invariants.run(MockProvider())
    failed = {o.name for o in report.failures}
    assert "production outranks development" in failed
    assert "an escalated repeat is NOT deduplicated" in failed


def test_a_provider_that_suppresses_an_escalation_is_caught():
    """The expensive mistake, and the one with no feedback signal in
    production: a duplicate call hides an alert nobody then sees."""

    class SuppressesEverything(RuleFollower):
        def _is_duplicate(self, item):
            return bool(item.get("similar_past"))

    outcome = invariants.escalated_repeat_is_not_deduplicated(SuppressesEverything())
    assert not outcome.passed


def test_a_check_that_raises_is_an_error_not_a_failure():
    """"The model broke RULE 2" and "we never got to ask" are different facts
    and lead to different actions. Reporting a rate limit or a dead key as a
    FAIL is the kind of noise that gets a suite ignored — which was not
    hypothetical: the first real-model run reported four rate limits as model
    failures."""
    class Exploding:
        def generate_list(self, *a, **kw):
            raise RuntimeError("model unreachable")

    report = invariants.run(Exploding())
    assert report.failures == [], "an unreachable model was blamed for breaking a rule"
    assert len(report.errors) == len(invariants.ALL)
    assert "could not run" in report.errors[0].detail
    assert "model unreachable" in report.errors[0].detail


def test_a_rate_limit_says_how_to_fix_itself():
    class RateLimited:
        def generate_list(self, *a, **kw):
            raise RuntimeError("429 RESOURCE_EXHAUSTED quota exceeded")

    report = invariants.run(RateLimited())
    assert "--pace" in report.errors[0].detail


def test_errors_do_not_count_towards_the_pass_ratio():
    class Exploding:
        def generate_list(self, *a, **kw):
            raise RuntimeError("nope")

    rendered = invariants.run(Exploding()).render()
    assert "could not run" in rendered
    assert "0/0 passed" in rendered or "nothing evaluated" in rendered


# --------------------------------------------------------------------------
# Groundedness
# --------------------------------------------------------------------------


def _output(alert, **kwargs) -> TriageOutput:
    base = dict(
        alert_id=alert.fingerprint(), summary="s", component=alert.project,
        suspected_cause="unknown", security_relevant=False, security_rationale="none",
        priority=Priority.medium, decision=TriageDecision.notify, reasoning="r",
        clean_title="t",
    )
    base.update(kwargs)
    return TriageOutput(**base)


def test_figures_read_off_the_payload_are_supported():
    alert = make_alert("a", "Publish failed", user_count=47, event_count=1234)
    output = _output(alert, summary="Affecting 47 users.", reasoning="1234 events all time.")
    assert groundedness.unsupported_figures(alert, output) == []


def test_an_invented_figure_is_flagged():
    alert = make_alert("a", "Publish failed", user_count=4)
    output = _output(alert, summary="Affecting 4700 users across the studio.")
    assert "4700" in groundedness.unsupported_figures(alert, output)


def test_a_summed_24h_total_counts_as_read_not_invented():
    """A model that adds up the hourly buckets is doing arithmetic, not making
    something up."""
    alert = make_alert("a", "Publish failed", hourly_counts=[(1, 30), (2, 12)])
    output = _output(alert, reasoning="42 events in the last 24 hours.")
    assert groundedness.unsupported_figures(alert, output) == []


def test_small_rhetorical_numbers_are_not_flagged():
    """Flagging 'one of the 3 services' would bury the real findings."""
    alert = make_alert("a", "Publish failed")
    output = _output(alert, summary="One of 2 handlers failed; 3 retries.")
    assert groundedness.unsupported_figures(alert, output) == []


def test_a_component_that_appears_nowhere_is_flagged():
    alert = make_alert("a", "Publish failed", project="asset-service")
    assert groundedness.component_is_grounded(alert, _output(alert, component="asset-service"))
    assert groundedness.component_is_grounded(alert, _output(alert, component="unknown"))
    assert not groundedness.component_is_grounded(alert, _output(alert, component="billing-gateway"))


def test_groundedness_suite_passes_a_grounded_provider():
    alerts = [make_alert(f"a{i}", f"Error {i}", user_count=i) for i in range(3)]
    report = groundedness.run(alerts, RuleFollower())
    assert report.failures == []


# --------------------------------------------------------------------------
# Stability
# --------------------------------------------------------------------------


def test_a_deterministic_provider_is_stable():
    alerts = [make_alert(f"a{i}", f"Error {i}", user_count=i * 10) for i in range(4)]
    report = stability.run(alerts, RuleFollower(), runs=3)
    assert report.failures == []


def test_a_provider_that_changes_its_mind_is_caught():
    """The number this suite exists to produce: a ceiling on how much any
    accuracy measurement can be trusted."""

    class Unstable(RuleFollower):
        def __init__(self):
            self.calls = 0

        def generate_list(self, system, prompt, item_schema):
            outputs = super().generate_list(system, prompt, item_schema)
            if item_schema is TriageOutput:
                self.calls += 1
                if self.calls % 2 == 0:
                    return [o.model_copy(update={"priority": Priority.critical}) for o in outputs]
            return outputs

    alerts = [make_alert(f"a{i}", f"Error {i}", user_count=1) for i in range(4)]
    report = stability.run(alerts, Unstable(), runs=2)
    assert any("priority" in o.name for o in report.failures)


def test_report_renders_failures_with_their_rationale():
    report = Report("t")
    report.add(invariants.Outcome("x", False, "detail", "why it matters"))
    rendered = report.render()
    assert "[FAIL] x" in rendered and "why it matters" in rendered


# --------------------------------------------------------------------------
# Component grounding
#
# Calibrated against a real model on real data: it answered `frontend/axios`,
# `Salesforce Integration` and `TinyMCE Editor` for alerts whose payloads
# contain axios, Salesforce and tinymce respectively. An earlier version
# demanded every token appear and failed all three — punishing the model for
# adding the descriptor that makes a component name readable.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "title,component",
    [
        ("Error: Request failed with status code 403 in axios", "frontend/axios"),
        ("SalesforceMalformedRequest: Malformed request", "Salesforce Integration"),
        ("Error in tinymce: Attribute value was not simple", "TinyMCE Editor"),
        ("OperationalError on asset-service", "asset-service"),
    ],
)
def test_a_descriptor_added_to_a_real_anchor_is_grounded(title, component):
    alert = make_alert("a", title)
    assert groundedness.component_is_grounded(alert, _output(alert, component=component))


def test_a_name_built_only_from_descriptors_is_not_grounded():
    """The case this check exists for: `play() failed` is the HTML5 autoplay
    error, and "Frontend Media Player" is a reasonable inference — but not one
    of those three words appears in the alert. Confident naming of a thing that
    is not there sends someone to the wrong team."""
    alert = make_alert("a", "Error: NotAllowedError: play() failed because the user didn't interact")
    assert not groundedness.component_is_grounded(
        alert, _output(alert, component="Frontend Media Player"))


def test_an_invented_service_name_is_still_caught():
    alert = make_alert("a", "TypeError: cannot read property id", project="asset-service")
    assert not groundedness.component_is_grounded(
        alert, _output(alert, component="billing-gateway"))


def test_unknown_is_an_honest_answer():
    alert = make_alert("a", "Something broke")
    for answer in ("unknown", "unclear", "n/a", "none", ""):
        assert groundedness.component_is_grounded(alert, _output(alert, component=answer))
