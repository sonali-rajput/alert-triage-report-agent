import pytest

from pipeline.agents.providers import INPUT_MARKER, LLMError, MockProvider, extract_input_json
from pipeline.agents.triage import (
    PROMPT_VERSION,
    _payload,
    build_system_prompt,
    summarize_and_triage,
    triage_alerts,
)
from shared.models import Alert, AlertSource, Priority, TriageDecision, TriageOutput


def make_alert(title: str, count: int = 1, source_id: str | None = None, **kwargs) -> Alert:
    base = dict(
        source=AlertSource.sentry, source_id=source_id or title, kind="sentry_issue",
        title=title, body="details", project="asset-service", event_count=count,
    )
    base.update(kwargs)
    return Alert(**base)


def test_extract_input_json():
    prompt = f"Do things.\n{INPUT_MARKER}\n[{{\"a\": 1}}]"
    assert extract_input_json(prompt) == [{"a": 1}]


# --------------------------------------------------------------------------
# The merged call
# --------------------------------------------------------------------------


def test_returns_one_output_per_alert_in_order():
    alerts = [make_alert(f"error {i}") for i in range(5)]
    outputs = triage_alerts(alerts, MockProvider())
    assert [o.alert_id for o in outputs] == [a.fingerprint() for a in alerts]


def test_empty_input_makes_no_call():
    class Exploding:
        def generate_list(self, *a, **kw):
            raise AssertionError("should not be called")

    assert triage_alerts([], Exploding()) == []


def test_one_output_carries_both_halves():
    alerts = [make_alert("Database connection pool exhausted, all users affected", count=800)]
    output = triage_alerts(alerts, MockProvider())[0]
    assert output.summary and output.component          # summarizer half
    assert output.priority == Priority.critical          # triage half
    assert output.decision == TriageDecision.notify
    assert output.reasoning


def test_triage_ignores_deprecation_noise():
    outputs = triage_alerts([make_alert("DeprecationWarning: imp module is deprecated")], MockProvider())
    assert outputs[0].decision == TriageDecision.ignore


def test_omitted_alert_fails_safe_rather_than_vanishing():
    """The fail-safe that must survive the merge: an alert the model drops is
    surfaced at medium/notify, never lost."""

    class DropsEverything:
        def generate_list(self, system, prompt, item_schema):
            return []

    outputs = triage_alerts([make_alert("something")], DropsEverything())
    assert len(outputs) == 1
    assert outputs[0].priority == Priority.medium
    assert outputs[0].decision == TriageDecision.notify
    assert "omitted" in outputs[0].reasoning


def test_the_whole_selection_goes_in_one_call():
    """No batching. The top-issues agent hands this stage ten alerts, so one
    call covers the lot -- and one call lets the model weigh them against each
    other, which a batch boundary silently prevents."""
    calls: list[int] = []

    class Counting(MockProvider):
        def generate_list(self, system, prompt, item_schema):
            calls.append(len(extract_input_json(prompt)))
            return super().generate_list(system, prompt, item_schema)

    alerts = [make_alert(f"error {i}", source_id=str(i)) for i in range(10)]
    outputs = triage_alerts(alerts, Counting())

    assert calls == [10]
    assert len({o.alert_id for o in outputs}) == 10


def test_a_failed_call_is_a_stage_failure():
    """There is no partial success left to salvage: one call means it either
    worked or it did not, and the orchestrator's degraded-digest path (which
    lists ALL alerts raw) is the right response to 'it did not'."""

    class AlwaysFails:
        def generate_list(self, system, prompt, item_schema):
            raise LLMError("vertex is down")

    with pytest.raises(LLMError):
        triage_alerts([make_alert("a"), make_alert("b")], AlwaysFails())


# --------------------------------------------------------------------------
# The payload the model actually sees
# --------------------------------------------------------------------------


def test_payload_includes_the_alert_body():
    """The defect the merge fixes: the old triage stage never forwarded
    alert.body, so it judged priority without seeing the original error."""
    assert _payload(make_alert("x", body="the original stack trace"))["body"] == "the original stack trace"


def test_payload_carries_the_impact_signals():
    alert = make_alert(
        "x", user_count=480, environment="production", substatus="regressed",
        is_unhandled=True, sentry_priority="high", masking_hits=["bearer-token"],
        selection_reason="480 users in production", similar_past=[{"alert_id": "old"}],
    )
    payload = _payload(alert)
    assert payload["user_count"] == 480
    assert payload["environment"] == "production"
    assert payload["substatus"] == "regressed"
    assert payload["is_unhandled"] is True
    assert payload["sentry_priority"] == "high"
    assert payload["masking_hits"] == ["bearer-token"]
    # Why the top-issues agent picked it, and what we decided about it before.
    # Both are context the triage agent is told it may disagree with.
    assert payload["selection_reason"] == "480 users in production"
    assert payload["similar_past"] == [{"alert_id": "old"}]


def test_payload_events_24h_prefers_stats_over_count():
    alert = make_alert("x", count=9999, hourly_counts=[(1, 3), (2, 4)])
    payload = _payload(alert)
    assert payload["events_24h"] == 7
    # Named for the period it covers. On the real org-issues endpoint `count`
    # is the ALL-TIME total -- one real issue reads 206,017 against 223 events
    # in the last 24h -- so a bare `event_count` beside `events_24h` is an
    # invitation for the model to anchor on the wrong number.
    assert payload["event_count_all_time"] == 9999
    assert "event_count" not in payload


# --------------------------------------------------------------------------
# Response schema ordering -- analysis before verdict
# --------------------------------------------------------------------------


def test_analysis_fields_are_generated_before_the_verdict():
    """Generation is autoregressive, so field order in the schema decides what
    the model commits to first. Reordering this silently weakens the reasoning."""
    fields = list(TriageOutput.model_fields)
    for analytical in ("summary", "component", "suspected_cause", "security_relevant"):
        assert fields.index(analytical) < fields.index("priority"), analytical
    assert fields.index("priority") < fields.index("reasoning")


def test_views_round_trip_to_the_old_shapes():
    output = triage_alerts([make_alert("LDAP bind failed")], MockProvider())[0]
    summary, triage = output.to_summary(), output.to_triage_result()
    assert summary.alert_id == triage.alert_id == output.alert_id
    assert summary.summary == output.summary
    assert triage.priority == output.priority


def test_summarize_and_triage_returns_aligned_lists():
    alerts = [make_alert(f"e{i}", source_id=str(i)) for i in range(4)]
    summaries, results = summarize_and_triage(alerts, MockProvider())
    assert len(summaries) == len(results) == 4
    assert [s.alert_id for s in summaries] == [r.alert_id for r in results]


# --------------------------------------------------------------------------
# Security assessment
# --------------------------------------------------------------------------


def test_credential_masking_hit_marks_the_alert_security_relevant():
    outputs = triage_alerts([make_alert("push failed", masking_hits=["bearer-token"])], MockProvider())
    assert outputs[0].security_relevant is True
    assert "bearer-token" in outputs[0].security_rationale


def test_plain_error_is_not_security_relevant():
    outputs = triage_alerts([make_alert("TypeError: undefined is not a function")], MockProvider())
    assert outputs[0].security_relevant is False


# --------------------------------------------------------------------------
# Prompt
# --------------------------------------------------------------------------


def test_system_prompt_contains_the_priority_rules():
    prompt = build_system_prompt()
    for level in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
        assert level in prompt
    assert "audit" in prompt


def test_the_priority_rules_are_prose_not_numbers():
    """The rules live in config/priority_matrix.yaml as sentences a human can
    read and argue with. A threshold creeping back into the prompt is a
    calibration exercise creeping back with it."""
    prompt = build_system_prompt()
    assert "impact_score" not in prompt
    assert "score_bands" not in prompt


def test_the_prompt_asks_for_no_owning_team():
    """Triage says what broke and how bad it is; it does not name who fixes it.
    `routing` was also the only field the response schema could not constrain
    to valid values, so a hallucinated team read as authoritative as a real
    one. The report links to the Sentry issue instead."""
    prompt = build_system_prompt().lower()
    assert "routing" not in prompt
    assert "route to" not in prompt


def test_system_prompt_explains_the_new_signals():
    prompt = build_system_prompt()
    for signal in ("user_count", "events_24h", "selection_reason", "similar_past", "masking_hits"):
        assert signal in prompt


def test_system_prompt_warns_that_titles_are_unreliable():
    """The payload finding that drives the whole design: numbers beat text."""
    assert "RaiseException" in build_system_prompt()


def test_prompt_version_is_set():
    assert PROMPT_VERSION
