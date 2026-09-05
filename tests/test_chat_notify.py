"""The Google Chat card.

The card is a front door, not the report: the tests here are about what a
reader sees first and how much of it there is.
"""

from __future__ import annotations

import json

import pytest

import pipeline.chat_notify as cn
from pipeline.chat_notify import (
    COLLAPSED_ALERTS,
    VISIBLE_ALERTS,
    _impact_line,
    post_error,
    post_fallback_digest,
    post_run_summary,
)
from shared.models import (
    Alert,
    AlertSource,
    AlertSummary,
    Priority,
    RunStats,
    TriagedAlert,
    TriageDecision,
    TriageResult,
)

WEBHOOK = "https://chat.googleapis.com/v1/spaces/x/messages?key=k"


@pytest.fixture
def posted(monkeypatch):
    """Captures the payload instead of posting it."""
    box: list[dict] = []
    monkeypatch.setattr(cn, "_post", lambda url, payload: box.append(payload))
    return box


def ta(title="err", priority="medium", decision="notify", **kwargs) -> TriagedAlert:
    base = dict(
        alert_id=title, source=AlertSource.sentry, title=title, project="core-tools",
        url="https://framestore.sentry.io/issues/1/",
        summary=AlertSummary(alert_id=title, title=title, summary="s", component="c", suspected_cause="u"),
        triage=TriageResult(
            alert_id=title, priority=Priority(priority), decision=TriageDecision(decision),
            reasoning="because",
        ),
    )
    base.update(kwargs)
    return TriagedAlert(**base)


def card_of(payload: dict) -> dict:
    return payload["cardsV2"][0]["card"]


def text_of(payload: dict) -> str:
    # ensure_ascii=False so the separators the card uses (·, →, ▲) survive as
    # themselves rather than \u escapes.
    return json.dumps(payload, ensure_ascii=False)


# --------------------------------------------------------------------------
# Impact line
# --------------------------------------------------------------------------


def test_impact_line_leads_with_people_affected():
    line = _impact_line(ta(user_count=480, events_24h=530, substatus="new", is_unhandled=True))
    assert line.startswith("480 users")
    assert line == "480 users · 530 events · new · unhandled"


def test_impact_line_is_singular_for_one():
    assert _impact_line(ta(user_count=1, events_24h=1)) == "1 user · 1 event"


def test_impact_line_falls_back_to_event_count():
    """Alerts predating the stats projection still have event_count."""
    assert "7 events" in _impact_line(ta(event_count=7, events_24h=0))


def test_impact_line_survives_an_alert_with_no_numbers():
    assert _impact_line(ta(event_count=0)) == "no impact data"


def test_impact_line_omits_ongoing_on_the_first_day():
    assert "ongoing" not in _impact_line(ta(user_count=2, ongoing_days=1))


# --------------------------------------------------------------------------
# Card structure
# --------------------------------------------------------------------------


def test_no_webhook_configured_is_a_no_op(posted):
    post_run_summary("", "2026-08-02", RunStats(), [ta()])
    assert posted == []


def test_shows_only_the_visible_count_expanded(posted):
    results = [ta(f"e{i}", rank=i + 1) for i in range(10)]
    post_run_summary(WEBHOOK, "2026-08-02", RunStats(), results)
    sections = card_of(posted[0])["sections"]
    alert_section = sections[1]
    assert len(alert_section["widgets"]) == VISIBLE_ALERTS


def test_the_next_few_are_collapsed_into_one_message(posted):
    results = [ta(f"e{i}", rank=i + 1) for i in range(10)]
    post_run_summary(WEBHOOK, "2026-08-02", RunStats(), results)
    collapsed = [s for s in card_of(posted[0])["sections"] if s.get("collapsible")]
    assert len(collapsed) == 1
    assert len(collapsed[0]["widgets"]) == COLLAPSED_ALERTS


def test_the_remainder_is_pointed_at_the_report(posted):
    results = [ta(f"e{i}", rank=i + 1) for i in range(VISIBLE_ALERTS + COLLAPSED_ALERTS + 4)]
    post_run_summary(WEBHOOK, "2026-08-02", RunStats(), results)
    assert "and 4 more in the report" in text_of(posted[0])


def test_no_collapsed_section_when_everything_fits(posted):
    post_run_summary(WEBHOOK, "2026-08-02", RunStats(), [ta("a"), ta("b")])
    assert not [s for s in card_of(posted[0])["sections"] if s.get("collapsible")]


def test_quiet_day_says_so(posted):
    post_run_summary(WEBHOOK, "2026-08-02", RunStats(ingested=40, prefiltered=40), [])
    assert "No alerts require attention today" in text_of(posted[0])


def test_ignored_alerts_are_not_shown(posted):
    post_run_summary(WEBHOOK, "2026-08-02", RunStats(), [ta("noise", decision="ignore")])
    assert "No alerts require attention today" in text_of(posted[0])


# --------------------------------------------------------------------------
# Ordering and content
# --------------------------------------------------------------------------


def test_rows_follow_the_selection_rank(posted):
    """The card and the report's Top 10 must agree on ordering."""
    results = [ta("triaged-critical", "critical", rank=2), ta("ranked-first", "low", rank=1)]
    post_run_summary(WEBHOOK, "2026-08-02", RunStats(), results)
    first = card_of(posted[0])["sections"][1]["widgets"][0]["decoratedText"]
    assert "ranked-first" in first["text"]


def test_an_unranked_alert_sorts_last_rather_than_first(posted):
    """rank 0 is the fail-safe path, not "most important"."""
    results = [ta("unranked", rank=0), ta("ranked", rank=9)]
    post_run_summary(WEBHOOK, "2026-08-02", RunStats(), results)
    first = card_of(posted[0])["sections"][1]["widgets"][0]["decoratedText"]
    assert "ranked" in first["text"] and "unranked" not in first["text"]


def test_a_top_ranked_low_alert_explains_itself(posted):
    """The two agents disagreeing puts a LOW at the top of the card; without
    the marker that looks broken rather than informative."""
    results = [ta("quiet-but-ranked-first", "low", rank=1, disagreement=True)]
    post_run_summary(WEBHOOK, "2026-08-02", RunStats(), results)
    assert "ranked #1 today" in text_of(posted[0])


def test_environment_appears_on_every_row(posted):
    post_run_summary(WEBHOOK, "2026-08-02", RunStats(), [ta(environment="staging")])
    assert "staging" in card_of(posted[0])["sections"][1]["widgets"][0]["decoratedText"]["topLabel"]


def test_delta_line_reports_what_changed(posted):
    results = [ta("a", substatus="new"), ta("b", substatus="new"), ta("c", substatus="regressed")]
    post_run_summary(WEBHOOK, "2026-08-02", RunStats(), results)
    assert "2 new · 1 regressed" in text_of(posted[0])


def test_delta_line_when_nothing_changed(posted):
    post_run_summary(WEBHOOK, "2026-08-02", RunStats(), [ta(substatus="ongoing")])
    assert "No new, regressed or escalating alerts" in text_of(posted[0])


def test_delta_line_counts_only_what_the_card_shows(posted):
    """An ignored alert must not make the card announce '1 new' and then show
    nothing -- the delta describes the rows, and the rows are notify-only."""
    post_run_summary(WEBHOOK, "2026-08-02", RunStats(), [ta("noise", decision="ignore", substatus="new")])
    assert "No new, regressed or escalating alerts" in text_of(posted[0])


def test_titles_are_escaped(posted):
    """Chat renders an HTML subset in decoratedText; a '<' in an exception
    title (common in Python reprs) breaks the widget."""
    result = ta("Error in <module> at <stdin>")
    post_run_summary(WEBHOOK, "2026-08-02", RunStats(), [result])
    text = card_of(posted[0])["sections"][1]["widgets"][0]["decoratedText"]["text"]
    assert "<module>" not in text
    assert "&lt;module&gt;" in text


def test_priority_counts_and_funnel_are_in_the_header(posted):
    stats = RunStats(ingested=312, prefiltered=88, deduped=63, processed=71,
                     by_priority={"critical": 2, "high": 5})
    post_run_summary(WEBHOOK, "2026-08-02", stats, [ta()])
    body = text_of(posted[0])
    assert "2 critical" in body and "5 high" in body
    assert "312 ingested → 88 noise → 63 duplicates → 71 triaged" in body


def test_pdf_button_is_included_when_the_url_is_signed(posted):
    post_run_summary(WEBHOOK, "2026-08-02", RunStats(), [ta()], "https://storage.googleapis.com/x.pdf")
    assert "Full report (PDF)" in text_of(posted[0])


def test_local_file_path_is_not_offered_as_a_button(posted):
    """Local dev writes the report to disk; a file path is not clickable."""
    post_run_summary(WEBHOOK, "2026-08-02", RunStats(), [ta()], "/tmp/artifacts/report.pdf")
    assert "Full report (PDF)" not in text_of(posted[0])


# --------------------------------------------------------------------------
# Fail-loud paths — deliberately unchanged
# --------------------------------------------------------------------------


def test_fallback_digest_lists_every_alert(posted):
    alerts = [Alert(source=AlertSource.sentry, source_id=str(i), title=f"err {i}") for i in range(3)]
    post_fallback_digest(WEBHOOK, "2026-08-02", alerts, "vertex exploded")
    body = posted[0]["text"]
    assert "degraded" in body.lower()
    for i in range(3):
        assert f"err {i}" in body


def test_fallback_digest_truncates_but_says_so(posted):
    alerts = [Alert(source=AlertSource.sentry, source_id=str(i), title=f"err {i}") for i in range(50)]
    post_fallback_digest(WEBHOOK, "2026-08-02", alerts, "boom")
    assert "and 10 more" in posted[0]["text"]


def test_error_notice_names_the_stage(posted):
    post_error(WEBHOOK, "2026-08-02", "sentry-fetch", "connection refused")
    assert "sentry-fetch" in posted[0]["text"]
    assert "connection refused" in posted[0]["text"]


# --------------------------------------------------------------------------
# The quiet day
# --------------------------------------------------------------------------


def test_a_day_with_nothing_to_report_still_posts_a_card(posted):
    """Not an exotic state: the day every alert is judged a repeat produces
    zero results, and that is the pipeline working. Silence would read as
    'the run did not happen', which is the one message it must never send by
    accident."""
    stats = RunStats(ingested=19, prefiltered=0, deduped=19, considered=19, processed=0)
    post_run_summary(WEBHOOK, "2026-08-12", stats, [], "")

    assert len(posted) == 1
    text = text_of(posted[0])
    assert "No alerts require attention today" in text
    # The funnel still has to show the work: 19 in, 19 recognised as repeats.
    assert "19" in text


def test_the_quiet_day_card_still_links_the_report(posted):
    stats = RunStats(ingested=19, deduped=19, considered=19, processed=0)
    post_run_summary(WEBHOOK, "2026-08-12", stats, [], "https://signed.example/report.pdf")
    assert "https://signed.example/report.pdf" in text_of(posted[0])


def test_a_local_report_path_is_not_offered_as_a_button(posted):
    """Offline the report is a filesystem path. A button pointing at one is a
    dead button in everyone else's Chat client."""
    post_run_summary(WEBHOOK, "2026-08-12", RunStats(), [ta("e")], "/home/me/artifacts/r.pdf")
    assert "buttonList" not in text_of(posted[0])
