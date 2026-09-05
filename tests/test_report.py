"""Report section data.

Everything the template shows is computed in build_sections, so this is where
the report is actually tested — rendering is then just markup.
"""

from __future__ import annotations

import pytest

from pipeline.pdf_report import (
    OFFENDER_DAYS,
    TOP_N,
    build_scorecard,
    build_sections,
    render_report_html,
    sparkline_bars,
)
from shared.models import (
    AlertSource,
    AlertSummary,
    Priority,
    RunStats,
    TriagedAlert,
    TriageDecision,
    TriageResult,
)


def ta(title="err", priority="medium", decision="notify", **kwargs) -> TriagedAlert:
    base = dict(
        alert_id=title, source=AlertSource.sentry, title=title, project="core-tools",
        summary=AlertSummary(alert_id=title, title=title, summary="s", component="c", suspected_cause="u"),
        triage=TriageResult(
            alert_id=title, priority=Priority(priority),
            decision=TriageDecision(decision), reasoning="because",
        ),
    )
    base.update(kwargs)
    return TriagedAlert(**base)


def stats_for(results, **kwargs) -> RunStats:
    base = dict(
        ingested=len(results) * 3, processed=len(results),
        ignored=sum(1 for r in results if r.triage.decision == TriageDecision.ignore),
        by_priority={
            lvl: sum(1 for r in results if r.triage.priority.value == lvl)
            for lvl in ("critical", "high", "medium", "low")
        },
    )
    base.update(kwargs)
    return RunStats(**base)


# --------------------------------------------------------------------------
# Scorecard deltas
# --------------------------------------------------------------------------


def test_delta_reports_no_history_rather_than_a_fake_zero():
    """A first run must not claim '0 vs yesterday' — it has no yesterday."""
    d = build_scorecard(RunStats(ingested=10), [])["ingested"]
    assert d.previous is None and d.change is None and d.average is None
    assert d.arrow == "="


def test_delta_compares_against_the_most_recent_run():
    history = [
        {"stats": {"ingested": 100}},   # newest first
        {"stats": {"ingested": 200}},
    ]
    d = build_scorecard(RunStats(ingested=120), history)["ingested"]
    assert d.previous == 100
    assert d.change == 20
    assert d.arrow == "▲"
    assert d.average == pytest.approx(150.0)


def test_delta_arrow_points_down_when_the_count_falls():
    d = build_scorecard(RunStats(ingested=50), [{"stats": {"ingested": 90}}])["ingested"]
    assert d.change == -40 and d.arrow == "▼"


def test_delta_reads_nested_priority_counts():
    history = [{"stats": {"by_priority": {"critical": 5}}}]
    d = build_scorecard(RunStats(by_priority={"critical": 2}), history)["critical"]
    assert d.previous == 5 and d.change == -3


def test_malformed_history_is_skipped_not_crashed():
    """Run docs come from Firestore; a degraded run has no by_priority at all."""
    history = [{}, {"stats": None}, {"stats": {"ingested": "nonsense"}}, {"stats": {"ingested": 10}}]
    d = build_scorecard(RunStats(ingested=20), history)["ingested"]
    assert d.previous == 10


# --------------------------------------------------------------------------
# Section 2 — top by selection rank
# --------------------------------------------------------------------------


def test_top_follows_the_selection_rank_not_the_triage_priority():
    """The whole point of the section. "What deserves attention today" and
    "how urgent is it" are two separate judgements, and the first one decides
    reading order."""
    results = [
        ta("triaged-critical", "critical", rank=2),
        ta("ranked-first", "low", rank=1),
    ]
    top = build_sections(stats_for(results), results).top_alerts
    assert top[0].title == "ranked-first"


def test_top_is_capped():
    results = [ta(f"e{i}", rank=i + 1) for i in range(TOP_N + 5)]
    assert len(build_sections(stats_for(results), results).top_alerts) == TOP_N


def test_an_unranked_alert_sorts_last_rather_than_first():
    """rank 0 means the fail-safe path, not "most important". Sorting it
    naively would put every un-ranked alert at the top of the report."""
    results = [ta("unranked", rank=0), ta("ranked", rank=5)]
    assert build_sections(stats_for(results), results).top_alerts[0].title == "ranked"


# --------------------------------------------------------------------------
# Section 3 — security
# --------------------------------------------------------------------------


def test_security_and_credential_leaks_are_separate_populations():
    """A masking hit is a finding about the APPLICATION (it logs secrets),
    which is not the same as the model judging the error security-relevant."""
    results = [
        ta("auth-error", security_relevant=True),
        ta("leaky-app", masking_hits=["bearer-token"]),
    ]
    sections = build_sections(stats_for(results), results)
    assert [r.title for r in sections.security] == ["auth-error"]
    assert [r.title for r in sections.credential_leaks] == ["leaky-app"]


def test_leaking_projects_counts_applications_not_alerts():
    """Two alerts from one app is one misbehaving application, not two."""
    results = [
        ta("a", project="publish-hooks", masking_hits=["bearer-token"]),
        ta("b", project="publish-hooks", masking_hits=["aws-access-key"]),
        ta("c", project="deploy-tools", masking_hits=["gitlab-pat"]),
    ]
    sections = build_sections(stats_for(results), results)
    assert len(sections.credential_leaks) == 3
    assert sections.leaking_projects == ["deploy-tools", "publish-hooks"]


# --------------------------------------------------------------------------
# Sections 4-7
# --------------------------------------------------------------------------


def test_production_sorts_first_among_environments():
    results = [ta("a", environment="staging"), ta("b", environment="production"), ta("c", environment="dev")]
    assert list(build_sections(stats_for(results), results).by_environment) == ["production", "dev", "staging"]


def test_alerts_without_an_environment_are_bucketed_as_unknown():
    results = [ta("a", environment="")]
    assert "unknown" in build_sections(stats_for(results), results).by_environment


def test_changed_section_groups_by_substatus():
    results = [
        ta("a", substatus="new"), ta("b", substatus="regressed"),
        ta("c", substatus="escalating"), ta("d", substatus="ongoing"),
    ]
    changed = build_sections(stats_for(results), results).changed
    assert {k: len(v) for k, v in changed.items()} == {"new": 1, "regressed": 1, "escalating": 1}


def test_offenders_need_the_full_day_threshold():
    results = [ta("old", ongoing_days=OFFENDER_DAYS), ta("newish", ongoing_days=OFFENDER_DAYS - 1)]
    offenders = build_sections(stats_for(results), results).offenders
    assert [r.title for r in offenders] == ["old"]


def test_offenders_are_worst_first():
    results = [ta("a", ongoing_days=4), ta("b", ongoing_days=11)]
    assert [r.title for r in build_sections(stats_for(results), results).offenders] == ["b", "a"]


def test_only_flagged_disagreements_appear():
    results = [ta("agrees", rank=2), ta("diverges", "low", disagreement=True, rank=1)]
    disagreements = build_sections(stats_for(results), results).disagreements
    assert [r.title for r in disagreements] == ["diverges"]


# --------------------------------------------------------------------------
# Section 8 — appendix
# --------------------------------------------------------------------------


def test_appendix_holds_everything_with_notified_first():
    results = [ta("noise", "low", "ignore"), ta("real", "critical")]
    appendix = build_sections(stats_for(results), results).appendix
    assert [r.title for r in appendix] == ["real", "noise"]


# --------------------------------------------------------------------------
# Sparklines
# --------------------------------------------------------------------------


def test_sparkline_scales_to_the_busiest_bucket():
    bars = sparkline_bars([(0, 0), (1, 50), (2, 100)])
    assert bars[-1]["height"] > bars[1]["height"] > bars[0]["height"]


def test_sparkline_of_an_all_zero_day_does_not_divide_by_zero():
    bars = sparkline_bars([(i, 0) for i in range(24)])
    assert len(bars) == 24
    assert all(b["height"] == 1 for b in bars)


def test_sparkline_without_data_is_empty():
    assert sparkline_bars([]) == []


def test_sparkline_keeps_every_bucket():
    assert len(sparkline_bars([(i, i) for i in range(24)])) == 24


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def test_renders_all_eight_section_headings():
    results = [ta("a", "critical", environment="production", masking_hits=["bearer-token"],
                  security_relevant=True, disagreement=True, rank=1,
                  ongoing_days=9, substatus="new", hourly_counts=[(0, 1), (1, 5)])]
    html = render_report_html("run-1", "2026-08-02", stats_for(results), results)
    for n in range(1, 9):
        assert f">{n} · " in html or f"{n} · " in html


def test_a_critical_alert_names_the_project_it_came_from():
    """The project has no say in an alert's PRIORITY -- ranking is issue risk
    only -- but it is what tells a reader whether a critical alert is theirs, so
    it has to be on the row. These are two separate concerns and removing the
    scoring term must never take the label with it."""
    results = [ta("a", "critical", project="prodtools-render-submit", user_count=40)]
    html = render_report_html("run-1", "2026-08-02", stats_for(results), results)
    assert "prodtools-render-submit" in html


def test_an_alert_with_no_project_still_renders_a_label():
    results = [ta("a", "critical", project="")]
    html = render_report_html("run-1", "2026-08-02", stats_for(results), results)
    assert "unknown project" in html


def test_renders_with_no_results_at_all():
    """A quiet day must still produce a readable report, not a stack trace."""
    html = render_report_html("run-1", "2026-08-02", RunStats(ingested=0), [])
    assert "Nothing triaged today" in html
    assert "No standing offenders" in html


def test_degraded_banner_is_shown():
    html = render_report_html("run-1", "2026-08-02", RunStats(), [], degraded=True)
    assert "Degraded run" in html


def test_security_section_prefers_the_models_security_rationale():
    results = [ta("a", security_relevant=True, security_rationale="LDAP bind rejected credentials")]
    html = render_report_html("run-1", "2026-08-02", stats_for(results), results)
    assert "LDAP bind rejected credentials" in html


def test_alert_titles_and_llm_output_are_escaped():
    """Autoescaping was silently OFF: select_autoescape(["html"]) matches on
    the filename suffix and 'report.html.j2' ends in '.j2'. Sentry titles and
    LLM output are attacker-influenced text and must never land in the HTML
    raw -- the HTML fallback report is opened in a browser."""
    results = [ta('<script>alert("xss")</script>')]
    html = render_report_html("run-1", "2026-08-02", stats_for(results), results)
    assert "<script>alert(" not in html
    assert "&lt;script&gt;" in html
