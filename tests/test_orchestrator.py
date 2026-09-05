"""End-to-end pipeline tests: fake Sentry, mock LLM, local store, offline
embedder. No network, no GCP, no token."""

import pytest

from pipeline.agents.providers import LLMError, MockProvider
from pipeline.bq import LocalBQStore
from pipeline.embeddings import HashEmbedder
from pipeline.orchestrator import execute_run
from pipeline.storage import ArtifactStore
from shared.models import Alert, AlertSource


class FakeSentry:
    """In-memory SentrySource: returns a fixed list of already-detailed alerts."""

    def __init__(self, alerts: list[Alert]):
        self._alerts = alerts

    def fetch_issues(self, window_hours: int) -> list[Alert]:
        return list(self._alerts)

    def close(self) -> None:
        pass


class FailingSentry:
    def fetch_issues(self, window_hours: int) -> list[Alert]:
        raise RuntimeError("sentry unreachable")

    def close(self) -> None:
        pass


@pytest.fixture
def store(tmp_path):
    return LocalBQStore(str(tmp_path / "state"))


@pytest.fixture
def artifacts(tmp_path):
    return ArtifactStore(local_dir=str(tmp_path / "artifacts"))


def sample_alerts() -> list[Alert]:
    return [
        Alert(
            source=AlertSource.sentry, source_id="1", kind="sentry_issue",
            title="Connection pool exhausted, all users affected",
            body="fatal", project="asset-service", event_count=900, user_count=400,
            environment="production", url="https://framestore.sentry.io/issues/1/",
        ),
        Alert(
            source=AlertSource.sentry, source_id="2", kind="sentry_issue",
            title="Known flaky test failing intermittently",
            body="flaky test under CI load", project="render-submit", event_count=4,
        ),
        Alert(
            source=AlertSource.sentry, source_id="3", kind="sentry_issue",
            title="DeprecationWarning: imp module is deprecated",
            body="warning", project="publish-hooks", event_count=500, user_count=2,
        ),
    ]


def run(store, artifacts, provider=None, run_date="2026-07-15", alerts=None, **kwargs):
    return execute_run(
        run_date,
        sentry=FakeSentry(sample_alerts() if alerts is None else alerts),
        store=store,
        provider=provider or MockProvider(),
        embedder=HashEmbedder(),
        artifacts=artifacts,
        **kwargs,
    )


def test_full_run(store, artifacts):
    result = run(store, artifacts)

    assert not result.degraded
    assert result.stats.ingested == 3
    # The flaky-test alert is dropped by the prefilter before anything is
    # embedded, stored or sent to a model.
    assert result.stats.prefiltered == 1
    assert result.stats.considered == 2
    assert result.stats.processed == 2
    assert len(result.results) == 2

    # Ranks are dense and start at 1, whatever the model returned.
    assert sorted(r.rank for r in result.results) == [1, 2]
    # The 400-user production alert outranks the deprecation warning.
    assert result.results[0].project == "asset-service"
    assert result.results[0].selection_reason

    critical = [r for r in result.results if r.triage.priority.value == "critical"]
    assert len(critical) == 1

    # Report artifact written (HTML in local dev, PDF in Docker).
    assert result.pdf_url

    # Run recorded, and the stats round-trip through the JSON column.
    recorded = store.get_run("2026-07-15")
    assert recorded["run_id"] == result.run_id
    assert recorded["stats"]["processed"] == 2


def test_every_considered_alert_gets_an_audit_row(store, artifacts, tmp_path):
    """Selected or not. 'Why was this NOT in today's report' is the question
    the audit trail most often has to answer."""
    run(store, artifacts, alerts=sample_alerts() + [
        Alert(source=AlertSource.sentry, source_id="9", kind="sentry_issue",
              title="Quiet unrelated warning", body="nothing much", project="misc",
              event_count=2),
    ], top_n=1)

    rows = store._load("triage")
    assert len(rows) == 3  # the three that survived the prefilter
    assert sum(1 for r in rows if r["selected"]) == 1
    unselected = [r for r in rows if not r["selected"]]
    assert all(r["selection_reason"] for r in unselected)


def test_alerts_and_embeddings_are_stored_for_the_next_run(store, artifacts):
    run(store, artifacts)
    rows = store._load("alerts")
    assert len(rows) == 2
    assert all(len(r["embedding"]) == 768 for r in rows)
    # The projection, not the raw payload.
    assert "activity" not in rows[0]


def test_second_run_dedups_via_vector_search(store, artifacts):
    """The same errors on the next day are near-identical in vector space, so
    the top-issues agent marks them repeats and the report stays empty."""
    run(store, artifacts, run_date="2026-07-15")
    result = run(store, artifacts, run_date="2026-07-16")

    assert result.stats.deduped == 2
    assert result.stats.processed == 0


def test_a_changed_error_is_not_deduped(store, artifacts):
    """Dedup is about the error, not the counter: a genuinely different error
    the next day still gets through."""
    run(store, artifacts, run_date="2026-07-15")
    fresh = [
        Alert(
            source=AlertSource.sentry, source_id="7", kind="sentry_issue",
            title="TLS certificate expired on the publish gateway",
            body="ssl handshake failed", project="publish-gateway",
            event_count=30, user_count=12, environment="production",
        )
    ]
    result = run(store, artifacts, run_date="2026-07-16", alerts=fresh)

    assert result.stats.deduped == 0
    assert result.stats.processed == 1


def test_run_history_round_trips_from_the_store_into_the_report(store, artifacts):
    """A seam three modules share and none of them owns: the orchestrator
    writes `stats` as a JSON *string* (RunStats gains fields too often for a
    RECORD), the store decodes it, and the report's 'vs yesterday' line reads
    it as a dict. A regression anywhere along that path costs the deltas
    silently — the report still renders, just without the comparison."""
    from pipeline.pdf_report import build_scorecard

    run(store, artifacts, run_date="2026-07-15")
    result = run(store, artifacts, run_date="2026-07-16")

    history = store.recent_runs("2026-07-16", limit=7)
    assert history, "yesterday's run did not come back"
    assert isinstance(history[0]["stats"], dict), "stats did not survive the JSON column"

    scorecard = build_scorecard(result.stats, history)
    assert scorecard["ingested"].change is not None


def test_vector_search_failure_does_not_fail_the_run(store, artifacts, monkeypatch):
    """Losing the history costs the dedup signal, not the run -- and it
    over-reports rather than suppressing, which is the right way round."""
    monkeypatch.setattr(
        store, "similar_past", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("bq down"))
    )
    result = run(store, artifacts)
    assert not result.degraded
    assert result.stats.processed == 2
    assert all(r.similar_past_count == 0 for r in result.results)


class FailingProvider:
    def generate_list(self, system, prompt, item_schema):
        raise LLMError("vertex is down")


def test_degraded_run_on_llm_failure(store, artifacts):
    result = run(store, artifacts, provider=FailingProvider())
    assert result.degraded
    assert result.results == []
    assert store.get_run("2026-07-15")["degraded"] is True


class FailingTriageProvider(MockProvider):
    """Selection succeeds, triage fails. The run still has to degrade rather
    than report ten issues with no verdicts."""

    def generate_list(self, system, prompt, item_schema):
        from shared.models import TriageOutput

        if item_schema is TriageOutput:
            raise LLMError("vertex is down")
        return super().generate_list(system, prompt, item_schema)


def test_degraded_run_when_only_triage_fails(store, artifacts):
    result = run(store, artifacts, provider=FailingTriageProvider())
    assert result.degraded
    # The stats earned before the failure are kept, not zeroed.
    assert result.stats.considered == 2


def test_second_call_for_the_same_date_is_a_no_op(store, artifacts):
    """The retry-safety guard. Without it a re-run would find today's alerts
    already in the vector table and dutifully mark every one of them a
    duplicate of itself -- posting an empty 'all clear' card, and paying for a
    second Sentry fetch and two more LLM passes to do it."""
    first = run(store, artifacts)
    again = run(store, artifacts)
    assert again.run_id == first.run_id
    assert again.results == []
    assert again.stats.processed == first.stats.processed


def test_force_reruns_a_completed_date(store, artifacts):
    first = run(store, artifacts)
    forced = run(store, artifacts, force=True)
    assert forced.run_id != first.run_id


def test_a_degraded_run_does_not_block_a_retry(store, artifacts):
    """Only a *completed* run short-circuits: after a degraded run the same
    date must still be re-runnable so the day actually gets triaged."""
    run(store, artifacts, provider=FailingProvider())
    result = run(store, artifacts)
    assert not result.degraded
    assert result.stats.processed == 2


# --------------------------------------------------------------------------
# The "log and continue" promises
#
# Each of these stages is wrapped for a reason: by the time they run, the
# expensive work is done and the answer exists. Losing the answer because the
# cheap tail failed would be the worst possible trade.
# --------------------------------------------------------------------------


def test_a_report_failure_still_notifies_and_still_records(store, artifacts, monkeypatch):
    """WeasyPrint's native libs are a real-world absentee. The team still needs
    the card, and tomorrow still needs the run record."""
    from pipeline import orchestrator

    posted: list = []
    monkeypatch.setattr(orchestrator, "render_report",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no pango")))
    monkeypatch.setattr(orchestrator, "post_run_summary",
                        lambda *a, **k: posted.append(a))

    result = run(store, artifacts)

    assert not result.degraded
    assert result.pdf_url == ""
    assert posted, "the Chat card was lost with the report"
    assert store.get_run("2026-07-15")["run_id"] == result.run_id


def test_a_chat_failure_still_records_the_run(store, artifacts, monkeypatch):
    """If the run is not recorded, the Scheduler retry re-runs the whole day —
    and finds today's alerts already in the vector table, so it dedups the lot
    and posts an empty card. A webhook outage must not cause that."""
    from pipeline import orchestrator

    monkeypatch.setattr(orchestrator, "post_run_summary",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("chat 503")))

    result = run(store, artifacts)

    assert result.stats.processed == 2
    assert store.get_run("2026-07-15") is not None


def test_an_audit_write_failure_does_not_cost_the_report(store, artifacts, monkeypatch):
    """The record of the answer is worth less than the answer."""
    monkeypatch.setattr(store, "record_triage",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("bq insert failed")))

    result = run(store, artifacts)

    assert result.pdf_url
    assert len(result.results) == 2


def test_a_failure_to_record_the_run_still_returns_the_result(store, artifacts, monkeypatch):
    """The caller has the triaged alerts in hand; the cost of this failure is
    that the date is no longer idempotent, not that the work is lost."""
    monkeypatch.setattr(store, "record_run",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("bq down")))

    result = run(store, artifacts)
    assert len(result.results) == 2


def test_unreadable_history_costs_the_deltas_not_the_report(store, artifacts, monkeypatch):
    monkeypatch.setattr(store, "recent_runs",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("bq down")))

    result = run(store, artifacts)
    assert result.pdf_url


def test_a_degraded_run_that_cannot_post_still_records_itself(store, artifacts, monkeypatch):
    """Worst case: the model failed AND Chat is down. The run record is what
    stops the retry loop from being infinite and silent."""
    from pipeline import orchestrator

    monkeypatch.setattr(orchestrator, "post_fallback_digest",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("chat 503")))

    result = run(store, artifacts, provider=FailingProvider())

    assert result.degraded
    assert store.get_run("2026-07-15")["degraded"] is True


def test_sentry_fetch_failure_raises(store, artifacts):
    with pytest.raises(RuntimeError):
        execute_run(
            "2026-07-15",
            sentry=FailingSentry(),
            store=store,
            provider=MockProvider(),
            embedder=HashEmbedder(),
            artifacts=artifacts,
        )
    # No run record written when ingestion never completed.
    assert store.get_run("2026-07-15") is None


def test_alerts_carry_both_kinds_of_neighbour(store, artifacts):
    """The two searches answer different questions and both have to reach the
    agent: history decides `is_duplicate`, same-run siblings decide how many
    report slots one incident gets."""
    twins = [
        Alert(source=AlertSource.sentry, source_id="t1", kind="sentry_issue",
              title="Lost connection to the server", body="fatal",
              project="framecast-ui", user_count=105, environment="production"),
        Alert(source=AlertSource.sentry, source_id="t2", kind="sentry_issue",
              title="Lost connection to the server (retry)", body="fatal",
              project="framecast-ui", user_count=1, environment="production"),
    ]
    seen: dict[str, list] = {}

    class Recording(MockProvider):
        def generate_list(self, system, prompt, item_schema):
            from pipeline.agents.providers import extract_input_json
            from shared.models import SelectedIssue

            if item_schema is SelectedIssue:
                for item in extract_input_json(prompt):
                    seen[item["alert_id"]] = item.get("similar_today", [])
            return super().generate_list(system, prompt, item_schema)

    run(store, artifacts, alerts=twins, provider=Recording())

    # Day one: no history, but the twins must still see each other.
    assert any(seen.values()), "the agent never saw a same-run sibling"
    assert all("distance" in s for siblings in seen.values() for s in siblings)


def test_a_same_run_sibling_is_not_treated_as_history(store, artifacts):
    """A sibling must never make something a duplicate: nothing in today's run
    has been reported yet, so 'we have seen this' is false by construction."""
    twins = [
        Alert(source=AlertSource.sentry, source_id="t1", kind="sentry_issue",
              title="Identical error text", body="same", project="p", user_count=5),
        Alert(source=AlertSource.sentry, source_id="t2", kind="sentry_issue",
              title="Identical error text", body="same", project="p", user_count=5),
    ]
    result = run(store, artifacts, alerts=twins)

    assert result.stats.deduped == 0, "a same-run sibling was counted as a repeat"
    assert all(not a.similar_past for a in twins)


def test_a_sibling_search_failure_costs_the_signal_not_the_run(store, artifacts, monkeypatch):
    monkeypatch.setattr(
        store, "similar_within_run", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("bq down"))
    )
    result = run(store, artifacts)
    assert not result.degraded
    assert result.stats.processed == 2
