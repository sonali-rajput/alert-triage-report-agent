"""The store: vector search for history, plus the audit trail and run history.

Exercised through LocalBQStore, which is the offline twin of BigQueryStore.
The behaviour under test is the contract both sides implement -- neighbours
ranked by cosine distance, today's own rows excluded, a distance ceiling, and
past verdicts attached -- not BigQuery's SQL.
"""

from __future__ import annotations

import pytest

from pipeline.bq import MAX_NEIGHBOUR_DISTANCE, LocalBQStore, build_store
from pipeline.embeddings import HashEmbedder


@pytest.fixture
def store(tmp_path):
    return LocalBQStore(str(tmp_path / "state"))


def row(alert_id: str, run_id: str, run_date: str, text: str, **kwargs) -> dict:
    base = {
        "run_id": run_id,
        "run_date": run_date,
        "alert_id": alert_id,
        "title": text,
        "project": "core-tools",
        "embedding": HashEmbedder().embed([text])[0],
    }
    base.update(kwargs)
    return base


def test_finds_the_same_error_from_a_previous_run(store):
    text = "ConnectionError: could not reach internal-db-01"
    store.insert_alerts("r1", "2026-07-15", [row("old", "r1", "2026-07-15", text)])
    store.insert_alerts("r2", "2026-07-16", [row("new", "r2", "2026-07-16", text)])

    neighbours = store.similar_past("r2", "2026-07-16")

    assert list(neighbours) == ["new"]
    assert neighbours["new"][0]["alert_id"] == "old"
    assert neighbours["new"][0]["distance"] == pytest.approx(0.0, abs=1e-9)


def test_an_alert_is_never_its_own_neighbour(store):
    """Today's rows are in the table before the search runs -- that is what
    makes the search one query instead of one per alert -- so excluding them
    is load-bearing. Without it every alert is a perfect match for itself and
    the agent would mark the entire run duplicate."""
    store.insert_alerts(
        "r1", "2026-07-15",
        [row("a", "r1", "2026-07-15", "same text"), row("b", "r1", "2026-07-15", "same text")],
    )
    assert store.similar_past("r1", "2026-07-15") == {}


def test_unrelated_history_is_not_offered_as_context(store):
    """Beyond the ceiling the model is being asked to compare an error against
    an unrelated one, which invites a spurious duplicate call."""
    store.insert_alerts(
        "r1", "2026-07-15",
        [row("old", "r1", "2026-07-15", "TypeError cannot read property length of undefined")],
    )
    store.insert_alerts(
        "r2", "2026-07-16",
        [row("new", "r2", "2026-07-16", "SSL certificate verify failed on the publish gateway")],
    )
    assert store.similar_past("r2", "2026-07-16") == {}


def test_neighbours_are_ranked_closest_first_and_capped(store):
    base = "ConnectionError could not reach internal-db-01 port 5432"
    store.insert_alerts("r1", "2026-07-14", [
        row("exact", "r1", "2026-07-14", base),
        row("close", "r1", "2026-07-14", base + " retry 2"),
        row("closer", "r1", "2026-07-14", base + " retry"),
        row("far", "r1", "2026-07-14", base + " retry 2 3 4 timeout gateway"),
    ])
    store.insert_alerts("r2", "2026-07-16", [row("new", "r2", "2026-07-16", base)])

    neighbours = store.similar_past("r2", "2026-07-16", top_k=2)["new"]

    assert len(neighbours) == 2
    assert neighbours[0]["alert_id"] == "exact"
    assert neighbours[0]["distance"] <= neighbours[1]["distance"] <= MAX_NEIGHBOUR_DISTANCE


def test_a_neighbour_carries_what_we_decided_about_it_last_time(store):
    """The fact that should stop an alert being re-reported is not "we have
    seen this" but "we saw it and called it low three days running"."""
    text = "Scheduled cleanup job failed"
    store.insert_alerts("r1", "2026-07-15", [row("old", "r1", "2026-07-15", text)])
    store.record_triage([
        {"run_id": "r1", "run_date": "2026-07-15", "alert_id": "old",
         "priority": "low", "decision": "ignore"}
    ])
    store.insert_alerts("r2", "2026-07-16", [row("new", "r2", "2026-07-16", text)])

    neighbour = store.similar_past("r2", "2026-07-16")["new"][0]
    assert neighbour["past_priority"] == "low"
    assert neighbour["past_decision"] == "ignore"


def test_an_untriaged_neighbour_says_so_rather_than_looking_clean(store):
    """A run that degraded wrote alerts but no verdicts. An empty priority
    rendering as "" would read to the model as "we decided nothing was wrong"."""
    text = "Some error"
    store.insert_alerts("r1", "2026-07-15", [row("old", "r1", "2026-07-15", text)])
    store.insert_alerts("r2", "2026-07-16", [row("new", "r2", "2026-07-16", text)])

    neighbour = store.similar_past("r2", "2026-07-16")["new"][0]
    assert neighbour["past_priority"] == "not triaged"


def test_alerts_with_no_embedding_are_skipped(store):
    """An alert stored before the embedding stage existed, or one whose embed
    call failed. It must not be compared, and must not crash the scan."""
    store.insert_alerts("r1", "2026-07-15", [row("old", "r1", "2026-07-15", "text", embedding=[])])
    store.insert_alerts("r2", "2026-07-16", [row("new", "r2", "2026-07-16", "text")])
    assert store.similar_past("r2", "2026-07-16") == {}


# --------------------------------------------------------------------------
# Run history
# --------------------------------------------------------------------------


def test_get_run_returns_the_latest_attempt_for_a_date(store):
    """A degraded run and its retry both write a row for the same date. The
    idempotency guard reads this, so it has to see the retry."""
    store.record_run("2026-07-15", {"run_id": "first", "degraded": True, "stats": "{}"})
    store.record_run("2026-07-15", {"run_id": "second", "degraded": False, "stats": "{}"})
    assert store.get_run("2026-07-15")["run_id"] == "second"


def test_stats_round_trip_through_the_json_column(store):
    """`stats` is a JSON string, not a RECORD -- RunStats gains a field every
    time the pipeline learns to count something new, and a schema migration
    per counter is not a trade worth making."""
    store.record_run("2026-07-15", {"run_id": "r1", "stats": '{"processed": 7}'})
    assert store.get_run("2026-07-15")["stats"]["processed"] == 7


def test_recent_runs_is_newest_first_and_strictly_before_the_date(store):
    for day in ("13", "14", "15", "16"):
        store.record_run(f"2026-07-{day}", {"run_id": day, "stats": "{}"})
    recent = store.recent_runs("2026-07-16", limit=2)
    assert [r["run_id"] for r in recent] == ["15", "14"]


def test_unknown_backend_is_rejected_loudly():
    with pytest.raises(ValueError):
        build_store("firestore", "p", "d")


# --------------------------------------------------------------------------
# Same-run siblings
#
# Sentry groups events into issues by a fingerprint derived from the stack, so
# one failure arrives as several issues whenever the stack shape differs. On a
# real prodtools run, two issues 0.027 apart took slots #1 and #4 of a ten-slot
# report. `similar_past` cannot show that -- it looks only at previous runs.
# --------------------------------------------------------------------------


def test_siblings_are_found_within_the_same_run(store):
    text = "ConnectionError: could not reach internal-db-01"
    store.insert_alerts("r1", "2026-07-15", [
        row("a", "r1", "2026-07-15", text),
        row("b", "r1", "2026-07-15", text + " (retry)"),
    ])

    siblings = store.similar_within_run("r1")

    assert siblings["a"][0]["alert_id"] == "b"
    assert siblings["b"][0]["alert_id"] == "a"


def test_an_alert_is_never_its_own_sibling(store):
    store.insert_alerts("r1", "2026-07-15", [row("only", "r1", "2026-07-15", "lonely error")])
    assert store.similar_within_run("r1") == {}


def test_siblings_never_come_from_another_run(store):
    """The two searches answer different questions. A historical match is a
    duplicate; a same-run match is a report-slot decision. Mixing them would
    let yesterday's alert suppress today's."""
    text = "ConnectionError: could not reach internal-db-01"
    store.insert_alerts("r1", "2026-07-15", [row("old", "r1", "2026-07-15", text)])
    store.insert_alerts("r2", "2026-07-16", [row("new", "r2", "2026-07-16", text)])

    assert store.similar_within_run("r2") == {}
    assert store.similar_past("r2", "2026-07-16")["new"][0]["alert_id"] == "old"


def test_the_sibling_ceiling_is_tighter_than_the_historical_one(store):
    """A same-run merge costs an issue nobody sees; a missed historical
    duplicate costs a repeated row. The failure directions are not symmetric,
    so the thresholds are not either."""
    from pipeline.bq import MAX_NEIGHBOUR_DISTANCE, MAX_SIBLING_DISTANCE

    assert MAX_SIBLING_DISTANCE < MAX_NEIGHBOUR_DISTANCE

    base = "ConnectionError could not reach internal-db-01 port 5432"
    store.insert_alerts("r1", "2026-07-15", [
        row("a", "r1", "2026-07-15", base),
        row("far", "r1", "2026-07-15", "TypeError cannot read property length of undefined"),
    ])
    assert "far" not in {s["alert_id"] for s in store.similar_within_run("r1").get("a", [])}


def test_a_sibling_carries_no_verdict_because_there_is_none_yet(store):
    """Nothing in this run has been triaged. A `past_priority` field here would
    be an invitation to read one of today's alerts as already-decided."""
    text = "Some error"
    store.insert_alerts("r1", "2026-07-15", [
        row("a", "r1", "2026-07-15", text), row("b", "r1", "2026-07-15", text)])

    sibling = store.similar_within_run("r1")["a"][0]
    assert set(sibling) == {"alert_id", "title", "project", "distance"}


def test_siblings_are_capped_per_alert(store):
    """This rides on every alert's payload, so an unbounded list multiplies
    across the whole prompt."""
    text = "ConnectionError could not reach internal-db-01"
    store.insert_alerts("r1", "2026-07-15",
                        [row(f"a{i}", "r1", "2026-07-15", text) for i in range(6)])
    assert all(len(s) <= 2 for s in store.similar_within_run("r1", top_k=2).values())
