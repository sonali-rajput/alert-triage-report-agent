"""The BigQuery query contract.

`LocalBQStore` is what every other test exercises, so this is the only place
the real SQL is looked at. It cannot run BigQuery, and it does not pretend to —
what it pins is the contract: which tables are read, which filters are present,
and which parameters are bound with which types.

That is worth pinning because two lines in `similar_past` are load-bearing and
silently catastrophic if lost:

  * `base.run_date < @run_date` — without it, every alert is its own nearest
    neighbour at distance 0, the agent marks the entire run duplicate, and the
    report goes empty. It would look like a very quiet day.
  * the top_k padding — VECTOR_SEARCH returns today's rows too, so an unpadded
    top_k comes back full of same-run siblings and no history at all.

Neither would raise. Both would just quietly stop the pipeline working.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from google.cloud import bigquery

from pipeline.bq import (
    ALERTS_TABLE,
    MAX_NEIGHBOUR_DISTANCE,
    RUNS_TABLE,
    SELF_MATCH_PADDING,
    TRIAGE_TABLE,
    BigQueryStore,
)


class StubClient:
    """Records what the store asked for and returns canned rows."""

    def __init__(self, rows: list[dict] | None = None, insert_errors: list | None = None):
        self.project = "test-project"
        self.rows = rows or []
        self.insert_errors = insert_errors or []
        self.queries: list[str] = []
        self.params: list[list] = []
        self.inserts: list[tuple[str, list[dict]]] = []

    def query(self, sql, job_config=None):
        self.queries.append(sql)
        self.params.append(list(job_config.query_parameters) if job_config else [])
        return SimpleNamespace(result=lambda: [dict(r) for r in self.rows])

    def insert_rows_json(self, table, rows):
        self.inserts.append((table, rows))
        return self.insert_errors


def store_over(client: StubClient) -> BigQueryStore:
    s = BigQueryStore.__new__(BigQueryStore)   # skip __init__: it builds a real client
    s._client = client
    s._project = client.project
    s._dataset = "prodtools_triage"
    return s


def params_of(client: StubClient, call: int = 0) -> dict:
    return {p.name: p for p in client.params[call]}


# --------------------------------------------------------------------------
# The vector search
# --------------------------------------------------------------------------


def test_the_search_excludes_the_current_run_from_the_history():
    """The single most dangerous line in the SQL. VECTOR_SEARCH's base side has
    to be a real table, so today's rows — inserted moments earlier — are in it,
    and each alert matches itself perfectly."""
    client = StubClient()
    store_over(client).similar_past("run-1", "2026-08-18")

    sql = client.queries[0]
    assert "base.run_date < @run_date" in sql


def test_the_search_asks_for_more_neighbours_than_it_needs():
    """Padding absorbs the self-match and same-run siblings before the real
    history begins. Without it the caller gets today's alerts back as 'history'."""
    client = StubClient()
    store_over(client).similar_past("run-1", "2026-08-18", top_k=3)

    padded = params_of(client)["padded_k"]
    assert padded.value == 3 + SELF_MATCH_PADDING
    assert padded.type_ == "INT64"


def test_the_search_is_a_cosine_vector_search_over_the_alerts_table():
    client = StubClient()
    store_over(client).similar_past("run-1", "2026-08-18")

    sql = client.queries[0]
    assert "VECTOR_SEARCH" in sql
    assert "distance_type => 'COSINE'" in sql
    assert f"test-project.prodtools_triage.{ALERTS_TABLE}" in sql


def test_the_search_joins_the_past_verdicts():
    """'We have seen this' is weak; 'we saw it and called it low three days
    running' is the fact that should stop it being re-reported."""
    client = StubClient()
    store_over(client).similar_past("r", "2026-08-18")
    assert f"test-project.prodtools_triage.{TRIAGE_TABLE}" in client.queries[0]
    assert "LEFT JOIN" in client.queries[0]


def test_distant_neighbours_are_excluded_in_sql_not_in_python():
    """Filtering server-side keeps the response small; the ceiling itself is a
    display decision, never the duplicate decision."""
    client = StubClient()
    store_over(client).similar_past("r", "2026-08-18")

    assert "distance <= @max_distance" in client.queries[0]
    assert params_of(client)["max_distance"].value == MAX_NEIGHBOUR_DISTANCE


def test_parameters_are_bound_not_interpolated():
    """run_date reaches here from a CLI flag. Bound parameters are what keep
    that from being a SQL injection surface."""
    client = StubClient()
    store_over(client).similar_past("run-1", "2026-08-18")

    params = params_of(client)
    assert params["run_date"].type_ == "DATE"
    assert params["run_id"].type_ == "STRING"
    assert "2026-08-18" not in client.queries[0]


def test_results_are_grouped_by_alert_and_trimmed_to_top_k():
    rows = [
        {"alert_id": "a", "past_alert_id": f"p{i}", "title": "t", "project": "proj",
         "run_date": "2026-08-17", "priority": "low", "decision": "notify", "distance": 0.01 * i}
        for i in range(5)
    ]
    neighbours = store_over(StubClient(rows)).similar_past("r", "2026-08-18", top_k=2)

    assert list(neighbours) == ["a"]
    assert [n["alert_id"] for n in neighbours["a"]] == ["p0", "p1"]


def test_an_untriaged_neighbour_is_labelled_not_guessed():
    """A run that degraded wrote alerts but no verdicts, so the join returns
    NULL. Rendering that as an empty string would read to the model as 'we
    looked at it and decided nothing was wrong'."""
    rows = [{"alert_id": "a", "past_alert_id": "p", "title": "t", "project": "x",
             "run_date": "2026-08-17", "priority": None, "decision": None, "distance": 0.02}]
    neighbour = store_over(StubClient(rows)).similar_past("r", "2026-08-18")["a"][0]

    assert neighbour["past_priority"] == "not triaged"
    assert neighbour["past_decision"] == "not triaged"


# --------------------------------------------------------------------------
# Writes
# --------------------------------------------------------------------------


def test_alerts_are_inserted_into_the_alerts_table():
    client = StubClient()
    store_over(client).insert_alerts("r1", "2026-08-18", [{"alert_id": "a"}])

    table, rows = client.inserts[0]
    assert table.endswith(f".{ALERTS_TABLE}")
    assert rows == [{"alert_id": "a"}]


def test_an_insert_failure_is_raised_not_swallowed():
    """The orchestrator catches this and continues without history. It can only
    do that if the store actually says something went wrong — a silent partial
    insert would poison the next run's dedup instead."""
    client = StubClient(insert_errors=[{"index": 0, "errors": ["bad row"]}])
    with pytest.raises(RuntimeError, match="bad row"):
        store_over(client).insert_alerts("r1", "2026-08-18", [{"alert_id": "a"}])


def test_an_empty_insert_does_not_call_the_api():
    client = StubClient()
    store_over(client).insert_alerts("r1", "2026-08-18", [])
    store_over(client).record_triage([])
    assert client.inserts == []


def test_the_run_row_carries_its_date():
    client = StubClient()
    store_over(client).record_run("2026-08-18", {"run_id": "r1"})

    table, rows = client.inserts[0]
    assert table.endswith(f".{RUNS_TABLE}")
    assert rows[0]["run_date"] == "2026-08-18"


# --------------------------------------------------------------------------
# Run history
# --------------------------------------------------------------------------


def test_get_run_takes_the_latest_attempt_for_a_date():
    """A degraded run and its retry both write a row for the same date, and the
    idempotency guard reads this."""
    client = StubClient([{"run_id": "second", "stats": "{}"}])
    store_over(client).get_run("2026-08-18")
    assert "ORDER BY completed_at DESC" in client.queries[0]
    assert "LIMIT 1" in client.queries[0]


def test_stats_are_decoded_from_the_json_column():
    client = StubClient([{"run_id": "r", "stats": '{"processed": 7}'}])
    assert store_over(client).get_run("2026-08-18")["stats"]["processed"] == 7


def test_unparseable_stats_degrade_to_empty_rather_than_crashing():
    """A truncated write should cost the report its delta line, not the run."""
    client = StubClient([{"run_id": "r", "stats": "{not json"}])
    assert store_over(client).get_run("2026-08-18")["stats"] == {}


def test_recent_runs_is_bounded_and_strictly_before_the_date():
    client = StubClient([{"run_id": "r", "stats": "{}"}])
    store_over(client).recent_runs("2026-08-18", limit=7)

    sql = client.queries[0]
    assert "run_date < @before_date" in sql
    assert "ORDER BY run_date DESC" in sql
    assert params_of(client)["limit"].value == 7


def test_query_parameters_are_the_sdk_types_the_client_expects():
    """Cheap guard against a hand-rolled dict that only fails at call time."""
    client = StubClient()
    store_over(client).similar_past("r", "2026-08-18")
    assert all(isinstance(p, bigquery.ScalarQueryParameter) for p in client.params[0])


# --------------------------------------------------------------------------
# The same-run sibling query
# --------------------------------------------------------------------------


def test_the_sibling_search_stays_inside_the_run():
    """The mirror image of the historical search's filter. If this one leaked
    other runs, yesterday's alert could take today's report slot."""
    client = StubClient()
    store_over(client).similar_within_run("run-1")

    sql = client.queries[0]
    assert "base.run_id = @run_id" in sql
    assert "base.alert_id != query.alert_id" in sql


def test_the_sibling_search_uses_the_tighter_ceiling():
    from pipeline.bq import MAX_SIBLING_DISTANCE

    client = StubClient()
    store_over(client).similar_within_run("run-1")
    assert params_of(client)["max_distance"].value == MAX_SIBLING_DISTANCE


def test_the_sibling_search_does_not_join_verdicts():
    """Nothing in this run has a verdict yet; joining would return NULLs and
    invite the model to read a row as already-triaged."""
    client = StubClient()
    store_over(client).similar_within_run("run-1")
    # Not a bare `TRIAGE_TABLE not in sql`: "triage" is a substring of the
    # dataset name `prodtools_triage`, so that assertion can never fail.
    assert f".{TRIAGE_TABLE}`" not in client.queries[0]
    assert "LEFT JOIN" not in client.queries[0]
