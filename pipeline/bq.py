"""BigQuery: the vector store, the audit trail and the run history, in one place.

This replaced three things at once -- Firestore for alert state, a hash-based
dedup table, and a GCS archive of the raw payload -- and the consolidation is
the point:

  * Dedup is a *similarity* question, so it needs the vectors. The vectors have
    to live in a queryable store. BigQuery does both natively with
    VECTOR_SEARCH, so adding Firestore alongside it would be a second database
    holding a worse copy of the same facts.
  * Everything the pipeline records is something someone will eventually want
    to ask a question about ("show me every alert triaged critical last month,
    by project"). In BigQuery that is one SQL query. In Firestore plus GCS JSON
    it is an export job first.

Three tables, created by Terraform (infra/modules/bigquery), never by this
code:

  alerts  one row per alert per run, including its embedding. This is both the
          audit archive of what we ingested and the vector index we search.
  triage  one row per triaged alert: the selection verdict and the triage
          verdict, with the prompt version that produced them.
  runs    one row per run: stats, the report URI, whether it degraded.

LocalBQStore is the offline twin, JSON files plus a brute-force cosine scan.
It exists so the whole dedup path can be tested and demoed with no GCP project,
and it ranks neighbours identically because both sides use cosine distance.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any, Protocol

from pipeline.embeddings import cosine_distance

logger = logging.getLogger(__name__)

ALERTS_TABLE = "alerts"
TRIAGE_TABLE = "triage"
RUNS_TABLE = "runs"

# How many historical neighbours to hand the top-issues agent per alert. Three
# is enough to show "we have seen this on each of the last three days" without
# tripling the prompt.
DEFAULT_TOP_K = 3
# Same-run neighbours per alert. Smaller than DEFAULT_TOP_K because this rides
# on EVERY alert's payload rather than being a historical footnote, and because
# the question it answers -- "is this the same incident as another one in
# today's list?" -- is usually settled by the single closest sibling.
DEFAULT_SIBLING_K = 2
# Neighbours further away than this are not shown at all. A cosine distance of
# 0.35 on normalized embeddings is "clearly about the same subsystem"; beyond
# it the model is being asked to compare an error against an unrelated one,
# which invites a spurious duplicate call. This is a *display* threshold only
# -- the duplicate decision itself is the model's, never a threshold's.
# FALLBACKS ONLY. The real thresholds come from the embedder that produced the
# vectors -- see Embedder.neighbour_distance -- because they are properties of
# the vector space, and the two embedders' spaces are scaled very differently
# (median pair 0.791 offline vs 0.352 for gemini-embedding-001 on the same 19
# alerts). These defaults match the offline embedder so a caller that passes
# nothing gets the old behaviour rather than a silent change.
MAX_NEIGHBOUR_DISTANCE = 0.35
# Tighter than the historical ceiling. A historical neighbour at 0.3 is useful
# context ("we have seen something like this before"); a same-run sibling at 0.3
# is a different error in the same subsystem, and showing it invites the model
# to collapse two genuine issues into one report slot. The failure directions
# are not symmetric: a missed historical duplicate costs a repeated row, a
# wrong same-run merge costs an issue nobody sees.
MAX_SIBLING_DISTANCE = 0.15
# Extra top_k requested from VECTOR_SEARCH to absorb the self-match and other
# alerts from the same run before the historical neighbours begin.
SELF_MATCH_PADDING = 8


class TriageStore(Protocol):
    def insert_alerts(self, run_id: str, run_date: str, rows: list[dict[str, Any]]) -> None: ...
    def similar_past(
        self, run_id: str, run_date: str, top_k: int = DEFAULT_TOP_K,
        max_distance: float = MAX_NEIGHBOUR_DISTANCE,
    ) -> dict[str, list[dict[str, Any]]]: ...
    def similar_within_run(
        self, run_id: str, top_k: int = DEFAULT_SIBLING_K,
        max_distance: float = MAX_SIBLING_DISTANCE,
    ) -> dict[str, list[dict[str, Any]]]: ...
    def record_triage(self, rows: list[dict[str, Any]]) -> None: ...
    def record_run(self, run_date: str, data: dict[str, Any]) -> None: ...
    def get_run(self, run_date: str) -> dict[str, Any] | None: ...
    def recent_runs(self, before_date: str, limit: int = 7) -> list[dict[str, Any]]: ...


class BigQueryStore:
    def __init__(self, project: str, dataset: str):
        from google.cloud import bigquery

        self._client = bigquery.Client(project=project or None)
        self._project = self._client.project
        self._dataset = dataset

    def _table(self, name: str) -> str:
        return f"{self._project}.{self._dataset}.{name}"

    def _query(self, sql: str, params: list) -> list[dict[str, Any]]:
        from google.cloud import bigquery

        job = self._client.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=params))
        return [dict(row) for row in job.result()]

    # -- writes -------------------------------------------------------------

    def insert_alerts(self, run_id: str, run_date: str, rows: list[dict[str, Any]]) -> None:
        """Insert this run's alerts, embeddings included.

        This happens BEFORE the vector search, not after, so the search can run
        as a single query with today's rows as its query side rather than one
        round trip per alert. Today's rows are excluded from the base side by
        run_date in the search's WHERE clause, so an alert never matches itself.
        """
        if not rows:
            return
        errors = self._client.insert_rows_json(self._table(ALERTS_TABLE), rows)
        if errors:
            raise RuntimeError(f"BigQuery alert insert failed: {errors[:3]}")
        logger.info("bq: inserted %d alerts for run %s", len(rows), run_id)

    def record_triage(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        errors = self._client.insert_rows_json(self._table(TRIAGE_TABLE), rows)
        if errors:
            raise RuntimeError(f"BigQuery triage insert failed: {errors[:3]}")

    def record_run(self, run_date: str, data: dict[str, Any]) -> None:
        errors = self._client.insert_rows_json(
            self._table(RUNS_TABLE), [{**data, "run_date": run_date}]
        )
        if errors:
            raise RuntimeError(f"BigQuery run insert failed: {errors[:3]}")

    # -- reads --------------------------------------------------------------

    def similar_past(
        self, run_id: str, run_date: str, top_k: int = DEFAULT_TOP_K,
        max_distance: float = MAX_NEIGHBOUR_DISTANCE,
    ) -> dict[str, list[dict[str, Any]]]:
        """Nearest historical alerts for every alert in this run.

        One VECTOR_SEARCH over the whole run. The base side is every alert from
        a PREVIOUS run (`run_date < @run_date`), joined to its triage verdict so
        the model sees not just "we have seen this before" but "we saw it and
        called it low three days running" -- which is the fact that should stop
        it being re-reported today.
        """
        from google.cloud import bigquery

        # VECTOR_SEARCH's base side has to be a real table, so today's rows --
        # which live in the same table, having been inserted a moment ago --
        # are excluded afterwards, in the WHERE. Every alert is its own nearest
        # neighbour at distance 0, so top_k is padded to leave room for that
        # self-match plus a few same-run siblings before the real history
        # starts; the list is trimmed back to top_k in Python.
        sql = f"""
        WITH neighbours AS (
          SELECT
            query.alert_id AS alert_id,
            base.alert_id  AS past_alert_id,
            base.title     AS title,
            base.project   AS project,
            CAST(base.run_date AS STRING) AS run_date,
            distance
          FROM VECTOR_SEARCH(
            TABLE `{self._table(ALERTS_TABLE)}`, 'embedding',
            (SELECT alert_id, embedding
             FROM `{self._table(ALERTS_TABLE)}`
             WHERE run_id = @run_id AND ARRAY_LENGTH(embedding) > 0),
            'embedding',
            top_k => @padded_k, distance_type => 'COSINE')
          WHERE base.run_date < @run_date AND distance <= @max_distance
        )
        SELECT n.*, t.priority AS priority, t.decision AS decision
        FROM neighbours n
        LEFT JOIN `{self._table(TRIAGE_TABLE)}` t
          ON t.alert_id = n.past_alert_id AND CAST(t.run_date AS STRING) = n.run_date
        ORDER BY alert_id, distance
        """
        params = [
            bigquery.ScalarQueryParameter("run_date", "DATE", run_date),
            bigquery.ScalarQueryParameter("run_id", "STRING", run_id),
            bigquery.ScalarQueryParameter("padded_k", "INT64", top_k + SELF_MATCH_PADDING),
            bigquery.ScalarQueryParameter("max_distance", "FLOAT64", max_distance),
        ]
        out: dict[str, list[dict[str, Any]]] = {}
        for row in self._query(sql, params):
            neighbours = out.setdefault(row["alert_id"], [])
            if len(neighbours) < top_k:
                neighbours.append(_neighbour(row))
        logger.info("bq: vector search returned neighbours for %d alerts", len(out))
        return out

    def similar_within_run(
        self, run_id: str, top_k: int = DEFAULT_SIBLING_K,
        max_distance: float = MAX_SIBLING_DISTANCE,
    ) -> dict[str, list[dict[str, Any]]]:
        """Nearest neighbours among the alerts of THIS run.

        `similar_past` deliberately looks only at previous runs -- it has to,
        or every alert is its own perfect match. That leaves a gap this fills:
        Sentry groups events into issues by a fingerprint derived from the
        stack, so the same underlying failure arrives as two or three separate
        issues whenever the stack shape differs (a wrapped exception, a second
        code path, a rebuilt JS bundle). Measured on a real prodtools run, two
        issues 0.027 apart took slots #1 and #4 of a ten-slot report.

        This is EVIDENCE, not a merge. The selection rules ask the model to
        prefer breadth when two candidates are clearly the same failure; until
        now it had to infer that from the titles alone. Nothing is dropped
        here: both alerts keep their audit row either way.
        """
        from google.cloud import bigquery

        sql = f"""
        SELECT
          query.alert_id AS alert_id,
          base.alert_id  AS past_alert_id,
          base.title     AS title,
          base.project   AS project,
          CAST(base.run_date AS STRING) AS run_date,
          distance
        FROM VECTOR_SEARCH(
          TABLE `{self._table(ALERTS_TABLE)}`, 'embedding',
          (SELECT alert_id, embedding
           FROM `{self._table(ALERTS_TABLE)}`
           WHERE run_id = @run_id AND ARRAY_LENGTH(embedding) > 0),
          'embedding',
          top_k => @padded_k, distance_type => 'COSINE')
        WHERE base.run_id = @run_id
          AND base.alert_id != query.alert_id
          AND distance <= @max_distance
        ORDER BY alert_id, distance
        """
        params = [
            bigquery.ScalarQueryParameter("run_id", "STRING", run_id),
            bigquery.ScalarQueryParameter("padded_k", "INT64", top_k + SELF_MATCH_PADDING),
            bigquery.ScalarQueryParameter("max_distance", "FLOAT64", max_distance),
        ]
        out: dict[str, list[dict[str, Any]]] = {}
        for row in self._query(sql, params):
            siblings = out.setdefault(row["alert_id"], [])
            if len(siblings) < top_k:
                siblings.append(_sibling(row))
        logger.info("bq: %d alerts have a same-run sibling", len(out))
        return out

    def get_run(self, run_date: str) -> dict[str, Any] | None:
        from google.cloud import bigquery

        sql = f"""
        SELECT * FROM `{self._table(RUNS_TABLE)}`
        WHERE run_date = @run_date
        ORDER BY completed_at DESC LIMIT 1
        """
        rows = self._query(sql, [bigquery.ScalarQueryParameter("run_date", "DATE", run_date)])
        return _decode_stats(rows[0]) if rows else None

    def recent_runs(self, before_date: str, limit: int = 7) -> list[dict[str, Any]]:
        from google.cloud import bigquery

        sql = f"""
        SELECT * FROM `{self._table(RUNS_TABLE)}`
        WHERE run_date < @before_date
        ORDER BY run_date DESC LIMIT @limit
        """
        rows = self._query(
            sql,
            [
                bigquery.ScalarQueryParameter("before_date", "DATE", before_date),
                bigquery.ScalarQueryParameter("limit", "INT64", limit),
            ],
        )
        return [_decode_stats(r) for r in rows]


def _sibling(row: dict[str, Any]) -> dict[str, Any]:
    """A same-run neighbour. Deliberately thinner than `_neighbour`: there is no
    verdict to report yet (nothing in this run has been triaged), and this shape
    rides on every alert's payload, so every field costs tokens N times over."""
    return {
        "alert_id": row.get("past_alert_id", ""),
        "title": row.get("title", ""),
        "project": row.get("project", ""),
        "distance": round(float(row.get("distance", 1.0)), 4),
    }


def _neighbour(row: dict[str, Any]) -> dict[str, Any]:
    """The compact neighbour shape the prompt sees. Deliberately small: this
    goes into the payload for every alert, so a verbose shape multiplies."""
    return {
        "alert_id": row.get("past_alert_id", ""),
        "title": row.get("title", ""),
        "project": row.get("project", ""),
        "last_seen_run": row.get("run_date", ""),
        "past_priority": row.get("priority") or "not triaged",
        "past_decision": row.get("decision") or "not triaged",
        "distance": round(float(row.get("distance", 1.0)), 4),
    }


def _decode_stats(row: dict[str, Any]) -> dict[str, Any]:
    """`stats` is a JSON string column, not a RECORD -- RunStats gains fields
    and a schema migration per field is not worth it for an MVP. The report's
    history reader wants a dict."""
    stats = row.get("stats")
    if isinstance(stats, str):
        try:
            row = {**row, "stats": json.loads(stats)}
        except json.JSONDecodeError:
            row = {**row, "stats": {}}
    return row


class LocalBQStore:
    """JSON-file twin of BigQueryStore for offline runs and tests."""

    def __init__(self, directory: str = ".local_state"):
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _load(self, name: str) -> list[dict[str, Any]]:
        path = self._dir / f"{name}.json"
        if not path.exists():
            return []
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)

    def _append(self, name: str, rows: list[dict[str, Any]]) -> None:
        with self._lock:
            existing = self._load(name)
            existing.extend(rows)
            self._dir.mkdir(parents=True, exist_ok=True)
            with open(self._dir / f"{name}.json", "w", encoding="utf-8") as fh:
                json.dump(existing, fh, indent=2, default=str)

    def insert_alerts(self, run_id: str, run_date: str, rows: list[dict[str, Any]]) -> None:
        self._append(ALERTS_TABLE, rows)

    def record_triage(self, rows: list[dict[str, Any]]) -> None:
        self._append(TRIAGE_TABLE, rows)

    def record_run(self, run_date: str, data: dict[str, Any]) -> None:
        self._append(RUNS_TABLE, [{**data, "run_date": run_date}])

    def similar_past(
        self, run_id: str, run_date: str, top_k: int = DEFAULT_TOP_K,
        max_distance: float = MAX_NEIGHBOUR_DISTANCE,
    ) -> dict[str, list[dict[str, Any]]]:
        alerts = self._load(ALERTS_TABLE)
        verdicts = {(t.get("alert_id"), t.get("run_date")): t for t in self._load(TRIAGE_TABLE)}
        history = [a for a in alerts if str(a.get("run_date", "")) < run_date and a.get("embedding")]
        today = [a for a in alerts if a.get("run_id") == run_id and a.get("embedding")]

        out: dict[str, list[dict[str, Any]]] = {}
        for alert in today:
            scored = []
            for past in history:
                distance = cosine_distance(alert["embedding"], past["embedding"])
                if distance > max_distance:
                    continue
                verdict = verdicts.get((past.get("alert_id"), past.get("run_date")), {})
                scored.append(
                    _neighbour(
                        {
                            "past_alert_id": past.get("alert_id", ""),
                            "title": past.get("title", ""),
                            "project": past.get("project", ""),
                            "run_date": str(past.get("run_date", "")),
                            "priority": verdict.get("priority"),
                            "decision": verdict.get("decision"),
                            "distance": distance,
                        }
                    )
                )
            scored.sort(key=lambda n: n["distance"])
            if scored:
                out[alert["alert_id"]] = scored[:top_k]
        return out

    def similar_within_run(
        self, run_id: str, top_k: int = DEFAULT_SIBLING_K,
        max_distance: float = MAX_SIBLING_DISTANCE,
    ) -> dict[str, list[dict[str, Any]]]:
        rows = [a for a in self._load(ALERTS_TABLE)
                if a.get("run_id") == run_id and a.get("embedding")]
        out: dict[str, list[dict[str, Any]]] = {}
        for alert in rows:
            scored = []
            for other in rows:
                if other["alert_id"] == alert["alert_id"]:
                    continue
                distance = cosine_distance(alert["embedding"], other["embedding"])
                if distance > max_distance:
                    continue
                scored.append(_sibling({
                    "past_alert_id": other.get("alert_id", ""),
                    "title": other.get("title", ""),
                    "project": other.get("project", ""),
                    "distance": distance,
                }))
            scored.sort(key=lambda s: s["distance"])
            if scored:
                out[alert["alert_id"]] = scored[:top_k]
        return out

    def get_run(self, run_date: str) -> dict[str, Any] | None:
        runs = [r for r in self._load(RUNS_TABLE) if str(r.get("run_date")) == run_date]
        return _decode_stats(runs[-1]) if runs else None

    def recent_runs(self, before_date: str, limit: int = 7) -> list[dict[str, Any]]:
        runs = [r for r in self._load(RUNS_TABLE) if str(r.get("run_date", "")) < before_date]
        runs.sort(key=lambda r: str(r.get("run_date", "")), reverse=True)
        return [_decode_stats(r) for r in runs[:limit]]


def build_store(backend: str, project: str, dataset: str) -> TriageStore:
    if backend == "bigquery":
        return BigQueryStore(project, dataset)
    if backend == "local":
        return LocalBQStore()
    raise ValueError(f"unknown store backend: {backend}")
