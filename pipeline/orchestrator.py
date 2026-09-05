"""Core pipeline flow, independent of any web framework so it is unit-testable:

    fetch (issues + stack traces, one pass) -> mask -> prefilter
      -> embed -> write alerts to BigQuery -> vector search for history
      -> top-issues agent (dedup + rank + select 10)
      -> triage agent (one call over the 10)
      -> BigQuery audit trail -> PDF to private GCS + signed URL
      -> Google Chat card

Two agents, two BigQuery round trips, one Chat post. What is *not* here is as
much of the design as what is: no hash dedup, no numeric scoring, no Firestore,
no separate enrichment stage. Each of those was a mechanism standing in for a
judgement, and the judgement is now made where it can be explained -- by a
model, against rules written in prose, with its reasoning stored next to its
verdict in a table anyone can query.

Failure policy is unchanged in spirit. If Sentry is unreachable the run fails
loud (error to Chat, exception re-raised so the caller exits non-zero and Cloud
Scheduler retries). If the LLM stages fail permanently the run degrades to a
raw digest posted to Chat, so nothing is silently dropped.
"""

from __future__ import annotations

import logging
import uuid

from pipeline.agents.providers import LLMError, LLMProvider
from pipeline.agents.top_issues import (
    DEFAULT_TOP_N,
    SELECTION_PROMPT_VERSION,
    duplicate_count,
    select_top_issues,
)
from pipeline.agents.triage import PROMPT_VERSION, triage_alerts
from pipeline.bq import TriageStore
from pipeline.chat_notify import post_error, post_fallback_digest, post_run_summary
from pipeline.embeddings import Embedder
from pipeline.masking import Masker
from pipeline.pdf_report import render_report
from pipeline.prefilter import Prefilter
from pipeline.sentry_client import SentrySource
from pipeline.storage import ArtifactStore
from shared.models import (
    Alert,
    Priority,
    RunResult,
    RunStats,
    SelectedIssue,
    TriagedAlert,
    TriageDecision,
    utcnow,
)

logger = logging.getLogger(__name__)

# A selection/triage disagreement worth flagging: the top-issues agent called
# this one of the ten most important things today and the triage agent then
# called it low. Anything less than that is ordinary boundary noise -- rank 8
# turning out to be medium is the system working.
DISAGREEMENT_TOP_RANK = 3


def _alert_row(alert: Alert, embedding: list[float], run_id: str, run_date: str) -> dict:
    """One row of the BigQuery `alerts` table.

    This is both the audit archive of what we ingested and the vector index the
    next run searches, which is why the embedding travels with the projection
    rather than in a table of its own.
    """
    return {
        "run_id": run_id,
        "run_date": run_date,
        "alert_id": alert.fingerprint(),
        "source": alert.source.value,
        "source_id": alert.source_id,
        "short_id": alert.short_id,
        "title": alert.title,
        "body": alert.body,
        "url": alert.url,
        "project": alert.project,
        "environment": alert.environment,
        "level": alert.level,
        "substatus": alert.substatus,
        "is_unhandled": alert.is_unhandled,
        "user_count": alert.user_count,
        "events_24h": alert.events_24h(),
        "event_count_all_time": alert.event_count,
        "sentry_priority": alert.sentry_priority,
        "platform": alert.platform,
        "last_release": alert.last_release,
        "first_seen": alert.first_seen.isoformat() if alert.first_seen else None,
        "last_seen": alert.last_seen.isoformat() if alert.last_seen else None,
        "masking_hits": alert.masking_hits,
        "embedding": embedding,
        "ingested_at": utcnow().isoformat(),
    }


def _triage_row(result: TriagedAlert, verdict: SelectedIssue | None, run_id: str, run_date: str) -> dict:
    """One row of the BigQuery `triage` table.

    Every alert that reached the top-issues agent gets a row, selected or not:
    "why was this NOT in today's report" is the question the audit trail most
    often has to answer, and it is unanswerable if only the winners are stored.
    Both prompt versions are recorded so a change in behaviour can be
    attributed to a prompt change rather than guessed at.
    """
    return {
        "run_id": run_id,
        "run_date": run_date,
        "alert_id": result.alert_id,
        "title": result.title,
        "project": result.project,
        "selected": result.rank > 0,
        "rank": result.rank,
        "is_duplicate": bool(verdict.is_duplicate) if verdict else False,
        "duplicate_of": (verdict.duplicate_of if verdict else "") or "",
        "selection_reason": result.selection_reason,
        "priority": result.triage.priority.value,
        "decision": result.triage.decision.value,
        "reasoning": result.triage.reasoning,
        "summary": result.summary.summary,
        "component": result.summary.component,
        "suspected_cause": result.summary.suspected_cause,
        "security_relevant": result.security_relevant,
        "security_rationale": result.security_rationale,
        "disagreement": result.disagreement,
        "user_count": result.user_count,
        "events_24h": result.events_24h,
        "selection_prompt_version": SELECTION_PROMPT_VERSION,
        "triage_prompt_version": PROMPT_VERSION,
        "triaged_at": utcnow().isoformat(),
    }


def _unselected_row(
    alert: Alert, verdict: SelectedIssue | None, run_id: str, run_date: str
) -> dict:
    """Audit row for an alert the top-issues agent did not put in the report."""
    return {
        "run_id": run_id,
        "run_date": run_date,
        "alert_id": alert.fingerprint(),
        "title": alert.title,
        "project": alert.project,
        "selected": False,
        "rank": 0,
        "is_duplicate": bool(verdict.is_duplicate) if verdict else False,
        "duplicate_of": (verdict.duplicate_of if verdict else "") or "",
        "selection_reason": verdict.reason if verdict else "not assessed",
        "priority": "",
        "decision": "",
        "reasoning": "",
        "summary": "",
        "component": "",
        "suspected_cause": "",
        "security_relevant": False,
        "security_rationale": "",
        "disagreement": False,
        "user_count": alert.user_count,
        "events_24h": alert.events_24h(),
        "selection_prompt_version": SELECTION_PROMPT_VERSION,
        "triage_prompt_version": "",
        "triaged_at": utcnow().isoformat(),
    }


def _degraded_response(
    run_id: str,
    run_date: str,
    stats: RunStats,
    alerts: list[Alert],
    error: str,
    store: TriageStore,
    chat_webhook_url: str,
) -> RunResult:
    logger.error("LLM pipeline failed; degrading to raw digest: %s", error)
    try:
        post_fallback_digest(chat_webhook_url, run_date, alerts, error)
    except Exception:
        logger.exception("fallback digest post failed too")
    try:
        store.record_run(
            run_date,
            {
                "run_id": run_id,
                "degraded": True,
                "error": error[:500],
                "stats": stats.model_dump_json(),
                "pdf_url": "",
                "completed_at": utcnow().isoformat(),
            },
        )
    except Exception:
        logger.exception("could not record the degraded run")
    return RunResult(run_id=run_id, run_date=run_date, degraded=True, stats=stats)


def execute_run(
    run_date: str,
    *,
    sentry: SentrySource,
    store: TriageStore,
    provider: LLMProvider,
    embedder: Embedder,
    artifacts: ArtifactStore,
    masker: Masker | None = None,
    prefilter: Prefilter | None = None,
    chat_webhook_url: str = "",
    window_hours: int = 24,
    top_n: int = DEFAULT_TOP_N,
    selection_chunk_size: int = 25,
    force: bool = False,
) -> RunResult:
    # Idempotency guard. The audit trail is written before the report and Chat
    # steps, so a failure in those and the Scheduler retry that follows would
    # find today's alerts already in the vector table -- and the top-issues
    # agent would dutifully mark every one of them a duplicate of itself. A
    # completed non-degraded run for the date therefore short-circuits; `force`
    # is the deliberate-replay escape hatch.
    if not force:
        existing = store.get_run(run_date)
        if existing and not existing.get("degraded"):
            logger.info("run for %s already completed (%s); skipping", run_date, existing.get("run_id"))
            return RunResult(
                run_id=str(existing.get("run_id", "")),
                run_date=run_date,
                stats=RunStats(**(existing.get("stats") or {})),
                pdf_url=str(existing.get("pdf_url", "")),
            )

    run_id = f"{run_date}-{uuid.uuid4().hex[:8]}"
    masker = masker or Masker()
    prefilter = prefilter or Prefilter()

    # 1. Fetch. Issues AND their stack traces and breadcrumbs, in one pass.
    # A total failure is fatal for the run: notify and re-raise so the caller
    # exits non-zero and Cloud Scheduler retries.
    try:
        raw_alerts = sentry.fetch_issues(window_hours)
    except Exception as exc:
        logger.exception("run %s: Sentry fetch failed", run_id)
        try:
            post_error(chat_webhook_url, run_date, "sentry-fetch", str(exc))
            # The entrypoint's catch-all also posts an error notice for anything
            # that escapes; the marker stops this one being reported twice.
            exc._chat_notified = True  # type: ignore[attr-defined]
        except Exception:
            logger.exception("error notice post failed too")
        raise
    ingested = len(raw_alerts)
    logger.info("run %s: fetched %d Sentry issues with detail", run_id, ingested)

    # 2. Mask secrets/PII before anything is stored or sent to a model. This
    # runs after the stack traces are already in the body, which is the point:
    # frames, local variables and breadcrumb messages are exactly where
    # credentials and personal data hide.
    masked = masker.mask_alerts(raw_alerts)

    # 3. Drop known noise -- cheap, deterministic, and it never has to be
    # explained to anyone. Everything downstream costs tokens.
    alerts, prefiltered = prefilter.apply(masked)
    logger.info("run %s: %d after prefilter (%d dropped as noise)", run_id, len(alerts), prefiltered)

    # 4. Embed and store. The alerts have to be in BigQuery before the vector
    # search runs, because the search reads today's rows as its query side.
    similar: dict[str, list[dict]] = {}
    siblings: dict[str, list[dict]] = {}
    try:
        vectors = embedder.embed([a.embedding_text() for a in alerts])
        store.insert_alerts(
            run_id, run_date, [_alert_row(a, v, run_id, run_date) for a, v in zip(alerts, vectors)]
        )
        # 5. Two vector searches, answering two different questions.
        #
        #   similar_past  -- "have we reported this before, and what did we
        #                     decide?" Looks only at PREVIOUS runs.
        #   similar_today -- "is another row of today's list the same incident?"
        #                     Sentry groups events into issues by a fingerprint
        #                     derived from the stack, so one failure arrives as
        #                     two or three issues whenever the stack shape
        #                     differs. Measured on a real run, two issues 0.027
        #                     apart took slots #1 and #4 of a ten-slot report.
        #
        # A failure here costs the dedup signal, not the run: the agent sees no
        # neighbours and judges everything as new, which over-reports rather
        # than suppresses.
        similar = store.similar_past(run_id, run_date)
        siblings = store.similar_within_run(run_id)
    except Exception:
        logger.exception("run %s: embedding/vector search failed; continuing without history", run_id)

    alerts = [
        a.model_copy(update={
            "similar_past": similar.get(a.fingerprint(), []),
            "similar_today": siblings.get(a.fingerprint(), []),
        })
        for a in alerts
    ]
    with_history = sum(1 for a in alerts if a.similar_past)
    with_siblings = sum(1 for a in alerts if a.similar_today)
    logger.info(
        "run %s: %d of %d alerts have historical neighbours, %d have a same-run sibling",
        run_id, with_history, len(alerts), with_siblings,
    )

    # 6. Top-issues agent: dedup, rank, and select the day's top N.
    try:
        selected, verdicts = select_top_issues(alerts, provider, top_n, selection_chunk_size)
    except LLMError as exc:
        stats = RunStats(ingested=ingested, prefiltered=prefiltered, considered=len(alerts))
        return _degraded_response(run_id, run_date, stats, alerts, str(exc), store, chat_webhook_url)

    deduped = duplicate_count(verdicts)
    logger.info("run %s: selected %d issues, %d judged duplicates", run_id, len(selected), deduped)

    # 7. Triage agent: one call over the selected issues. No batching -- ten
    # alerts fit in one prompt, and one prompt lets the model weigh them
    # against each other.
    try:
        outputs = triage_alerts(selected, provider)
    except LLMError as exc:
        stats = RunStats(
            ingested=ingested, prefiltered=prefiltered, deduped=deduped, considered=len(alerts)
        )
        return _degraded_response(run_id, run_date, stats, selected, str(exc), store, chat_webhook_url)

    results: list[TriagedAlert] = []
    for position, (alert, output) in enumerate(zip(selected, outputs), start=1):
        results.append(
            TriagedAlert(
                alert_id=alert.fingerprint(),
                source=alert.source,
                title=alert.title,
                url=alert.url,
                project=alert.project,
                event_count=alert.event_count,
                ongoing_days=alert.ongoing_days,
                summary=output.to_summary(),
                triage=output.to_triage_result(),
                rank=position,
                selection_reason=alert.selection_reason,
                # Two independent LLM judgements pointing different ways. With
                # no human correction loop, this is still the only accuracy
                # signal the pipeline produces about itself.
                disagreement=(
                    position <= DISAGREEMENT_TOP_RANK and output.priority == Priority.low
                ),
                security_relevant=output.security_relevant,
                security_rationale=output.security_rationale,
                environment=alert.environment,
                user_count=alert.user_count,
                substatus=alert.substatus,
                is_unhandled=alert.is_unhandled,
                level=alert.level,
                sentry_priority=alert.sentry_priority,
                events_24h=alert.events_24h(),
                hourly_counts=alert.hourly_counts,
                masking_hits=alert.masking_hits,
                similar_past_count=len(alert.similar_past),
            )
        )

    disagreements = sum(1 for r in results if r.disagreement)
    if disagreements:
        logger.info("run %s: %d selection-vs-triage disagreements", run_id, disagreements)

    stats = RunStats(
        ingested=ingested,
        prefiltered=prefiltered,
        deduped=deduped,
        considered=len(alerts),
        processed=len(results),
        ignored=sum(1 for r in results if r.triage.decision == TriageDecision.ignore),
        notified=sum(1 for r in results if r.triage.decision == TriageDecision.notify),
        by_priority={
            level: sum(1 for r in results if r.triage.priority.value == level)
            for level in ["critical", "high", "medium", "low"]
        },
    )

    # 8. Audit trail: the triaged ten, plus a row for everything the top-issues
    # agent looked at and passed over.
    selected_ids = {r.alert_id for r in results}
    try:
        rows = [_triage_row(r, verdicts.get(r.alert_id), run_id, run_date) for r in results]
        rows += [
            _unselected_row(a, verdicts.get(a.fingerprint()), run_id, run_date)
            for a in alerts
            if a.fingerprint() not in selected_ids
        ]
        store.record_triage(rows)
    except Exception:
        logger.exception("run %s: audit trail write failed; continuing", run_id)

    # 9. PDF report. Past runs drive the scorecard's "vs yesterday / vs 7-day
    # average" line; a failure to read them costs the deltas, not the report.
    history: list[dict] = []
    try:
        history = store.recent_runs(run_date, limit=7)
    except Exception:
        logger.exception("could not load run history; rendering without deltas")

    pdf_url = ""
    try:
        document, extension = render_report(run_id, run_date, stats, results, history=history)
        pdf_url = artifacts.save_report(run_date, document, extension)
        logger.info("run %s: report stored (%s)", run_id, "signed URL" if pdf_url.startswith("https://") else pdf_url)
    except Exception:
        logger.exception("report generation failed; continuing without it")

    # 10. Chat notification.
    try:
        post_run_summary(chat_webhook_url, run_date, stats, results, pdf_url)
    except Exception:
        logger.exception("chat notification failed; continuing")

    # 11. Record the run.
    try:
        store.record_run(
            run_date,
            {
                "run_id": run_id,
                "degraded": False,
                "error": "",
                "stats": stats.model_dump_json(),
                "pdf_url": pdf_url,
                "completed_at": utcnow().isoformat(),
            },
        )
    except Exception:
        logger.exception("run %s: could not record the run", run_id)

    return RunResult(run_id=run_id, run_date=run_date, stats=stats, results=results, pdf_url=pdf_url)
