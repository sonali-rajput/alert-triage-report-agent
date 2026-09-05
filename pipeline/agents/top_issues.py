"""The top-issues agent: dedup, rank, and pick the day's top N.

This replaced `pipeline/scoring.py`, a hand-weighted numeric sum, and the
replacement is deliberate. That score was nine weights and four band
thresholds, all absolute, all coupled: moving `users` from 2.0 to 2.5 silently
invalidated the bands, and the bands existed only to be turned back into the
words "high" and "critical" at the end. The judgement was always textual --
"a lot of users, in production, nobody caught it" -- and encoding it as
arithmetic added a calibration exercise without adding accuracy.

So the rules now live in `config/priority_matrix.yaml` as prose, the model
applies them, and it says why in words that go straight into the report.

It also makes the dedup call. Each alert arrives carrying `similar_past`, its
nearest historical neighbours from the BigQuery vector search, and the model
decides whether today's error is the same one we already triaged. That is a
judgement, not a hash comparison: a stack trace that gained a frame overnight
hashes differently and embeds almost identically.

Volume handling: alerts are chunked and the chunks run concurrently, each
returning a shortlist; the shortlists are then re-ranked in one final call.
The map/reduce is what keeps the prompt small enough to stay accurate at a few
hundred issues. Vertex's *batch prediction* API was the obvious alternative and
is the wrong tool here -- it is asynchronous with no latency guarantee, and
this run has to finish and post to Chat while the on-call engineer is still
having coffee.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from pipeline.agents.providers import INPUT_MARKER, LLMError, LLMProvider
from shared.config import priority_matrix
from shared.models import Alert, SelectedIssue

logger = logging.getLogger(__name__)

# Bump when the prompt changes shape; stored on every triage row so a change in
# behaviour can be attributed rather than guessed at.
SELECTION_PROMPT_VERSION = "3.0-top-issues"

DEFAULT_TOP_N = 10
# Alerts per concurrent call. Small enough that the model attends to every
# alert in the chunk, large enough that a few hundred issues is a handful of
# calls rather than dozens.
DEFAULT_CHUNK_SIZE = 25
# Concurrency cap: enough to keep the run short, low enough not to trip Vertex
# per-minute quotas on a dev project.
MAX_CONCURRENCY = 4
# The reduce step must be ONE call -- see _reduce_to_one_call. Its budget is
# larger than a map chunk because the shortlist is already the interesting
# alerts and the ranking has to be global.
REDUCE_MULTIPLIER = 2
# Safety valve on the tournament loop. Three halvings take 10,000 alerts down
# to a single call; anything still shrinking after that is pathological.
MAX_REDUCE_ROUNDS = 3


def build_system_prompt(matrix: dict[str, Any] | None = None) -> str:
    cfg = matrix if matrix is not None else priority_matrix()
    lines = [
        "You are the top-issues agent in an automated alert-triage pipeline for",
        "the prodtools department at Framestore. You receive a JSON array of",
        "alerts collected from Sentry in the last 24 hours. Secrets and PII are",
        "already masked; masked tokens look like [MASKED_...].",
        "",
        "Team context:",
        str(cfg.get("team_context", "")).strip(),
        "",
        "YOUR JOB",
        "",
        "1. Decide, for each alert, whether it is a REPEAT of something already",
        "   triaged (see DEDUPLICATION below).",
        "2. Rank what is left by how much it deserves a human's attention this",
        "   morning, using the rules below.",
        "3. Select the most important ones.",
        "",
        "PRIORITISATION RULES (apply in order; earlier rules outweigh later ones)",
        "",
        str(cfg.get("selection_rules", "")).strip(),
        "",
        "DEDUPLICATION",
        "",
        "Each alert carries TWO sets of neighbours, answering two different",
        "questions. Do not confuse them:",
        "",
        "  similar_past  -- the closest alerts from PREVIOUS runs, with the",
        "                   cosine `distance` (0 = identical text), when we last",
        "                   saw them, and how we triaged them then. These decide",
        "                   `is_duplicate`.",
        "  similar_today -- the closest OTHER alerts in THIS run. These NEVER",
        "                   make something a duplicate: nothing in today's list",
        "                   has been reported yet. They exist for RULE 7, so you",
        "                   can tell when two rows are one incident and spend one",
        "                   report slot instead of two. A `similar_today` entry",
        "                   may sit outside the batch you are looking at.",
        "",
        str(cfg.get("dedup_guidance", "")).strip(),
        "",
        "HOW TO READ THE NUMBERS",
        "",
        "  * user_count -- how many people are affected. The single strongest",
        "    signal.",
        "  * events_24h -- volume in the last 24 hours. NOT",
        "    event_count_all_time, which is the issue's lifetime total and is",
        "    routinely 1000x larger. A big lifetime count next to a small 24h",
        "    count means an OLD, steady error, not a big one.",
        "  * environment -- production outranks staging outranks dev.",
        "    Environment names are free text: '@prod', 'live' and 'prod' all",
        "    mean production. An empty environment means unknown, not non-prod.",
        "  * substatus -- new / regressed / escalating deserve attention;",
        "    'ongoing' has been seen before.",
        "  * is_unhandled -- nobody caught this.",
        "  * hourly_events -- the last 24 hourly buckets. A sharp rise in the",
        "    final buckets is happening right now.",
        "  * sentry_priority -- Sentry's own automatic assessment. A useful",
        "    second opinion, not an answer.",
        "  * masking_hits -- redaction rules that fired on this alert. A hit",
        "    like 'bearer-token' or 'aws-access-key' means the application is",
        "    logging credentials into its own error messages. That is a",
        "    security finding in its own right, whatever the error itself is.",
        "",
        "Trust the numbers over the text. Sentry titles are frequently useless",
        "(a real one is just 'RaiseException' with an empty culprit) while the",
        "impact numbers are always reliable.",
        "",
        "WHAT TO RETURN",
        "",
        "For EVERY alert in the input array, return exactly one object:",
        "- alert_id: copy the input alert_id EXACTLY, unchanged.",
        "- is_duplicate: true if this is a repeat of a similar_past entry that",
        "  needs no fresh attention.",
        "- duplicate_of: the alert_id of that past alert, or an empty string.",
        "- reason: one or two sentences citing the concrete signals you used.",
        "- selected: true if this belongs in today's top issues.",
        "- rank: 1 for the most important, then 2, 3, ... for each selected",
        "  alert. Use 0 for anything not selected. Never reuse a rank.",
        "",
        "Be factual. Do not invent details that are not in the alert.",
        "Return a JSON array with one object per input alert, in the same order.",
    ]
    return "\n".join(lines)


def _payload(alert: Alert) -> dict:
    """What the model sees for one alert.

    The body already contains the stack trace and breadcrumbs: they are fetched
    for every issue during ingestion rather than for a top-N chosen by a score,
    because choosing the top N is this agent's job and it should not have to
    make that choice from thinner evidence than it uses for everything else.
    """
    return {
        "alert_id": alert.fingerprint(),
        "title": alert.title,
        "body": alert.body,
        "project": alert.project,
        "environment": alert.environment,
        "level": alert.level,
        "substatus": alert.substatus,
        "is_unhandled": alert.is_unhandled,
        "user_count": alert.user_count,
        "events_24h": alert.events_24h(),
        "event_count_all_time": alert.event_count,
        "hourly_events": [count for _ts, count in alert.hourly_counts],
        "sentry_priority": alert.sentry_priority,
        "platform": alert.platform,
        "masking_hits": alert.masking_hits,
        "last_release": alert.last_release,
        "first_seen": alert.first_seen,
        "similar_past": alert.similar_past,
        "similar_today": alert.similar_today,
    }


def _call(provider: LLMProvider, system: str, instruction: str, batch: list[Alert]) -> list[SelectedIssue]:
    payload = json.dumps([_payload(a) for a in batch], default=str)
    prompt = f"{instruction}\n{INPUT_MARKER}\n{payload}"
    return provider.generate_list(system, prompt, SelectedIssue)


def _shortlist(
    alerts: list[Alert], provider: LLMProvider, system: str, chunk_size: int, per_chunk: int
) -> tuple[list[Alert], dict[str, SelectedIssue]]:
    """Map step: run the chunks concurrently and keep what each one selected.

    A chunk that fails is not fatal and is not silently dropped either -- its
    alerts go through to the reduce step unranked, so the worst case is that
    the final call sees a longer shortlist, not that a critical issue vanishes
    because one call timed out.
    """
    chunks = [alerts[i : i + chunk_size] for i in range(0, len(alerts), chunk_size)]
    instruction = (
        f"Select the {per_chunk} alerts in this chunk that most deserve attention, "
        "and mark every repeat."
    )

    def run(chunk: list[Alert]) -> tuple[list[Alert], list[SelectedIssue]]:
        try:
            return chunk, _call(provider, system, instruction, chunk)
        except LLMError as exc:
            logger.error("top-issues chunk of %d failed; passing it through: %s", len(chunk), exc)
            return chunk, []

    verdicts: dict[str, SelectedIssue] = {}
    survivors: list[Alert] = []
    with ThreadPoolExecutor(max_workers=min(MAX_CONCURRENCY, len(chunks))) as pool:
        for chunk, outputs in pool.map(run, chunks):
            if not outputs:
                survivors.extend(chunk)
                continue
            by_id = {o.alert_id: o for o in outputs}
            verdicts.update(by_id)
            for alert in chunk:
                verdict = by_id.get(alert.fingerprint())
                # No verdict means the model omitted it. Omission is not a
                # judgement, so it survives to the reduce step rather than
                # being dropped on the model's silence.
                if verdict is None or (not verdict.is_duplicate and verdict.selected):
                    survivors.append(alert)

    logger.info(
        "top-issues map: %d alerts in %d chunks -> %d shortlisted",
        len(alerts), len(chunks), len(survivors),
    )
    return survivors, verdicts


def _reduce_to_one_call(
    candidates: list[Alert],
    provider: LLMProvider,
    system: str,
    top_n: int,
    reduce_size: int,
    verdicts: dict[str, SelectedIssue],
) -> list[Alert]:
    """Shrink the shortlist until it fits in a single ranking call.

    This exists because of a real failure mode, not a hypothetical one. The
    final ranking has to be ONE call: two calls each return their own "rank 1",
    and sorting the union interleaves two independent rankings, so the report's
    "#1..#10" stops meaning anything. That only happens on a big org -- roughly
    250+ issues, where the map step's shortlist outgrows a single prompt --
    which is to say it degrades silently in exactly the situation where the
    ranking matters most.

    So instead of splitting the reduce, run more rounds of the same map step
    until the survivors fit. Every round narrows against the same rules, and
    every alert it drops still gets its verdict recorded.
    """
    rounds = 0
    while len(candidates) > reduce_size and rounds < MAX_REDUCE_ROUNDS:
        rounds += 1
        survivors, round_verdicts = _shortlist(
            candidates, provider, system, reduce_size, max(3, top_n)
        )
        verdicts.update(round_verdicts)
        # No progress -- every chunk failed, or the model selected everything.
        # Another identical round would not help, and looping on it would turn
        # a bad answer into an expensive bad answer.
        if not survivors or len(survivors) >= len(candidates):
            logger.warning(
                "top-issues: reduce round %d did not narrow %d candidates; stopping",
                rounds, len(candidates),
            )
            break
        candidates = survivors
        logger.info("top-issues: reduce round %d -> %d candidates", rounds, len(candidates))

    if len(candidates) > reduce_size:
        # Truncating loses alerts from the ranking, so it is the last resort and
        # it is loud. The dropped alerts keep the verdicts they earned in the
        # rounds above, so the audit trail still explains them.
        logger.warning(
            "top-issues: %d candidates still exceed one ranking call; ranking the first %d",
            len(candidates), reduce_size,
        )
        candidates = candidates[:reduce_size]
    return candidates


def select_top_issues(
    alerts: list[Alert],
    provider: LLMProvider,
    top_n: int = DEFAULT_TOP_N,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> tuple[list[Alert], dict[str, SelectedIssue]]:
    """Return (selected alerts in rank order, verdict for every input alert).

    The second element covers every alert, selected or not, because the audit
    trail records what happened to all of them -- "why was this NOT in today's
    report" is the question the trail most often has to answer.
    """
    if not alerts:
        return [], {}

    system = build_system_prompt()
    verdicts: dict[str, SelectedIssue] = {}
    candidates = alerts

    # Map step only when there is more than one chunk's worth. Below that the
    # single call below is already the whole job, and a map/reduce would pay
    # for two passes over the same alerts.
    if len(alerts) > chunk_size:
        # Each chunk shortlists a little more than its fair share of the final
        # top_n, so the reduce step has real choices to make rather than
        # rubber-stamping a list that is already exactly top_n long.
        per_chunk = max(3, top_n // 2)
        candidates, verdicts = _shortlist(alerts, provider, system, chunk_size, per_chunk)
        if not candidates:
            logger.warning("top-issues: nothing survived the shortlist; falling back to all alerts")
            candidates = alerts

    # The ranking is a single call, always. See _reduce_to_one_call.
    reduce_size = chunk_size * REDUCE_MULTIPLIER
    candidates = _reduce_to_one_call(candidates, provider, system, top_n, reduce_size, verdicts)

    instruction = (
        f"These are the candidate alerts for today. Select the {top_n} most important "
        f"and rank them 1..{top_n}. Mark every repeat."
    )
    final = _call(provider, system, instruction, candidates)
    verdicts.update({o.alert_id: o for o in final})

    by_id = {a.fingerprint(): a for a in alerts}
    # Drop verdicts for ids we never sent. A model that invents or garbles an
    # alert_id would otherwise land a row in the audit trail describing an
    # alert that does not exist, and -- because duplicate_count() reads this
    # dict -- inflate the "deduped" number in the report's funnel.
    unknown = set(verdicts) - set(by_id)
    if unknown:
        logger.warning("top-issues: discarding %d verdict(s) for unknown alert ids", len(unknown))
        verdicts = {k: v for k, v in verdicts.items() if k in by_id}

    selected = sorted(
        (o for o in final if o.selected and not o.is_duplicate and o.alert_id in by_id),
        # rank 0 on a selected alert means the model forgot to number it; sort
        # those last rather than letting a 0 jump the queue ahead of rank 1.
        key=lambda o: (o.rank if o.rank > 0 else 10**6),
    )[:top_n]

    out: list[Alert] = []
    for position, verdict in enumerate(selected, start=1):
        alert = by_id[verdict.alert_id]
        out.append(
            alert.model_copy(
                update={
                    "selection_reason": verdict.reason,
                    "is_duplicate": False,
                    "duplicate_of": "",
                }
            )
        )
        # Renumber densely. The model's own ranks can have gaps or repeats and
        # the report prints these as "#1..#10", which has to mean position.
        verdict.rank = position

    logger.info("top-issues: selected %d of %d alerts", len(out), len(alerts))
    return out, verdicts


def duplicate_count(verdicts: dict[str, SelectedIssue]) -> int:
    return sum(1 for v in verdicts.values() if v.is_duplicate)
