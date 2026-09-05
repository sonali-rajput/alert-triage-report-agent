"""The deterministic stages, evaluated on real data.

`invariants`, `groundedness` and `stability` all judge the two LLM agents. They
are the interesting part, but they are not the pipeline: by the time an agent
sees an alert it has been projected, detailed, masked, prefiltered and embedded,
and any of those can be quietly wrong in a way no agent check would notice.

A masking rule that stops firing does not make the model worse — it makes the
model's input contain a credential. A projection that starts leaking `seenBy`
does not change a single priority. An embedder that returns near-identical
vectors for everything makes dedup collapse the whole run, and every agent check
still passes.

So this suite runs the same stages the orchestrator runs, in the same order, on
the same fixture, and asserts the properties each stage exists to provide. It
needs no model, no labels and no network — which means it can run on every
commit, unlike anything that calls Gemini.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from eval.harness import Outcome, Report
from pipeline.agents.top_issues import _payload as selection_payload
from pipeline.embeddings import EMBEDDING_DIM, HashEmbedder, cosine_distance
from pipeline.masking import Masker
from pipeline.prefilter import Prefilter
from pipeline.sentry_client import PII_BEARING_FIELDS, FixtureSentryClient

# Shapes that must not survive masking. Deliberately independent of
# config/masking_patterns.yaml: a check that reuses the rules it is checking
# would pass by construction the moment a rule is deleted.
LEAK_SHAPES = {
    "email address": r"[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}",
    "IPv4 address": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
    "home directory (a username)": r"(?i)(?:/home/|/Users/)[^/\s\[\]]+",
    "bearer token": r"(?i)bearer\s+[a-z0-9._\-]{8,}",
    "JWT": r"\beyJ[A-Za-z0-9_\-]{8,}\.",
    "connection string password": r"[a-z][a-z0-9+.\-]*://[^:/\s@\"]+:[^@/\s\"]+@",
    "private key": r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
}

# Signals the prompts explicitly tell the agents to weigh. A field the rules
# reason about but the payload never carries is a rule that silently does
# nothing, and it looks exactly like a working rule from the outside.
REQUIRED_SIGNALS = (
    "user_count", "events_24h", "event_count_all_time", "environment",
    "substatus", "is_unhandled", "hourly_events", "masking_hits",
    "similar_past", "similar_today",
)


def _leaks(text: str) -> dict[str, int]:
    found: dict[str, int] = {}
    for label, pattern in LEAK_SHAPES.items():
        hits = [m for m in re.findall(pattern, text) if "MASKED" not in str(m)]
        if hits:
            found[label] = len(hits)
    return found


def run(fixture: str, detail_fixture: str = "", sample: int = 0) -> Report:
    report = Report("PIPELINE STAGES (deterministic, no model needed)")
    path = Path(fixture)
    if not path.exists():
        report.add(Outcome("fixture", False, f"{fixture} not found", errored=True))
        return report

    raw_issues = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw_issues, dict):
        raw_issues = [raw_issues]

    # -- Stage 1: fetch + projection ---------------------------------------
    alerts = FixtureSentryClient(fixture, "production", detail_fixture).fetch_issues(24)
    if sample:
        alerts = alerts[:sample]

    projected_count = len(FixtureSentryClient(fixture, "production").fetch_issues(24))
    report.add(Outcome(
        "1. every issue survives the projection",
        projected_count == len(raw_issues),
        f"{len(raw_issues)} issue(s) -> {projected_count} alert(s)"
        + (f" (evaluating the first {len(alerts)})" if sample else ""),
        "An issue silently lost in the projection is an alert nobody will ever see.",
    ))

    present_in_raw = {f for f in PII_BEARING_FIELDS if any(f in i for i in raw_issues)}
    projected = json.dumps([a.model_dump() for a in alerts], default=str)
    leaked_fields = {f for f in present_in_raw if f'"{f}"' in projected}
    report.add(Outcome(
        "1. the projection drops every PII-bearing field",
        not leaked_fields,
        f"{len(present_in_raw)} PII field(s) in the raw payload"
        + (f" — LEAKED: {sorted(leaked_fields)}" if leaked_fields else ", none in the projection"),
        "activity/seenBy/assignedTo carry colleague names, emails and gravatar "
        "hashes. The projection is the only thing between them and BigQuery.",
    ))

    if detail_fixture:
        with_trace = sum(1 for a in alerts if "Exception:" in a.body)
        report.add(Outcome(
            "1. event detail reaches the alert body",
            with_trace == len(alerts),
            f"{with_trace}/{len(alerts)} alerts carry a stack trace; "
            f"median body {sorted(len(a.body) for a in alerts)[len(alerts) // 2]} chars",
            "Without detail the agents rank on a ~200-char title, which is a "
            "materially easier input than production gives them.",
        ))

    # -- Stage 2: masking ---------------------------------------------------
    masker = Masker()
    masked = masker.mask_alerts(alerts)
    corpus = " ".join(f"{a.title} {a.body}" for a in masked)

    leaks = _leaks(corpus)
    report.add(Outcome(
        "2. nothing secret-shaped survives masking",
        not leaks,
        "clean" if not leaks else f"SURVIVED: {leaks}",
        "This is the last point before the text reaches Vertex AI and BigQuery. "
        "The patterns here are independent of masking_patterns.yaml on purpose: "
        "a check that reuses the rules it checks passes the moment one is deleted.",
    ))

    twice = masker.mask_alerts(masked)
    report.add(Outcome(
        "2. masking is idempotent",
        [a.body for a in masked] == [a.body for a in twice],
        "same output on a second pass",
        "A body is masked more than once (it grows when detail is folded in). A "
        "rule that rewrites its own output changes the alert's text every run, "
        "which changes its embedding and quietly breaks dedup.",
    ))

    # Compare BEFORE against AFTER. A first version flagged any short body and
    # blamed masking for `Level: error` -- an issue payload that was always
    # twelve characters long. The question is what masking REMOVED, not what
    # Sentry happened to provide.
    gutted = [
        f"{a.short_id or a.source_id}: {len(a.body)} -> {len(b.body)} chars"
        for a, b in zip(alerts, masked)
        # Purely relative: an absolute floor re-introduces the same false
        # positive from the other direction (12 -> 12 chars is not gutting).
        if a.body and len(b.body) < len(a.body) * 0.5
    ]
    report.add(Outcome(
        "2. masking does not gut the error",
        not gutted,
        "no body lost more than half its length" if not gutted else f"GUTTED: {gutted[:5]}",
        "Over-masking leaves the agents with no error to read, which is worse "
        "than a redacted one.",
    ))

    fired = Counter(h for a in masked for h in a.masking_hits)
    report.add(Outcome(
        "2. masking hits (informational)",
        True,
        str(dict(fired)) if fired else "no rules fired on this fixture",
        "",
    ))

    # -- Stage 3: prefilter -------------------------------------------------
    kept, dropped = Prefilter().apply(masked)
    report.add(Outcome(
        "3. the prefilter keeps a workable set",
        bool(kept),
        f"kept {len(kept)}, dropped {dropped}",
        "Dropping everything means the run reports nothing and looks like a "
        "quiet day.",
    ))

    if kept:
        stale = [a.short_id for a in kept if a.events_24h() == 0]
        report.add(Outcome(
            "3. nothing with zero events in the window survives",
            not stale,
            "none" if not stale else f"{len(stale)} kept with events_24h=0: {stale[:4]}",
            "`count` is the ALL-TIME total. Filtering on it keeps years-old dead "
            "issues and drops live ones -- exactly backwards.",
        ))

    # -- Stage 4: the payload the agents receive ----------------------------
    if kept:
        payload = selection_payload(kept[0])
        missing = [s for s in REQUIRED_SIGNALS if s not in payload]
        report.add(Outcome(
            "4. the payload carries every signal the rules name",
            not missing,
            "all present" if not missing else f"MISSING: {missing}",
            "A field the prompt reasons about but never receives is a rule that "
            "does nothing, and it is indistinguishable from a working one.",
        ))

    # -- Stage 5: embeddings, the premise dedup rests on --------------------
    #
    # Measured WITHOUT the detail fixture, deliberately. A fixture applies ONE
    # captured event to every issue, so every body ends with the same 2,000
    # characters -- which dominates the vector and drops the median pairwise
    # distance from 0.79 to 0.03. Every alert becomes a near-duplicate of every
    # other and dedup would collapse the run. That is an artifact of the
    # fixture, not of the pipeline: in production each issue carries its own
    # event. Measuring similarity on shared-detail bodies measures the fixture.
    embed_alerts = kept
    if detail_fixture:
        undetailed = FixtureSentryClient(fixture, "production").fetch_issues(24)
        embed_alerts, _ = Prefilter().apply(masker.mask_alerts(undetailed))
        if sample:
            embed_alerts = embed_alerts[:sample]

    if len(embed_alerts) >= 3:
        kept = embed_alerts
        vectors = HashEmbedder().embed([a.embedding_text() for a in kept])
        report.add(Outcome(
            "5. vectors match the width BigQuery expects",
            all(len(v) == EMBEDDING_DIM for v in vectors),
            f"{len(vectors)} x {EMBEDDING_DIM}",
            "VECTOR_SEARCH cannot compare vectors of different widths.",
        ))

        # key= is load-bearing: two pairs at the same distance would otherwise
        # fall through to comparing Alert objects, which are not orderable. It
        # never tripped while distances happened to be distinct.
        pairs = sorted(
            ((cosine_distance(vectors[i], vectors[j]), kept[i], kept[j])
             for i in range(len(kept)) for j in range(i + 1, len(kept))),
            key=lambda pair: pair[0],
        )
        closest, far = pairs[0], pairs[-1]
        distances = [d for d, _a, _b in pairs]
        median = sorted(distances)[len(distances) // 2]
        near_ratio = sum(1 for d in distances if d < 0.15) / len(distances)
        # An ABSOLUTE criterion, not a relative one. "closest is half the
        # furthest" passes happily when every pair sits between 0.01 and 0.04 --
        # which is exactly what a shared detail fixture produces, and exactly
        # the state in which dedup marks the whole run duplicate.
        report.add(Outcome(
            "5. the embedding separates similar from unrelated",
            median > 0.3 and near_ratio < 0.5,
            f"median pair {median:.3f}, closest {closest[0]:.3f} "
            f"({closest[1].short_id} / {closest[2].short_id}), furthest {far[0]:.3f}; "
            f"{near_ratio:.0%} of pairs under 0.15",
            "If most pairs are near-duplicates, dedup collapses the whole run and "
            "the report goes empty -- which looks like a quiet day, not a bug.",
        ))

        near = [(d, a, b) for d, a, b in pairs if d < 0.15]
        report.add(Outcome(
            "5. near-duplicate pairs Sentry filed separately (informational)",
            True,
            f"{len(near)} pair(s) under 0.15"
            + ("".join(f"\n         {d:.3f}  {a.short_id} / {b.short_id}" for d, a, b in near[:4])),
            "",
        ))

    return report
