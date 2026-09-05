#!/usr/bin/env python
"""Run the triage pipeline over a captured Sentry fixture and show its working.

Purpose: check the pipeline's *judgement* on real-shaped data, which the unit
tests cannot do -- they assert behaviour on hand-built alerts, and the golden
dataset carries none of the impact signals the agents' prompts depend on.

    # 1. deterministic stages only -- no LLM, no network, no cost
    python scripts/triage_fixture.py --stages

    # 2. exactly what would be sent to each agent, and how many tokens of it
    python scripts/triage_fixture.py --dry-run

    # 3. the real model, via an AI Studio key (Gemini Developer API)
    GEMINI_API_KEY=... python scripts/triage_fixture.py --provider gemini

    # 4. the whole orchestrator, into ./artifacts and a scratch state dir
    python scripts/triage_fixture.py --provider gemini --full-run

`--provider mock` (the default) uses the offline heuristic, so runs 1, 2 and 4
need no key and no network. Offline runs use the HashEmbedder and a local JSON
store in place of Vertex embeddings and BigQuery: the vector-search path is
real, the vectors are lexical rather than semantic.

Read three things in the output:

  * `rank` vs `model`  -- the top-issues agent ranking something into the top 3
    and the triage agent then calling it low is the only accuracy signal this
    pipeline produces about itself. Marked `!`.
  * `decision`         -- `ignore` on something that matters is the expensive
    mistake; the reasoning line has to justify it.
  * `dup`              -- a duplicate call suppresses an alert from the report
    entirely, so a wrong one is invisible by construction. Check each.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pipeline.agents.providers import LLMError, build_provider  # noqa: E402
from pipeline.agents.top_issues import _payload as selection_payload  # noqa: E402
from pipeline.agents.top_issues import (  # noqa: E402
    build_system_prompt as selection_prompt,
)
from pipeline.agents.top_issues import select_top_issues  # noqa: E402
from pipeline.agents.triage import _payload as triage_payload  # noqa: E402
from pipeline.agents.triage import build_system_prompt as triage_prompt  # noqa: E402
from pipeline.agents.triage import triage_alerts  # noqa: E402
from pipeline.bq import LocalBQStore  # noqa: E402
from pipeline.embeddings import HashEmbedder, cosine_distance  # noqa: E402
from pipeline.event_text import event_to_body_text  # noqa: E402
from pipeline.masking import Masker  # noqa: E402
from pipeline.prefilter import Prefilter  # noqa: E402
from pipeline.sentry_client import FixtureSentryClient, issue_to_alert  # noqa: E402
from shared.models import Alert  # noqa: E402

DEFAULT_FIXTURE = "fixtures/sentry_org_issues_last_24h_anonymized.json"


class CaptureSource:
    """A real `.sentry-capture/` directory, replayed offline.

    Unlike the fixture client -- which applies ONE event to every issue -- this
    pairs each issue with ITS OWN captured `events/latest/` payload, exactly as
    the live client folds detail in. That difference matters for anything that
    reads the body: with one shared stack trace the ranking stage sees N
    identical alerts and the embedding stage collapses the whole run.
    """

    def __init__(self, directory: str):
        self._dir = pathlib.Path(directory)

    def fetch_issues(self, window_hours: int) -> list[Alert]:
        issues = json.loads((self._dir / "issues_all_environments.json").read_text(encoding="utf-8"))
        events = json.loads((self._dir / "events_latest.json").read_text(encoding="utf-8"))
        env_path = self._dir / "issue_ids_per_environment.json"
        env_of: dict[str, str] = {}
        if env_path.exists():
            for env, ids in json.loads(env_path.read_text(encoding="utf-8")).items():
                for issue_id in ids:
                    env_of.setdefault(str(issue_id), env)

        alerts, detailed = [], 0
        for issue in issues:
            issue_id = str(issue["id"])
            alert = issue_to_alert(issue, env_of.get(issue_id, ""))
            event = events.get(issue_id)
            if isinstance(event, dict) and (extra := event_to_body_text(event)):
                alert = alert.model_copy(update={"body": f"{alert.body}\n\n{extra}"})
                detailed += 1
            alerts.append(alert)
        print(f"capture: {len(alerts)} real issues, {detailed} with their own real event detail")
        return alerts

    def close(self) -> None:
        pass


def load_env_file(path: str = ".env") -> None:
    """Read KEY=value lines into the environment if they are not already set.

    The scripts read os.environ, while `python -m pipeline.main` reads `.env`
    through pydantic-settings. Without this, putting a key in `.env` works for
    one and silently not the other.

    An exported variable WINS over `.env`, matching pydantic-settings and every
    other tool -- but it says so. A stale key exported months ago in a shell
    profile, silently shadowing the one you just wrote to `.env`, produces a
    401 that looks like a bad `.env` and is genuinely hard to see.
    """
    env = pathlib.Path(path)
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("\"'")
        existing = os.environ.get(key)
        if existing is not None and existing != value:
            print(
                f"NOTE: {key} is exported in your shell and differs from .env — the "
                f"exported one wins.\n      Run `unset {key}` (and check your shell "
                "profile) if .env is the one you meant.",
                file=sys.stderr,
            )
        os.environ.setdefault(key, value)


def load_alerts(fixture: str, environment: str, detail: str = "") -> tuple[list[Alert], int, int]:
    """fixture -> mask -> prefilter. Returns (kept, ingested, dropped)."""
    raw = FixtureSentryClient(fixture, environment, detail).fetch_issues(24)
    masked = Masker().mask_alerts(raw)
    kept, dropped = Prefilter().apply(masked)
    return kept, len(raw), dropped


def print_stages(alerts: list[Alert], ingested: int, dropped: int) -> None:
    print(f"\ningested {ingested} -> prefilter dropped {dropped} -> {len(alerts)} to the agents\n")
    header = (
        f"{'short_id':<15}{'project':<18}{'env':<12}{'users':>6}{'ev24h':>7}"
        f"{'lifetime':>10}{'unh':>5}{'substatus':>12}  title"
    )
    print(header)
    print("-" * len(header))
    for a in sorted(alerts, key=lambda a: (a.user_count, a.events_24h()), reverse=True):
        print(
            f"{a.short_id:<15}{a.project[:17]:<18}{(a.environment or '-'):<12}"
            f"{a.user_count:>6}{a.events_24h():>7}{a.event_count:>10}"
            f"{('yes' if a.is_unhandled else '-'):>5}{(a.substatus or '-'):>12}  {a.title[:50]}"
        )
    print_similarity(alerts)


def print_similarity(alerts: list[Alert]) -> None:
    """The closest pairs *within* this run, using the offline embedder.

    Not the dedup the pipeline performs -- that compares against history -- but
    the same distance metric over the same text, so it answers the question you
    actually want answered before trusting the dedup: does this embedding put
    the things you would call the same error close together, and everything
    else far apart?
    """
    if len(alerts) < 2:
        return
    vectors = HashEmbedder().embed([a.embedding_text() for a in alerts])
    pairs = [
        (cosine_distance(vectors[i], vectors[j]), alerts[i], alerts[j])
        for i in range(len(alerts))
        for j in range(i + 1, len(alerts))
    ]
    pairs.sort(key=lambda p: p[0])

    print("\nclosest pairs in this run (cosine distance, 0 = identical text)")
    for distance, a, b in pairs[:8]:
        print(f"  {distance:.3f}  {a.short_id:<14} {b.short_id:<14} {a.title[:32]} | {b.title[:32]}")


def print_dry_run(alerts: list[Alert], chunk_size: int, top_n: int) -> None:
    blocks: list[tuple[str, str, str]] = []

    system = selection_prompt()
    chunks = [alerts[i : i + chunk_size] for i in range(0, len(alerts), chunk_size)]
    for n, chunk in enumerate(chunks, 1):
        payload = json.dumps([selection_payload(a) for a in chunk], default=str)
        blocks.append(("TOP-ISSUES", f"chunk {n}/{len(chunks)}", payload))

    # The triage prompt is only ever shown for a stand-in top N -- which alerts
    # actually reach it is the model's call, and this path makes none.
    triage_system = triage_prompt()
    stand_in = alerts[:top_n]
    blocks.append(
        ("TRIAGE", f"first {len(stand_in)} alerts as a stand-in for the selection",
         json.dumps([triage_payload(a) for a in stand_in], default=str))
    )

    print("\n" + "=" * 72)
    print("TOP-ISSUES SYSTEM PROMPT")
    print("=" * 72)
    print(system)
    print("\n" + "=" * 72)
    print("TRIAGE SYSTEM PROMPT")
    print("=" * 72)
    print(triage_system)
    for agent, label, payload in blocks:
        print("\n" + "=" * 72)
        print(f"{agent} PAYLOAD -- {label} ({len(payload)} chars)")
        print("=" * 72)
        print(json.dumps(json.loads(payload), indent=2))

    # ~4 chars/token is the usual rough English ratio; good enough to compare a
    # prompt change against the numbers in FINDINGS.md, not a bill.
    total = len(system) * len(chunks) + len(triage_system) + sum(len(p) for _a, _l, p in blocks)
    print(
        f"\n{len(chunks)} selection call(s) + 1 triage call, ~{total // 4} prompt tokens total "
        "(each system prompt is re-sent per call)"
    )


def print_verdicts(selected: list[Alert], verdicts: dict, outputs: list, elapsed: float) -> None:
    by_id = {o.alert_id: o for o in outputs}

    duplicates = [v for v in verdicts.values() if v.is_duplicate]
    print(f"\n{len(outputs)} verdicts in {elapsed:.1f}s")
    print(f"selected {len(selected)} of {len(verdicts)} assessed, {len(duplicates)} judged duplicates\n")

    header = f"{'#':>3}  {'short_id':<15}{'model':<9}{'decision':<9}{'sec':<5}clean_title"
    print(header)
    print("-" * len(header))
    disagreements = 0
    for rank, alert in enumerate(selected, 1):
        out = by_id.get(alert.fingerprint())
        if out is None:
            continue
        clash = rank <= 3 and out.priority.value == "low"
        disagreements += clash
        print(
            f"{rank:>3}  {alert.short_id:<15}{out.priority.value + ('!' if clash else ''):<9}"
            f"{out.decision.value:<9}{('yes' if out.security_relevant else '-'):<5}"
            f"{out.clean_title[:60]}"
        )

    print(f"\npriority: {dict(Counter(o.priority.value for o in outputs))}")
    print(f"decision: {dict(Counter(o.decision.value for o in outputs))}")
    print(f"security_relevant: {sum(1 for o in outputs if o.security_relevant)}/{len(outputs)}")
    print(f"selection-vs-triage disagreements (top 3 called low, marked !): {disagreements}")

    if duplicates:
        print(f"\nDUPLICATES ({len(duplicates)}) -- each one is an alert nobody will see:")
        for verdict in duplicates:
            print(f"  {verdict.alert_id[:12]} -> {verdict.duplicate_of[:12]}: {verdict.reason}")

    ignored = [a for a in selected if (o := by_id.get(a.fingerprint())) and o.decision.value == "ignore"]
    if ignored:
        print(f"\nIGNORED ({len(ignored)}) -- check each one is genuinely noise:")
        for alert in ignored:
            print(f"  {alert.short_id}: {by_id[alert.fingerprint()].reasoning}")

    print("\ntop 3 -- why they were selected, and where a human goes next:")
    for alert in selected[:3]:
        print(f"  {alert.short_id}: {alert.selection_reason}")
        print(f"    -> {alert.url or '(no permalink)'}")


def full_run(args, environment: str) -> None:
    """The real orchestrator, so the vector store, the report and the audit
    trail are exercised too. Uses a scratch state dir so it never collides with
    .local_state -- and note that a second run against the SAME state dir is
    the only way to see the dedup path do anything, since the first has no
    history to compare against."""
    import os

    from pipeline.orchestrator import execute_run
    from pipeline.storage import ArtifactStore

    state_dir = args.state_dir or ".local_state_fixture"
    # Honour CHAT_WEBHOOK_URL so the notify stage is exercised too -- point it
    # at scripts/fake_chat_webhook.py to see the card without a Workspace
    # account. Empty (the default) skips the post and logs that it did.
    webhook = args.chat_webhook or os.environ.get("CHAT_WEBHOOK_URL", "")
    result = execute_run(
        args.run_date,
        sentry=(CaptureSource(args.capture_dir) if args.capture_dir
                else FixtureSentryClient(args.fixture, environment, args.detail_fixture)),
        store=LocalBQStore(state_dir),
        provider=build_provider(args.provider, "", "", args.model, args.api_key),
        embedder=HashEmbedder(),
        artifacts=ArtifactStore(local_dir=args.artifacts_dir),
        chat_webhook_url=webhook,
        top_n=args.top_n,
        selection_chunk_size=args.chunk_size,
        force=True,
    )
    print(f"\nrun_id      {result.run_id}")
    print(f"degraded    {result.degraded}")
    print(f"stats       {result.stats.model_dump()}")
    print(f"noise_ratio {result.stats.noise_ratio:.2f}")
    print(f"report      {result.pdf_url or '(none rendered)'}")
    print(f"state       {state_dir}/")
    if result.degraded:
        print("\nDEGRADED: no report and no results. With CHAT_WEBHOOK_URL unset this is")
        print("near-silent in a real run -- FINDINGS.md #4.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fixture", default=DEFAULT_FIXTURE)
    parser.add_argument(
        "--capture-dir",
        default="",
        help="run against a real .sentry-capture/ directory instead of a fixture, pairing "
        "each issue with its own captured event",
    )
    parser.add_argument(
        "--detail-fixture",
        default="",
        help="optional raw events/latest/ payload, applied to every issue so the "
        "offline bodies carry a stack trace and breadcrumbs like the live path",
    )
    parser.add_argument(
        "--environment",
        default="production",
        help="stamped onto every alert, as the live client stamps the env it queried "
        "(a real API response carries no environment of its own)",
    )
    parser.add_argument("--provider", default="mock", choices=["mock", "gemini", "vertex"])
    parser.add_argument("--model", default="gemini-3.5-flash", help="2.5-flash is retired on the Developer API")
    parser.add_argument("--api-key", default="", help="defaults to $GEMINI_API_KEY")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--chunk-size", type=int, default=25, help="alerts per top-issues call")
    parser.add_argument("--stages", action="store_true", help="deterministic stages only; no LLM")
    parser.add_argument("--dry-run", action="store_true", help="print the exact prompts and payloads, call nothing")
    parser.add_argument("--full-run", action="store_true", help="drive the whole orchestrator, incl. the report")
    parser.add_argument(
        "--run-date",
        default=datetime.now(UTC).strftime("%Y-%m-%d"),
        help="default: today (UTC), matching the deployed job. run_date is not just a "
        "label -- it is the dedup partition key and the idempotency key, so a stale "
        "one orders history wrongly. Pass 2026-08-11 to line up with the bundled "
        "fixture's last-seen day.",
    )
    parser.add_argument("--state-dir", default="")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument(
        "--chat-webhook",
        default="",
        help="--full-run only; defaults to $CHAT_WEBHOOK_URL. Empty skips the Chat post.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    load_env_file()
    if not args.api_key:
        args.api_key = os.environ.get("GEMINI_API_KEY", "")
    if args.provider == "gemini" and not args.api_key:
        print("--provider gemini needs GEMINI_API_KEY (or --api-key)", file=sys.stderr)
        return 2

    if args.full_run:
        full_run(args, args.environment)
        return 0

    if args.capture_dir:
        raw = CaptureSource(args.capture_dir).fetch_issues(24)
        kept, dropped = Prefilter().apply(Masker().mask_alerts(raw))
        alerts, ingested = kept, len(raw)
    else:
        alerts, ingested, dropped = load_alerts(args.fixture, args.environment, args.detail_fixture)
    print_stages(alerts, ingested, dropped)

    if args.stages:
        return 0
    if args.dry_run:
        print_dry_run(alerts, args.chunk_size, args.top_n)
        return 0

    provider = build_provider(args.provider, "", "", args.model, args.api_key)
    started = time.monotonic()
    try:
        selected, verdicts = select_top_issues(alerts, provider, args.top_n, args.chunk_size)
        outputs = triage_alerts(selected, provider)
    except LLMError as exc:
        # The same failure that sends the orchestrator down the degraded path.
        print(f"\nLLM stage failed: {exc}", file=sys.stderr)
        return 1
    print_verdicts(selected, verdicts, outputs, time.monotonic() - started)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
