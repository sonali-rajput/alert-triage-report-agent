"""Label-free evaluation of the two agents.

    python -m eval.run_checks                          # mock: checks the harness runs
    GEMINI_API_KEY=... python -m eval.run_checks --provider gemini --model gemini-3.5-flash
    python -m eval.run_checks --provider vertex        # the deployed model

Three suites, none of which needs a human to label anything:

  pipeline      the deterministic stages -- projection, detail, masking,
                prefilter, payload, embeddings -- on the same fixture. No model,
                no labels, no network. A masking rule that stops firing does not
                make the model worse; it puts a credential in the model's input,
                and every agent check still passes.
  invariants    properties the rules promise, checked by varying one field at a
                time and asserting the relation. Catches dropped rules, wrong
                anchoring, and dedup suppressing an escalation.
  groundedness  every figure in the agent's prose has to appear in its payload.
                Catches invented facts, which is the failure a reader cannot
                spot without opening Sentry.
  stability     the same input, run twice. Produces the ceiling on every other
                number in the suite.

Run this after ANY change to a prompt, to `config/priority_matrix.yaml`, or to
the model. It is the whole agent evaluation the MVP ships: the golden-dataset
harness was removed because it measured only the cheapest of the four ways this
pipeline can be wrong, and cost a labelling session to keep current.
EVALUATION.md has the reasoning and says when to bring it back.

Exit code is non-zero when a check fails, so this works as a CI gate -- except
with `--provider mock`, which always exits 0 because the mock is a keyword
heuristic that fails several invariants by construction. That is not a bug in
the mock: the invariants are asking whether a *model* follows the rules, and
the mock was never given any.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from eval import groundedness, invariants, pipeline_checks, stability
from pipeline.agents.providers import build_provider
from pipeline.masking import Masker
from pipeline.prefilter import Prefilter
from pipeline.sentry_client import FixtureSentryClient

DEFAULT_FIXTURE = "fixtures/sentry_org_issues_last_24h_anonymized.json"
DEFAULT_DETAIL = "fixtures/sentry_event_latest.sample.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--provider", default="mock", choices=["mock", "gemini", "vertex"])
    parser.add_argument("--model", default="gemini-3.5-flash")
    parser.add_argument("--api-key", default="", help="defaults to $GEMINI_API_KEY")
    parser.add_argument("--project", default="", help="GCP project, for --provider vertex")
    parser.add_argument("--location", default="europe-west2")
    parser.add_argument("--fixture", default=DEFAULT_FIXTURE,
                        help="real-shaped alerts for the groundedness and stability suites")
    parser.add_argument("--detail-fixture", default=DEFAULT_DETAIL,
                        help="a captured events/latest/ payload, folded into every alert's body "
                             "so the agents read production-shaped input. Without it the bodies "
                             "are issue-payload-only (~200 chars) and the eval is measuring the "
                             "model on materially easier input than it gets in production. "
                             "Pass '' to evaluate without stack traces.")
    parser.add_argument("--suite", default="all",
                        choices=["all", "pipeline", "invariants", "groundedness", "stability"])
    parser.add_argument("--stability-runs", type=int, default=2)
    parser.add_argument("--pace", type=float, default=None,
                        help="seconds between invariant checks. Defaults to 0 for the mock and "
                             "13 for a real provider, which fits the Gemini free tier's 5 "
                             "requests a minute. Raise it if you still see rate limits.")
    parser.add_argument("--sample", type=int, default=8,
                        help="alerts from the fixture to use (keeps a real-model run cheap)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    api_key = args.api_key or os.environ.get("GEMINI_API_KEY", "")
    if args.provider == "gemini" and not api_key:
        print("--provider gemini needs GEMINI_API_KEY (or --api-key)", file=sys.stderr)
        return 2
    provider = build_provider(args.provider, args.project, args.location, args.model, api_key)

    if args.provider == "mock":
        print(
            "NOTE: --provider mock. The mock is a keyword heuristic that was never given "
            "the rules,\n      so several invariants fail by construction. This run checks "
            "the HARNESS, not the\n      model, and always exits 0. Use --provider gemini or "
            "vertex for a real answer.\n"
        )

    # Real-shaped alerts for the suites that need input rather than constructing
    # their own. Masked and prefiltered exactly as the pipeline would.
    raw = FixtureSentryClient(args.fixture, "production", args.detail_fixture).fetch_issues(24)
    alerts, _dropped = Prefilter().apply(Masker().mask_alerts(raw))
    alerts = alerts[: args.sample]
    if alerts:
        median = sorted(len(a.body) for a in alerts)[len(alerts) // 2]
        detail = "with stack traces" if args.detail_fixture else "issue-payload bodies only"
        print(f"input: {len(alerts)} alert(s) from {args.fixture}")
        print(f"       {detail}, median body {median} chars\n")
    elif args.suite in ("groundedness", "stability"):
        # Evaluating nothing is not the same as passing, and the two used to
        # look identical: an empty input reported SKIP and exited 0.
        print(
            f"\nNOTHING TO EVALUATE: {args.fixture} produced {len(raw)} issue(s), and the\n"
            f"prefilter dropped all of them. The usual cause is an issue with no events in\n"
            f"the 24h window (events_24h = 0), which the prefilter drops by design.\n\n"
            f"Pick a fixture with live issues -- {DEFAULT_FIXTURE}\n"
            f"is the 19-issue real org pull -- or pass --suite invariants, which builds its\n"
            f"own alerts and needs no fixture at all.",
            file=sys.stderr,
        )
        return 2

    reports = []
    if args.suite in ("all", "pipeline"):
        # Deterministic, no model. Runs first because if masking or the
        # projection is broken, whatever the agents scored is beside the point.
        reports.append(pipeline_checks.run(args.fixture, args.detail_fixture, args.sample))

    pace = args.pace if args.pace is not None else (0.0 if args.provider == "mock" else 13.0)
    if args.suite in ("all", "invariants"):
        if pace:
            print(f"pacing {pace:g}s between checks for the free tier; "
                  f"this suite will take about {pace * len(invariants.ALL) / 60:.0f} minutes.\n")
        reports.append(invariants.run(provider, pace=pace))
    if args.suite in ("all", "groundedness"):
        reports.append(groundedness.run(alerts, provider))
    if args.suite in ("all", "stability"):
        reports.append(stability.run(alerts, provider, runs=args.stability_runs))

    failures = errors = 0
    for report in reports:
        print(report.render())
        failures += len(report.failures)
        errors += len(report.errors)

    print("\n" + "=" * 60)
    if failures:
        print(f"{failures} check(s) FAILED — the model was asked and got it wrong.")
        print("Read the detail line, decide whether the rule or the model is wrong,")
        print("and fix whichever it is.")
    if errors:
        print(f"{errors} check(s) COULD NOT RUN — no verdict either way.")
        print("Usually a rate limit (raise --pace) or a bad key. Not a claim about the model.")
    if not failures and not errors:
        print("All checks passed.")
    print("=" * 60)

    if args.provider == "mock":
        return 0
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
