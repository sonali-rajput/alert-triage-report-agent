"""Cloud Run Job entrypoint: one daily triage run, then exit.

A Job rather than a Service, because this is a scheduled batch task that does
not listen for HTTP traffic. There is no request to keep alive, no request
timeout to fit a fetch of several hundred issues into, and no endpoint anyone
could invoke by accident -- Cloud Scheduler starts the job, the job runs to
completion, the container stops and stops being billed.

Exit codes are the interface: 0 means the run completed (including the degraded
digest path, which is a completed run with a worse report), non-zero means it
did not, which is what Cloud Scheduler retries on.

    python -m pipeline.main [--run-date YYYY-MM-DD] [--force]
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from datetime import UTC, datetime

from pipeline.agents.providers import LLMProvider, build_provider
from pipeline.bq import build_store
from pipeline.chat_notify import post_error
from pipeline.embeddings import build_embedder
from pipeline.masking import Masker
from pipeline.orchestrator import execute_run
from pipeline.prefilter import Prefilter
from pipeline.sentry_client import build_sentry_source
from pipeline.settings import get_settings
from pipeline.storage import ArtifactStore
from shared.logging_setup import setup_logging
from shared.models import RunResult

logger = logging.getLogger("pipeline")

RUN_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _provider() -> LLMProvider:
    s = get_settings()
    if s.llm_provider == "mock":
        # A mock in a deployed environment is a silent quality disaster: every
        # alert gets a heuristic verdict written to the audit trail as though a
        # model produced it. Make it loud.
        logger.warning("LLM provider is 'mock' -- agent output is NOT from a real model")
    elif s.llm_provider == "gemini":
        # Real model, but the Developer API is a laptop transport: no VPC-SC, no
        # org-level data controls, and the key is not in Secret Manager.
        logger.warning(
            "LLM provider is 'gemini' (Developer API key, model %s) -- real model but "
            "NOT the Vertex transport",
            s.gemini_model,
        )
    else:
        logger.info("LLM provider: vertex (model %s)", s.gemini_model)
    return build_provider(
        s.llm_provider, s.gcp_project, s.gcp_location, s.gemini_model, s.gemini_api_key
    )


def run_once(run_date: str, force: bool = False) -> RunResult:
    s = get_settings()
    sentry = build_sentry_source(
        s.sentry_source,
        s.sentry_base_url,
        s.sentry_token,
        s.sentry_org,
        s.sentry_fixture_path,
        s.sentry_environment_list,
        s.sentry_fixture_detail_path,
        s.sentry_team,
    )
    try:
        return execute_run(
            run_date,
            sentry=sentry,
            store=build_store(s.store_backend, s.gcp_project, s.bigquery_dataset),
            provider=_provider(),
            embedder=build_embedder(
                s.embedding_provider, s.gcp_project, s.gcp_location,
                s.embedding_model, s.embedding_dimensions, s.gemini_api_key,
            ),
            artifacts=ArtifactStore(s.gcs_bucket, s.artifacts_dir, s.signed_url_days),
            masker=Masker(),
            prefilter=Prefilter(),
            chat_webhook_url=s.chat_webhook_url,
            window_hours=s.sentry_window_hours,
            top_n=s.top_n,
            selection_chunk_size=s.selection_chunk_size,
            force=force,
        )
    except Exception as exc:
        # A stage without its own handler would otherwise exit non-zero with no
        # Chat notice at all -- "the team knows the run did not complete" has to
        # hold for every failure class, not just the Sentry fetch.
        if not getattr(exc, "_chat_notified", False):
            try:
                post_error(s.chat_webhook_url, run_date, "pipeline", str(exc))
            except Exception:
                logger.exception("error notice post failed too")
        raise
    finally:
        sentry.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the daily alert triage pipeline.")
    parser.add_argument("--run-date", default=None, help="YYYY-MM-DD (default: today, UTC)")
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-run a date that already completed (the idempotency guard otherwise skips it)",
    )
    args = parser.parse_args(argv)

    setup_logging(get_settings().log_level)

    run_date = args.run_date or datetime.now(UTC).strftime("%Y-%m-%d")
    # run_date becomes a GCS object name and a BigQuery DATE parameter, so it is
    # pattern-locked rather than trusted as free text.
    if not RUN_DATE_RE.match(run_date):
        parser.error(f"--run-date must be YYYY-MM-DD, got {run_date!r}")

    try:
        result = run_once(run_date, force=args.force)
    except Exception:
        logger.exception("run for %s failed", run_date)
        return 1

    logger.info(
        "run %s finished: %s (degraded=%s) -- %s",
        result.run_id,
        result.stats.model_dump(),
        result.degraded,
        result.pdf_url or "no report",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
