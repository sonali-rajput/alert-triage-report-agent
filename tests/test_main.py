"""The Cloud Run Job entrypoint.

Exit codes are the interface: 0 means the run completed (including the degraded
digest path, which is a completed run with a worse report), non-zero means it
did not -- and non-zero is what Cloud Scheduler retries on.
"""

from __future__ import annotations

import json

import pytest

SAMPLE_ISSUES = [
    {
        "id": "42",
        "title": "Unhandled TypeError in export",
        "culprit": "review_tool.export",
        "level": "error",
        "count": "150",
        "userCount": 12,
        "permalink": "https://framestore.sentry.io/issues/42/",
        "project": {"slug": "review-tool"},
        "metadata": {"type": "TypeError", "value": "boom"},
    }
]


@pytest.fixture
def offline(monkeypatch, tmp_path):
    """A fully offline environment: fixture Sentry, mock LLM, hash embedder,
    local store, local artifacts, no Chat webhook."""
    fixture = tmp_path / "issues.json"
    fixture.write_text(json.dumps(SAMPLE_ISSUES), encoding="utf-8")

    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "hash")
    monkeypatch.setenv("STORE_BACKEND", "local")
    monkeypatch.setenv("GCS_BUCKET", "")
    monkeypatch.setenv("CHAT_WEBHOOK_URL", "")
    monkeypatch.setenv("SENTRY_SOURCE", "fixture")
    monkeypatch.setenv("SENTRY_FIXTURE_PATH", str(fixture))
    monkeypatch.chdir(tmp_path)

    from pipeline.settings import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_a_completed_run_exits_zero(offline):
    from pipeline.main import main

    assert main(["--run-date", "2026-07-15"]) == 0


def test_the_run_actually_triages_the_fixture(offline):
    from pipeline.main import run_once

    result = run_once("2026-07-15")
    assert result.stats.ingested == 1
    assert len(result.results) == 1
    assert result.results[0].triage.priority.value in {"low", "medium", "high", "critical"}


def test_run_date_defaults_to_today(offline):
    from pipeline.main import main

    assert main([]) == 0


def test_a_malformed_run_date_is_rejected_at_the_boundary(offline):
    """run_date becomes a GCS object name and a BigQuery DATE parameter, so it
    is shape-checked rather than trusted as free text."""
    from pipeline.main import main

    with pytest.raises(SystemExit):
        main(["--run-date", "../../etc/passwd"])


def test_a_failed_run_exits_non_zero_and_posts_an_error_notice(offline, monkeypatch):
    """Every failure class must reach Chat, not just the Sentry fetch: a stage
    with no handler of its own would otherwise exit quietly."""
    from pipeline import main as entrypoint

    notices: list[tuple[str, str]] = []

    def boom(*args, **kwargs):
        raise RuntimeError("bigquery quota exceeded")

    monkeypatch.setattr(entrypoint, "execute_run", boom)
    monkeypatch.setattr(
        entrypoint, "post_error", lambda url, run_date, stage, error: notices.append((stage, error))
    )

    assert entrypoint.main(["--run-date", "2026-07-15"]) == 1
    assert notices == [("pipeline", "bigquery quota exceeded")]


def test_an_error_already_reported_is_not_reported_twice(offline, monkeypatch):
    """The Sentry-fetch handler posts its own notice and marks the exception.
    Without the marker the team gets two Chat cards for one failure."""
    from pipeline import main as entrypoint

    notices: list[tuple[str, str]] = []

    def boom(*args, **kwargs):
        exc = RuntimeError("sentry unreachable")
        exc._chat_notified = True
        raise exc

    monkeypatch.setattr(entrypoint, "execute_run", boom)
    monkeypatch.setattr(
        entrypoint, "post_error", lambda url, run_date, stage, error: notices.append((stage, error))
    )

    assert entrypoint.main(["--run-date", "2026-07-15"]) == 1
    assert notices == []


# --------------------------------------------------------------------------
# SENTRY_ENVIRONMENTS parsing
#
# This was a deployment bug, not a hypothetical: the field was `list[str]`, and
# pydantic-settings JSON-decodes those inside the settings source before any
# validator can see them. The Terraform passes "production,prod,@prod" and the
# obvious hand-typed value is "production" — both raised SettingsError, so the
# Cloud Run Job would have died on its first start.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("production", ["production"]),
        ("production,prod,@prod", ["production", "prod", "@prod"]),
        ("production, live , @prod", ["production", "live", "@prod"]),   # tolerant of spaces
        ("", []),                                                        # unfiltered
        ("production,,prod", ["production", "prod"]),                    # tolerant of blanks
    ],
)
def test_sentry_environments_accepts_the_forms_people_and_terraform_write(raw, expected, monkeypatch):
    from pipeline.settings import PipelineSettings

    monkeypatch.setenv("SENTRY_ENVIRONMENTS", raw)
    settings = PipelineSettings(_env_file=None)
    assert settings.sentry_environment_list == expected


def test_the_terraform_default_parses():
    """The exact string infra/environments/dev passes as SENTRY_ENVIRONMENTS.

    Reads the variable DEFAULT out of variables.tf rather than a tfvars file:
    envs/*.tfvars carry real per-org values and are gitignored, so they are not
    present in CI or in the test image. The default is the committed contract.
    """
    import re
    from pathlib import Path

    from pipeline.settings import PipelineSettings

    tf = Path("infra/environments/dev/variables.tf").read_text(encoding="utf-8")
    block = re.search(
        r'variable\s+"sentry_environments"\s*\{(.*?)\n\}', tf, re.DOTALL
    )
    assert block, "infra/environments/dev no longer declares sentry_environments"
    match = re.search(r'default\s*=\s*"([^"]*)"', block.group(1))
    assert match, "sentry_environments no longer has a default"

    settings = PipelineSettings(_env_file=None, sentry_environments=match.group(1))
    assert len(settings.sentry_environment_list) >= 1


# --------------------------------------------------------------------------
# Terraform / code agreement
#
# The Terraform sets these as environment variables on the Cloud Run Job, so a
# value there SILENTLY OVERRIDES the code default. That is how the deployed
# model came to be `gemini-3.6-flash` while every real-model result in this
# project was measured against `gemini-3.5-flash` — an evaluation describing a
# system that was not the one running.
# --------------------------------------------------------------------------


def _terraform_defaults() -> dict[str, str]:
    """Parse `variable "x" { ... default = y }` out of the dev environment."""
    import re
    from pathlib import Path

    text = Path("infra/environments/dev/variables.tf").read_text(encoding="utf-8")
    found = {}
    for block in re.finditer(r'variable\s+"(\w+)"\s*\{(.*?)\n\}', text, re.S):
        name, body = block.group(1), block.group(2)
        default = re.search(r'^\s*default\s*=\s*"?([^"\n]+?)"?\s*$', body, re.M)
        if default:
            found[name] = default.group(1).strip()
    return found


@pytest.mark.parametrize(
    "tf_var,setting",
    [
        ("gemini_model", "gemini_model"),
        ("embedding_model", "embedding_model"),
        ("embedding_dimensions", "embedding_dimensions"),
    ],
)
def test_terraform_defaults_match_the_code_defaults(tf_var, setting):
    from pipeline.settings import PipelineSettings

    tf = _terraform_defaults()
    assert tf_var in tf, f"infra/environments/dev/variables.tf no longer defines {tf_var}"
    expected = getattr(PipelineSettings(_env_file=None), setting)
    assert str(tf[tf_var]) == str(expected), (
        f"Terraform sets {tf_var}={tf[tf_var]!r} but the code defaults to {expected!r}. "
        "The Terraform value wins at runtime, so the deployed job would use a model or "
        "width that nothing here was tested against."
    )


def test_the_terraform_actually_passes_those_to_the_job():
    """A variable nobody wires into the job's env is a variable that does
    nothing, and the test above would still pass."""
    from pathlib import Path

    main = Path("infra/environments/dev/main.tf").read_text(encoding="utf-8")
    for env_var, tf_var in [("GEMINI_MODEL", "gemini_model"),
                            ("EMBEDDING_MODEL", "embedding_model"),
                            ("EMBEDDING_DIMENSIONS", "embedding_dimensions")]:
        assert f"var.{tf_var}" in main, f"{tf_var} is never used"
        assert env_var in main, f"{env_var} is never set on the job"
