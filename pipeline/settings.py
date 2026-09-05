from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class PipelineSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    gcp_project: str = ""
    gcp_location: str = "europe-west2"

    # Sentry ingestion. source=api pulls sentry.io directly (the dev/cloud
    # path); source=fixture reads raw issue JSON from a file, so a full run
    # works offline with no token.
    sentry_source: str = "api"  # api | fixture
    sentry_base_url: str = "https://sentry.io"
    sentry_token: str = ""
    sentry_org: str = ""
    sentry_window_hours: int = 24
    # Scope every query to the projects this team owns, resolved from the slug
    # on each run. Empty means "whatever the token can see", which is a
    # property of the token rather than a decision -- a user token tends to
    # mean the teams you belong to, an org token means the whole organisation.
    sentry_team: str = ""
    sentry_fixture_path: str = "fixtures/sentry_issues.sample.json"
    # Optional raw events/latest/ payload applied to every fixture issue, so an
    # offline run exercises the stack-trace and breadcrumb formatting that the
    # live path produces for every issue.
    sentry_fixture_detail_path: str = ""

    # Which environments to fetch, comma-separated.
    #
    # A STRING, not a list. pydantic-settings JSON-decodes a `list[str]` env
    # var inside the settings source, before any validator can see it -- so
    # `SENTRY_ENVIRONMENTS=production`, the obvious thing to type, raised a
    # SettingsError, and so did the comma-separated form the Terraform passes.
    # The deployed job would have died on its first start, naming the field but
    # not the format it wanted. A string with an explicit split has no trap.
    #
    # The names are CANONICAL and expanded at runtime against what the org
    # actually reports: "production" also fetches `live`, `@prod` and
    # `<service>@production`. See SentryApiClient._expand_environments.
    # Empty means one unfiltered query across every environment.
    sentry_environments: str = "production"

    @property
    def sentry_environment_list(self) -> list[str]:
        """`"production, live"` -> `["production", "live"]`; `""` -> `[]`."""
        return [name.strip() for name in self.sentry_environments.split(",") if name.strip()]

    # How many issues the top-issues agent selects for triage and the report.
    top_n: int = 10
    # Alerts per concurrent top-issues call during the map step.
    selection_chunk_size: int = 25

    # vertex - Vertex AI via ADC (the deployed path)
    # gemini - Gemini Developer API via GEMINI_API_KEY (local smoke test against
    #          a real model; same prompts, same schemas, no GCP setup)
    # mock   - deterministic heuristic (tests, offline demo)
    llm_provider: str = "vertex"
    # NOTE: the Developer API has retired gemini-2.5-flash for new keys
    # (404 "no longer available to new users"); Vertex still serves it. With
    # LLM_PROVIDER=gemini, set GEMINI_MODEL=gemini-3.5-flash.
    gemini_model: str = "gemini-2.5-flash"
    gemini_api_key: str = ""

    # vertex - text-embedding-004 through Vertex AI
    # hash   - deterministic offline embedder (tests, offline demo). Not
    #          semantic: it matches text, not meaning.
    embedding_provider: str = "vertex"  # vertex | hash
    embedding_model: str = "text-embedding-004"

    # bigquery - the vector store, audit trail and run history (deployed)
    # local    - JSON files with a brute-force cosine scan (offline)
    store_backend: str = "bigquery"  # bigquery | local
    bigquery_dataset: str = "prodtools_triage"

    gcs_bucket: str = ""  # empty -> write the report locally
    artifacts_dir: str = "artifacts"
    # 7 days is the MAXIMUM a V4 signed URL can be valid for -- it cannot be
    # raised. The PDF itself is durable in GCS; only the link expires. A Chat
    # card older than a week will therefore have a dead PDF button, and the fix
    # is to re-sign on demand, not to change this number.
    signed_url_days: int = 7

    chat_webhook_url: str = ""

    log_level: str = "INFO"


@lru_cache(maxsize=1)
def get_settings() -> PipelineSettings:
    return PipelineSettings()
