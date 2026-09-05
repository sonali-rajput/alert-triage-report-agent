"""Sentry ingestion, inside the pipeline service.

The service pulls unresolved issues from Sentry (sentry.io SaaS) directly over
HTTPS, so there is no on-prem collector. For local development and tests there
is a fixture source that reads the same raw Sentry issue JSON from a file, so
the whole run works offline with no token — the same pattern as the mock LLM
provider. Which one is used is chosen by SENTRY_SOURCE (api | fixture).

Sentry is READ-ONLY for this project. Nothing here ever writes back.
"""

from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Protocol

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from pipeline.event_text import event_to_body_text
from shared.environments import canonical_environment
from shared.models import Alert, AlertSource

logger = logging.getLogger(__name__)


def _is_retryable(exc: BaseException) -> bool:
    """Retry transport errors and server-side failures only. A 4xx (bad token,
    404 on an issue with no retained events) is deterministic -- retrying it
    just burns 1+2+4s of backoff per call, and every issue now gets a detail
    call, so that is minutes of the job's timeout spent on nothing."""
    if isinstance(exc, httpx.TransportError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        return code >= 500 or code == 429
    return False


_RETRY: dict[str, Any] = dict(
    retry=retry_if_exception(_is_retryable),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    stop=stop_after_attempt(4),
    reraise=True,
)

# Parallel per-issue detail fetches. Every issue now gets its stack trace and
# breadcrumbs in the ingestion pass -- there is no "enrich the top N" step any
# more, because deciding the top N is the top-issues agent's job and it should
# not have to make that call from thinner evidence than it uses elsewhere. That
# turns one call per issue into hundreds of calls, so they run concurrently;
# 8 is well under Sentry's rate limit and keeps a 200-issue org under a minute.
DETAIL_CONCURRENCY = 8

# Safety cap so a runaway query can't blow up a run. 10 pages x 100 issues.
MAX_PAGES = 10
PAGE_SIZE = 100

# One entry of a Link header, e.g.
#   <https://sentry.io/api/0/...>; rel="next"; results="true"; cursor="0:100:0"
_LINK_ENTRY = re.compile(
    r'rel="(?P<rel>[^"]*)"[^,]*?results="(?P<results>[^"]*)"[^,]*?cursor="(?P<cursor>[^"]*)"'
)

# Fields deliberately NOT projected, kept here as documentation for the test in
# tests/test_sentry_client.py. Every one of these carries colleague names,
# email addresses, gravatar hashes (which are hashes OF emails) or login
# timestamps. The masker only ever sees `title` and `body`, so it would never
# catch any of this -- the projection is the only thing standing between the
# raw payload and the BigQuery audit table.
PII_BEARING_FIELDS = ("activity", "seenBy", "assignedTo", "participants", "annotations")


def _next_cursor(link_header: str) -> str | None:
    """Return the cursor for the next page, or None when the results are
    exhausted. Sentry signals the end with results="false" rather than by
    omitting the rel="next" entry."""
    if not link_header:
        return None
    for match in _LINK_ENTRY.finditer(link_header):
        if match.group("rel") == "next" and match.group("results") == "true":
            return match.group("cursor")
    return None


def _as_int(value: Any, default: int = 0) -> int:
    """Sentry returns some counters as strings (`count` is `"414"`)."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _hourly_counts(stats: object) -> list[tuple[int, int]]:
    """`stats["24h"]` is a list of [unix_ts, count] PAIRS, not a flat list."""
    if not isinstance(stats, list):
        return []
    pairs: list[tuple[int, int]] = []
    for entry in stats:
        if isinstance(entry, (list, tuple)) and len(entry) == 2:
            pairs.append((_as_int(entry[0]), _as_int(entry[1])))
    return pairs


def issue_to_alert(issue: dict, environment: str = "") -> Alert:
    """Map one raw Sentry issue object to a normalized Alert.

    This is a STRICT ALLOWLIST PROJECTION, not a filter-out. Only the fields
    named below ever leave this function, because the Alert it returns is what
    gets written to the BigQuery audit table. The raw response is never stored:
    it is ~14 KB per issue and carries employee PII (see PII_BEARING_FIELDS).
    The projection also takes it to ~600 bytes, which is what makes keeping
    every alert of every run affordable in the first place.

    Shared by the live API client and the fixture client so both exercise the
    same parsing.
    """
    level = issue.get("level", "error")
    culprit = issue.get("culprit") or ""
    metadata = issue.get("metadata") or {}
    project = issue.get("project") or {}
    first_release = issue.get("firstRelease") or {}
    last_release = issue.get("lastRelease") or {}
    stats = issue.get("stats") or {}

    body_parts = [
        f"Level: {level}",
        f"Culprit: {culprit}" if culprit else "",
        f"Type: {metadata.get('type', '')}" if metadata.get("type") else "",
        f"Function: {metadata.get('function', '')}" if metadata.get("function") else "",
        f"Value: {metadata.get('value', '')}" if metadata.get("value") else "",
    ]

    labels = [f"level:{level}"]
    if issue.get("substatus"):
        labels.append(f"substatus:{issue['substatus']}")
    if environment:
        labels.append(f"environment:{environment}")

    return Alert(
        source=AlertSource.sentry,
        source_id=str(issue["id"]),
        kind="sentry_issue",
        title=issue.get("title", ""),
        body="\n".join(p for p in body_parts if p),
        url=issue.get("permalink", ""),
        project=project.get("slug", ""),
        event_count=_as_int(issue.get("count"), 1),
        first_seen=issue.get("firstSeen"),
        last_seen=issue.get("lastSeen"),
        labels=labels,
        # --- impact signals ---
        short_id=issue.get("shortId") or "",
        user_count=_as_int(issue.get("userCount")),
        environment=environment,
        substatus=issue.get("substatus") or "",
        is_unhandled=bool(issue.get("isUnhandled")),
        level=level,
        sentry_priority=issue.get("priority") or "",
        platform=issue.get("platform") or "",
        seer_fixability=issue.get("seerFixabilityScore"),
        hourly_counts=_hourly_counts(stats.get("24h")),
        first_release=first_release.get("version") or "",
        last_release=last_release.get("version") or "",
        last_release_date=last_release.get("dateCreated"),
    )


class SentrySource(Protocol):
    def fetch_issues(self, window_hours: int) -> list[Alert]:
        """Every unresolved issue in the window, WITH its stack trace and
        breadcrumbs already folded into the body."""
        ...

    def close(self) -> None: ...


class SentryApiClient:
    """Live Sentry SaaS client."""

    def __init__(
        self,
        base_url: str,
        token: str,
        org: str,
        environments: list[str] | None = None,
        timeout: float = 30.0,
        team: str = "",
    ):
        self._org = org
        self._team = team.lstrip("#")
        self._environments = list(environments or []) or [""]
        # Resolved lazily on the first fetch, then reused for the run.
        self._project_ids: list[str] | None = None
        self._client = httpx.Client(
            base_url=f"{base_url.rstrip('/')}/api/0",
            headers={"Authorization": f"Bearer {token}"},
            timeout=timeout,
        )

    @retry(**_RETRY)
    def _request(self, path: str, params: dict | None = None) -> httpx.Response:
        """Returns the response rather than the parsed body so pagination can
        read the Link header. Retries still apply."""
        resp = self._client.get(path, params=params or {})
        resp.raise_for_status()
        return resp

    def _get(self, path: str, params: dict | None = None) -> list | dict:
        return self._request(path, params).json()

    def _team_project_ids(self) -> list[str]:
        """Project IDs owned by `self._team`, resolved once per run.

        Scoping by TEAM rather than by project list is what makes this
        configurable with a single slug: the team gains a project, the next run
        picks it up. Hard-coding project ids would need a config change every
        time someone spins up a service.

        Two endpoints, because token scopes vary: the direct one first, then
        the org-wide team listing filtered by slug. On the org this was built
        against, an `event:read` + `org:read` token could read the listing.

        Resolving to ZERO projects raises. It is the same silent-empty-run
        failure the environment allowlist has -- a typo'd slug would otherwise
        fetch the whole org (no `project` filter at all), which is the opposite
        of what was asked for and looks like a working run.
        """
        if self._project_ids is not None:
            return self._project_ids

        projects: list = []
        try:
            direct = self._get(f"/teams/{self._org}/{self._team}/projects/")
            projects = direct if isinstance(direct, list) else []
        except Exception as exc:
            logger.info("sentry api: /teams/%s/%s/projects/ unavailable (%s); "
                        "falling back to the org team listing", self._org, self._team, exc)

        if not projects:
            teams = self._get(f"/organizations/{self._org}/teams/")
            match = next(
                (t for t in (teams if isinstance(teams, list) else [])
                 if str(t.get("slug", "")).lower() == self._team.lower()),
                None,
            )
            if match is None:
                available = [t.get("slug") for t in teams] if isinstance(teams, list) else []
                raise ValueError(
                    f"SENTRY_TEAM='{self._team}' is not a team in org '{self._org}'. "
                    f"Available: {available}"
                )
            projects = match.get("projects") or []

        ids = [str(p["id"]) for p in projects if isinstance(p, dict) and p.get("id")]
        if not ids:
            raise ValueError(
                f"team '{self._team}' resolved to 0 projects. Refusing to fall back to an "
                "unscoped org-wide fetch -- that would silently triage every team's issues."
            )

        slugs = [p.get("slug") for p in projects if isinstance(p, dict)]
        logger.info("sentry api: team '%s' owns %d project(s): %s", self._team, len(ids), slugs)
        self._project_ids = ids
        return ids

    def _expand_environments(self) -> list[str]:
        """Turn the configured CANONICAL environments into the org's actual
        spellings, by asking the org what it has.

        `SENTRY_ENVIRONMENTS=["production"]` means "every environment that IS
        production", not "the environment literally called production". The
        difference is not academic: measured on the real org, the literal
        spelling reached 14 issues and the expanded list reached 18. The four
        it missed were only reachable through `live`, and nobody was being told
        about them.

        That org's list also contains `flock-cron@production`,
        `help-web-api@production`, `flock-websockets@production` and
        `None@production` -- a `<service>@<environment>` convention that no
        hand-maintained allowlist would have kept up with. Expansion is
        self-maintaining: a service added next week is picked up on the next
        run, because `canonical_environment` reads the half after the `@`.

        This deliberately does NOT read the environment off the event's tags.
        On the same org, 2 of 18 issues had a latest event from a different
        environment than the one they were queried under, and for 1 of them
        the event said `testing` while the issue also fires in production.
        The query is the stronger claim -- it says "this occurs in
        production", where the tag only says "the last one happened to be".
        """
        if not any(self._environments):
            return [""]  # explicitly unfiltered: one query, every environment

        try:
            available = self.list_environments()
        except Exception as exc:
            logger.warning(
                "sentry api: could not list environments (%s); querying the configured "
                "names literally, which will miss any other spelling the org uses", exc
            )
            return self._environments

        if not available:
            # No evidence either way -- not the same as "your config is wrong".
            return self._environments

        wanted = {canonical_environment(e) for e in self._environments}
        expanded = [name for name in available if canonical_environment(name) in wanted]
        if not expanded:
            raise ValueError(
                f"SENTRY_ENVIRONMENTS={self._environments} matches none of the "
                f"{len(available)} environments this org reports: {available}. "
                "Refusing to run a query that would ingest nothing."
            )

        added = [e for e in expanded if e not in self._environments]
        if added:
            logger.info(
                "sentry api: %s expands to %d environment(s); also querying %s",
                self._environments, len(expanded), added,
            )
        return expanded

    def _fetch_environment(self, environment: str, stats_period: str) -> list[Alert]:
        """Page through the issues endpoint for one environment.

        Sentry pages via the Link response header rather than an offset, and
        signals exhaustion with results="false" -- a rel="next" entry is always
        present, so following it blindly would loop forever.
        """
        alerts: list[Alert] = []
        cursor: str | None = None

        for page in range(MAX_PAGES):
            params: dict = {
                "statsPeriod": stats_period,
                "query": "is:unresolved",
                "limit": PAGE_SIZE,
                "sort": "freq",
            }
            if environment:
                params["environment"] = environment
            if self._team:
                # Repeated `project` params. Without this the org-issues
                # endpoint returns whatever the TOKEN can see -- which on a
                # user token happens to be "projects of teams I belong to" and
                # on an org token is the entire organisation. Relying on that
                # difference means the scope changes when the token does.
                params["project"] = self._team_project_ids()
            if cursor:
                params["cursor"] = cursor

            resp = self._request(f"/organizations/{self._org}/issues/", params)
            issues = resp.json()
            if not issues:
                break
            for issue in issues:
                try:
                    alerts.append(issue_to_alert(issue, environment))
                except Exception:
                    # One malformed issue must not kill the whole fetch.
                    issue_id = issue.get("id", "?") if isinstance(issue, dict) else "?"
                    logger.exception("sentry api: skipping malformed issue %s", issue_id)

            cursor = _next_cursor(resp.headers.get("Link", ""))
            if not cursor:
                break
        else:
            logger.warning(
                "sentry api: hit the %d-page cap for environment '%s'; "
                "results are truncated at %d issues",
                MAX_PAGES,
                environment or "(all)",
                len(alerts),
            )

        logger.info(
            "sentry api: fetched %d issues for environment '%s' (%s window)",
            len(alerts),
            environment or "(all)",
            stats_period,
        )
        return alerts

    def fetch_issues(self, window_hours: int) -> list[Alert]:
        """Fetch unresolved issues once per configured environment.

        The issue payload's `tags` only carries key/name/totalValues, not tag
        *values*, so the environment of an issue cannot be read off the payload.
        Querying per environment and stamping the value on is the way to get it.
        """
        # Sentry accepts hour-granular periods ("36h"), so only use days when
        # the window is a whole number of them -- "36h" -> "1d" would silently
        # drop 12 hours of the query window.
        if window_hours >= 24 and window_hours % 24 == 0:
            stats_period = f"{window_hours // 24}d"
        else:
            stats_period = f"{max(1, window_hours)}h"

        alerts: list[Alert] = []
        seen: set[str] = set()
        for environment in self._expand_environments():
            for alert in self._fetch_environment(environment, stats_period):
                # An issue can appear under more than one environment; keep the
                # first, which is the earliest-listed (highest priority) env.
                if alert.source_id in seen:
                    continue
                seen.add(alert.source_id)
                alerts.append(alert)

        logger.info("sentry api: %d unique issues", len(alerts))
        return self._add_details(alerts)

    def list_environments(self) -> list[str]:
        """Every environment name the org actually reports.

        `SENTRY_ENVIRONMENTS` is an allowlist -- anything not in it is never
        fetched at all -- and Sentry environment names are free text set by each
        SDK's own config. A real org's list is not `["production", "staging"]`;
        it contains `prod`, `@prod`, `@dev`, `live`, `test` and one-offs left
        behind by services configured years ago. Guessing the name means the
        daily run quietly ingests nothing from that environment.

        Not called by the pipeline. It exists so the list can be read off the
        org before `SENTRY_ENVIRONMENTS` is filled in:

            python -c "from pipeline.sentry_client import SentryApiClient; \\
                print(SentryApiClient(url, token, org).list_environments())"
        """
        payload = self._get(f"/organizations/{self._org}/environments/")
        if not isinstance(payload, list):
            return []
        return [str(e.get("name", "")) for e in payload if isinstance(e, dict) and e.get("name")]

    def _fetch_latest_event(self, issue_id: str) -> dict | None:
        """The most recent event for an issue: stack frames and breadcrumbs."""
        event = self._get(f"/issues/{issue_id}/events/latest/")
        return event if isinstance(event, dict) else None

    def _add_details(self, alerts: list[Alert]) -> list[Alert]:
        """Fold each issue's latest event into its body, concurrently.

        Per-issue failures are swallowed by design: an alert that could not be
        detailed keeps its issue-payload body and still goes through the rest
        of the pipeline. A missing stack trace makes an alert thinner, it does
        not make it untrue. (404 on `events/latest/` is routine -- Sentry
        retains issues longer than it retains their events.)
        """
        if not alerts:
            return alerts

        def detail(alert: Alert) -> Alert:
            try:
                event = self._fetch_latest_event(alert.source_id)
            except Exception as exc:
                logger.warning("sentry api: no detail for issue %s (%s)", alert.source_id, exc)
                return alert
            extra = event_to_body_text(event) if event else ""
            if not extra:
                return alert
            return alert.model_copy(update={"body": f"{alert.body}\n\n{extra}"})

        with ThreadPoolExecutor(max_workers=DETAIL_CONCURRENCY) as pool:
            detailed = list(pool.map(detail, alerts))

        with_body = sum(1 for a, b in zip(alerts, detailed) if a.body != b.body)
        logger.info("sentry api: added event detail to %d of %d issues", with_body, len(alerts))
        return detailed

    def close(self) -> None:
        self._client.close()


class FixtureSentryClient:
    """Offline source that reads raw Sentry issue JSON from a file. Used for
    local dev and tests so a full run needs no Sentry token."""

    def __init__(self, path: str, environment: str = "", detail_path: str = ""):
        self._path = Path(path)
        # An optional second fixture: a raw `events/latest/` payload, applied to
        # every issue so the offline run exercises the stack-trace and
        # breadcrumb formatting the live path produces. Without it the offline
        # bodies are issue-payload-only, which is a materially easier input
        # than production ever hands the agents.
        self._detail_path = Path(detail_path) if detail_path else None
        # A real API response carries NO environment: the issue payload's `tags`
        # lists tag keys without values, which is why the live client stamps the
        # environment it queried onto every alert. A fixture captured from that
        # response therefore has nothing to read either, and without this
        # every offline alert would score `environment: ""` -- silently zeroing
        # a 1.5-weight signal and disabling the drop_environments prefilter.
        # Hand-written fixtures may still carry a per-issue `environment` key,
        # which wins.
        self._environment = environment

    def fetch_issues(self, window_hours: int) -> list[Alert]:
        if not self._path.exists():
            logger.warning("sentry fixture %s not found; ingesting 0 issues", self._path)
            return []
        issues = json.loads(self._path.read_text(encoding="utf-8"))
        if isinstance(issues, dict):  # a single-issue payload dump
            issues = [issues]
        alerts = [
            issue_to_alert(issue, issue.get("environment") or self._environment)
            for issue in issues
        ]
        extra = self._detail_text()
        if extra:
            alerts = [a.model_copy(update={"body": f"{a.body}\n\n{extra}"}) for a in alerts]
        logger.info(
            "sentry fixture: loaded %d issues from %s (environment '%s')",
            len(alerts),
            self._path,
            self._environment or "(per-issue / none)",
        )
        return alerts

    def _detail_text(self) -> str:
        if self._detail_path is None or not self._detail_path.exists():
            return ""
        try:
            return event_to_body_text(json.loads(self._detail_path.read_text(encoding="utf-8")))
        except Exception:
            logger.exception("sentry fixture: could not read detail fixture %s", self._detail_path)
            return ""

    def close(self) -> None:  # nothing to close
        pass


def build_sentry_source(
    source: str,
    base_url: str,
    token: str,
    org: str,
    fixture_path: str,
    environments: list[str] | None = None,
    fixture_detail_path: str = "",
    team: str = "",
) -> SentrySource:
    if source == "api":
        return SentryApiClient(base_url, token, org, environments, team=team)
    if source == "fixture":
        # Stamp the first configured environment, mirroring what the live
        # client does with the environment it queried. Without it every offline
        # alert reads as environment-less, which silently disables the
        # drop_environments prefilter and one of the selection rules.
        return FixtureSentryClient(fixture_path, (environments or [""])[0], fixture_detail_path)
    raise ValueError(f"unknown sentry source: {source}")
