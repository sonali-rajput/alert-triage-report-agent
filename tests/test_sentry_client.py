"""Sentry client: pagination, the allowlist projection, and per-environment fetch.

The projection tests are the important ones. The raw issue payload carries
employee PII that the masker never sees (it only touches title and body), so
`issue_to_alert` is the only thing standing between the API response and our
Firestore/GCS archive.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import httpx
import pytest

from pipeline.sentry_client import (
    PII_BEARING_FIELDS,
    FixtureSentryClient,
    SentryApiClient,
    _is_retryable,
    _next_cursor,
    build_sentry_source,
    issue_to_alert,
)

FULL_PAYLOAD = Path("fixtures/sentry_issue_full.sample.json")


@pytest.fixture
def real_issue() -> dict:
    """The sanitized structural copy of a real prodtools issue. Its person
    fields hold synthetic stand-ins, but they are still PII-shaped, so a
    projection that leaked them fails these tests exactly as it would in
    production."""
    return json.loads(FULL_PAYLOAD.read_text(encoding="utf-8"))[0]


# --------------------------------------------------------------------------
# Allowlist projection / PII
# --------------------------------------------------------------------------


def test_projection_carries_no_email_addresses(real_issue):
    alert = issue_to_alert(real_issue, "production")
    blob = json.dumps(alert.model_dump(), default=str)
    assert "@example.invalid" not in blob
    assert not re.search(r"[\w.+-]+@[\w.-]+\.\w{2,}", blob), "an email address survived the projection"


def test_projection_carries_no_person_names_or_gravatars(real_issue):
    alert = issue_to_alert(real_issue, "production")
    blob = json.dumps(alert.model_dump(), default=str)
    assert "gravatar" not in blob.lower()
    for name in ("Ada Testperson", "Grace Fixture", "Alan Sample", "Edsger Mock"):
        assert name not in blob


def test_pii_bearing_fields_are_present_in_the_input(real_issue):
    """Guards the guard: if the fixture ever loses these, the tests above pass
    for the wrong reason."""
    for field in PII_BEARING_FIELDS:
        assert field in real_issue, f"fixture no longer carries {field}; PII tests are now vacuous"
    assert real_issue["activity"][0]["user"]["email"].endswith("@example.invalid")
    assert real_issue["seenBy"], "fixture no longer carries seenBy"


def test_projection_shrinks_the_payload(real_issue):
    """~10 KB of compact JSON down to ~1.3 KB, a bit under 8x. That is what
    makes archiving the projection to Firestore/GCS affordable. Most of what
    remains is `hourly_counts` (25 [ts, count] pairs)."""
    raw = len(json.dumps(real_issue))
    projected = len(json.dumps(issue_to_alert(real_issue, "production").model_dump(), default=str))
    assert projected < raw / 5


def test_projection_extracts_the_impact_signals(real_issue):
    alert = issue_to_alert(real_issue, "production")
    assert alert.source_id == "6316939353"
    assert alert.short_id == "P434-EF"
    assert alert.level == "fatal"
    assert alert.substatus == "ongoing"
    assert alert.is_unhandled is True
    assert alert.sentry_priority == "high"
    assert alert.platform == "native"
    assert alert.user_count == 9
    assert alert.event_count == 414  # `count` is the string "414" in the payload
    assert alert.environment == "production"
    assert alert.project == "p434"
    assert alert.first_release == "p434@1.0.0.0"
    assert alert.last_release == "p434@44553"
    assert alert.last_release_date is not None
    assert alert.seer_fixability == pytest.approx(0.37073206901550293)


def test_hourly_counts_are_timestamp_count_pairs(real_issue):
    alert = issue_to_alert(real_issue, "production")
    assert len(alert.hourly_counts) == 25
    assert all(isinstance(pair, tuple) and len(pair) == 2 for pair in alert.hourly_counts)
    # This sample is 13 days stale: 414 lifetime events, none in the last 24h.
    assert alert.events_24h() == 0
    assert alert.event_count == 414


def test_events_24h_falls_back_to_event_count_without_stats():
    alert = issue_to_alert({"id": "1", "count": "50"})
    assert alert.hourly_counts == []
    assert alert.events_24h() == 50


def test_missing_optional_fields_do_not_break_the_projection():
    alert = issue_to_alert({"id": "9"})
    assert alert.source_id == "9"
    assert alert.event_count == 1
    assert alert.user_count == 0
    assert alert.is_unhandled is False
    assert alert.seer_fixability is None


# --------------------------------------------------------------------------
# Link-header pagination
# --------------------------------------------------------------------------


def _link(cursor: str, results: str) -> str:
    return (
        '<https://sentry.io/api/0/x/?cursor=prev>; rel="previous"; results="false"; cursor="0:0:1", '
        f'<https://sentry.io/api/0/x/?cursor={cursor}>; rel="next"; results="{results}"; cursor="{cursor}"'
    )


def test_next_cursor_reads_the_next_entry():
    assert _next_cursor(_link("0:100:0", "true")) == "0:100:0"


def test_next_cursor_stops_on_results_false():
    """Sentry always emits a rel="next" entry; exhaustion is signalled by
    results="false". Following rel="next" blindly would loop forever."""
    assert _next_cursor(_link("0:300:0", "false")) is None
    assert _next_cursor("") is None


def _client_over(handler) -> SentryApiClient:
    client = SentryApiClient("https://sentry.io", "tok", "framestore", ["production"])
    client._client = httpx.Client(
        base_url="https://sentry.io/api/0", transport=httpx.MockTransport(handler)
    )
    return client


def _is_issue_list(request: httpx.Request) -> bool:
    """Every fetch_issues() call now makes a per-issue `events/latest/` request
    as well as the paged list request -- stack traces are fetched for every
    issue in the ingestion pass, not for a top N chosen later. Handlers that
    care about the list request have to say so."""
    return request.url.path.endswith("/issues/")


def _no_event_detail(request: httpx.Request) -> httpx.Response:
    """A 404 on `events/latest/` is routine: Sentry retains issues longer than
    it retains their events. The client has to treat it as a thinner alert,
    not a failure."""
    return httpx.Response(404, json={"detail": "no events"})


def test_pagination_follows_the_link_header_past_100_issues():
    """The bug being fixed: the client read one page of 100 and silently
    dropped everything after it."""
    pages = [
        ([{"id": f"a{i}"} for i in range(100)], _link("0:100:0", "true")),
        ([{"id": f"b{i}"} for i in range(100)], _link("0:200:0", "true")),
        ([{"id": f"c{i}"} for i in range(45)], _link("0:300:0", "false")),
    ]
    seen_cursors: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if not _is_issue_list(request):
            return _no_event_detail(request)
        cursor = request.url.params.get("cursor")
        seen_cursors.append(cursor)
        body, link = pages[len(seen_cursors) - 1]
        return httpx.Response(200, json=body, headers={"Link": link})

    alerts = _client_over(handler).fetch_issues(24)

    assert len(alerts) == 245
    assert seen_cursors == [None, "0:100:0", "0:200:0"]


def test_pagination_stops_at_the_page_cap():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=[{"id": f"x{request.url.params.get('cursor')}"}],
            headers={"Link": _link("0:999:0", "true")},  # never exhausts
        )

    alerts = _client_over(handler).fetch_issues(24)
    # Same cursor each time, so dedup by source_id collapses them; the point is
    # that it terminates at all.
    assert len(alerts) >= 1


def test_pagination_stops_on_an_empty_page():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if not _is_issue_list(request):
            return _no_event_detail(request)
        calls["n"] += 1
        body = [{"id": "only"}] if calls["n"] == 1 else []
        return httpx.Response(200, json=body, headers={"Link": _link("0:100:0", "true")})

    alerts = _client_over(handler).fetch_issues(24)
    assert len(alerts) == 1
    assert calls["n"] == 2


# --------------------------------------------------------------------------
# Event detail, fetched for EVERY issue in the ingestion pass
# --------------------------------------------------------------------------

EVENT_WITH_DETAIL = {
    "entries": [
        {
            "type": "exception",
            "data": {
                "values": [
                    {
                        "type": "ConnectionError",
                        "value": "host unreachable",
                        "stacktrace": {
                            "frames": [
                                {"filename": "app/db.py", "function": "connect", "lineNo": 42,
                                 "inApp": True}
                            ]
                        },
                    }
                ]
            },
        },
        {
            "type": "breadcrumbs",
            "data": {"values": [{"category": "http", "level": "warning", "message": "retry 3/3"}]},
        },
    ]
}


def test_every_issue_gets_its_stack_trace_and_breadcrumbs():
    """There is no top-N enrichment step any more. Choosing the top N is the
    top-issues agent's job, and it should not have to make that choice from
    thinner evidence than it uses for everything else."""

    def handler(request: httpx.Request) -> httpx.Response:
        if _is_issue_list(request):
            return httpx.Response(
                200,
                json=[{"id": "1"}, {"id": "2"}],
                headers={"Link": _link("0:100:0", "false")},
            )
        return httpx.Response(200, json=EVENT_WITH_DETAIL)

    alerts = _client_over(handler).fetch_issues(24)

    assert len(alerts) == 2
    for alert in alerts:
        assert "app/db.py:42 in connect" in alert.body
        assert "retry 3/3" in alert.body


def test_a_failed_detail_fetch_keeps_the_alert():
    """A missing stack trace makes an alert thinner. It does not make it
    untrue, and it must not cost the run an alert."""

    def handler(request: httpx.Request) -> httpx.Response:
        if _is_issue_list(request):
            return httpx.Response(
                200, json=[{"id": "1"}, {"id": "2"}], headers={"Link": _link("0:100:0", "false")}
            )
        if request.url.path.startswith("/api/0/issues/1/"):
            return httpx.Response(500, json={"detail": "boom"})
        return httpx.Response(200, json=EVENT_WITH_DETAIL)

    alerts = _client_over(handler).fetch_issues(24)

    assert [a.source_id for a in alerts] == ["1", "2"]
    assert "app/db.py" not in alerts[0].body
    assert "app/db.py" in alerts[1].body


# --------------------------------------------------------------------------
# Per-environment fetch
# --------------------------------------------------------------------------


def test_fetches_once_per_environment_and_stamps_the_value_on():
    client = SentryApiClient("https://sentry.io", "tok", "framestore", ["production", "staging"])
    requested: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if not _is_issue_list(request):
            return _no_event_detail(request)
        env = request.url.params.get("environment")
        requested.append(env)
        return httpx.Response(
            200, json=[{"id": f"{env}-1"}], headers={"Link": _link("0:100:0", "false")}
        )

    client._client = httpx.Client(
        base_url="https://sentry.io/api/0", transport=httpx.MockTransport(handler)
    )
    alerts = client.fetch_issues(24)

    assert requested == ["production", "staging"]
    assert {a.environment for a in alerts} == {"production", "staging"}
    assert {"environment:production", "environment:staging"} <= {
        label for a in alerts for label in a.labels
    }


def test_an_issue_in_two_environments_is_kept_once():
    client = SentryApiClient("https://sentry.io", "tok", "framestore", ["production", "staging"])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json=[{"id": "shared"}], headers={"Link": _link("0:100:0", "false")}
        )

    client._client = httpx.Client(
        base_url="https://sentry.io/api/0", transport=httpx.MockTransport(handler)
    )
    alerts = client.fetch_issues(24)

    assert len(alerts) == 1
    assert alerts[0].environment == "production"  # first configured env wins


def test_one_malformed_issue_does_not_kill_the_fetch():
    """`id` is the one field the projection cannot default; a single issue
    missing it must be skipped, not abort the whole run."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=[{"id": "good"}, {"title": "no id at all"}, {"id": "also-good"}],
            headers={"Link": _link("0:100:0", "false")},
        )

    alerts = _client_over(handler).fetch_issues(24)
    assert [a.source_id for a in alerts] == ["good", "also-good"]


def test_stats_period_does_not_round_hours_away():
    """36h must be sent as '36h', not floored to '1d' (losing 12 hours)."""
    periods: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        # Only the issue list carries a statsPeriod; the environments lookup
        # that expansion performs does not.
        if _is_issue_list(request):
            periods.append(request.url.params.get("statsPeriod"))
        return httpx.Response(200, json=[], headers={"Link": _link("0:100:0", "false")})

    client = _client_over(handler)
    for hours in (12, 24, 36, 48):
        client.fetch_issues(hours)
    assert periods == ["12h", "1d", "36h", "2d"]


# --------------------------------------------------------------------------
# Retry predicate
# --------------------------------------------------------------------------


def _status_error(code: int) -> httpx.HTTPStatusError:
    request = httpx.Request("GET", "https://sentry.io/api/0/x")
    return httpx.HTTPStatusError("boom", request=request, response=httpx.Response(code, request=request))


def test_client_errors_are_not_retried():
    """A 401 (bad token) or 404 (no retained events) is deterministic;
    retrying it burns 1+2+4s of backoff per call for nothing -- minutes of
    the job's timeout, now that every issue gets a detail call."""
    assert not _is_retryable(_status_error(401))
    assert not _is_retryable(_status_error(404))


def test_server_errors_and_rate_limits_are_retried():
    assert _is_retryable(_status_error(500))
    assert _is_retryable(_status_error(503))
    assert _is_retryable(_status_error(429))
    assert _is_retryable(httpx.ConnectError("refused"))


# --------------------------------------------------------------------------
# Fixture source
# --------------------------------------------------------------------------


def test_fixture_source_reads_the_full_payload():
    alerts = FixtureSentryClient(str(FULL_PAYLOAD)).fetch_issues(24)
    assert len(alerts) == 1
    assert alerts[0].environment == "production"
    assert alerts[0].short_id == "P434-EF"


def test_fixture_source_handles_a_missing_file(tmp_path):
    assert FixtureSentryClient(str(tmp_path / "nope.json")).fetch_issues(24) == []


# --------------------------------------------------------------------------
# Environment discovery
#
# Not called by the pipeline, but the runbook tells you to run it before
# setting SENTRY_ENVIRONMENTS — because that setting is an allowlist and a name
# that does not match means the run ingests nothing, with no error. A broken
# discovery helper is therefore a silent-empty-run generator.
# --------------------------------------------------------------------------


def test_list_environments_returns_the_names_the_org_actually_reports():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/environments/")
        return httpx.Response(200, json=[{"id": "1", "name": "production"},
                                         {"id": "2", "name": "@prod"},
                                         {"id": "3", "name": "Local"}])

    assert _client_over(handler).list_environments() == ["production", "@prod", "Local"]


def test_list_environments_tolerates_an_unexpected_shape():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"detail": "nope"})

    assert _client_over(handler).list_environments() == []


def test_list_environments_skips_nameless_entries():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[{"id": "1"}, {"name": ""}, {"name": "prod"}])

    assert _client_over(handler).list_environments() == ["prod"]


# --------------------------------------------------------------------------
# The fixture source's detail payload
# --------------------------------------------------------------------------


def test_a_detail_fixture_is_applied_to_every_issue(tmp_path):
    """Without it, offline alert bodies are issue-payload-only — a materially
    easier input than production ever hands the agents, which makes an offline
    quality judgement optimistic."""
    issues = tmp_path / "issues.json"
    issues.write_text(json.dumps([{"id": "1"}, {"id": "2"}]), encoding="utf-8")
    detail = tmp_path / "event.json"
    detail.write_text(json.dumps({
        "entries": [{"type": "exception", "data": {"values": [
            {"type": "ValueError", "value": "boom",
             "stacktrace": {"frames": [{"filename": "app/x.py", "function": "go", "lineNo": 7}]}}]}}]
    }), encoding="utf-8")

    alerts = FixtureSentryClient(str(issues), "production", str(detail)).fetch_issues(24)

    assert len(alerts) == 2
    assert all("app/x.py:7 in go" in a.body for a in alerts)


def test_no_detail_fixture_means_issue_payload_bodies_only(tmp_path):
    issues = tmp_path / "issues.json"
    issues.write_text(json.dumps([{"id": "1", "level": "error"}]), encoding="utf-8")

    alerts = FixtureSentryClient(str(issues), "production").fetch_issues(24)
    assert "Exception:" not in alerts[0].body


def test_an_unreadable_detail_fixture_degrades_to_no_detail(tmp_path):
    """A typo in the path must cost the stack traces, not the run."""
    issues = tmp_path / "issues.json"
    issues.write_text(json.dumps([{"id": "1"}]), encoding="utf-8")
    detail = tmp_path / "broken.json"
    detail.write_text("{not json", encoding="utf-8")

    alerts = FixtureSentryClient(str(issues), "production", str(detail)).fetch_issues(24)
    assert len(alerts) == 1


def test_a_missing_detail_fixture_path_is_ignored(tmp_path):
    issues = tmp_path / "issues.json"
    issues.write_text(json.dumps([{"id": "1"}]), encoding="utf-8")

    alerts = FixtureSentryClient(str(issues), "production", str(tmp_path / "absent.json")).fetch_issues(24)
    assert len(alerts) == 1


def test_build_sentry_source_rejects_an_unknown_source():
    with pytest.raises(ValueError):
        build_sentry_source("kafka", "https://sentry.io", "tok", "org", "f.json")


def test_the_fixture_source_gets_the_first_configured_environment():
    source = build_sentry_source("fixture", "", "", "", "f.json", ["@prod", "staging"])
    assert source._environment == "@prod"


# --------------------------------------------------------------------------
# Team scoping
#
# Measured against the real org: a *user* token's default scope happened to be
# "projects of teams I belong to" (18 issues) while `project=-1` returned 25.
# That difference is a property of the TOKEN, not a decision anyone made, so
# scoping is now explicit in the query.
# --------------------------------------------------------------------------


def _team_client(handler, team: str = "prodtools") -> SentryApiClient:
    client = SentryApiClient("https://sentry.io", "tok", "framestore", team=team)
    client._client = httpx.Client(
        base_url="https://sentry.io/api/0", transport=httpx.MockTransport(handler)
    )
    return client


def test_every_issue_query_is_scoped_to_the_teams_projects():
    sent: list[list[str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/projects/"):
            return httpx.Response(200, json=[{"id": "11", "slug": "prodtools-a"},
                                             {"id": "22", "slug": "prodtools-b"}])
        if _is_issue_list(request):
            sent.append(request.url.params.get_list("project"))
            return httpx.Response(200, json=[{"id": "1"}],
                                  headers={"Link": _link("0:100:0", "false")})
        return _no_event_detail(request)

    _team_client(handler).fetch_issues(24)
    assert sent == [["11", "22"]]


def test_the_team_is_resolved_once_per_run_not_once_per_environment():
    """A resolve per environment is a wasted round trip on every extra
    environment in the allowlist."""
    resolves = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/projects/"):
            resolves["n"] += 1
            return httpx.Response(200, json=[{"id": "11", "slug": "a"}])
        if _is_issue_list(request):
            return httpx.Response(200, json=[], headers={"Link": _link("0:100:0", "false")})
        return _no_event_detail(request)

    client = SentryApiClient("https://sentry.io", "tok", "framestore",
                             ["production", "staging"], team="prodtools")
    client._client = httpx.Client(base_url="https://sentry.io/api/0",
                                  transport=httpx.MockTransport(handler))
    client.fetch_issues(24)
    assert resolves["n"] == 1


def test_it_falls_back_to_the_org_team_listing():
    """The direct endpoint wants project:read; the org listing came back fine
    on an event:read + org:read token."""
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/projects/"):
            return httpx.Response(403, json={"detail": "no permission"})
        if request.url.path.endswith("/teams/"):
            return httpx.Response(200, json=[
                {"slug": "assetmake", "projects": [{"id": "99", "slug": "other"}]},
                {"slug": "prodtools", "projects": [{"id": "11", "slug": "ours"}]},
            ])
        if _is_issue_list(request):
            assert request.url.params.get_list("project") == ["11"]
            return httpx.Response(200, json=[], headers={"Link": _link("0:100:0", "false")})
        return _no_event_detail(request)

    assert _team_client(handler).fetch_issues(24) == []


def test_an_unknown_team_slug_fails_loudly_with_the_real_options():
    """Silently fetching the whole org instead would be the exact opposite of
    what team scoping was asked for, and it would look like a working run."""
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/projects/"):
            return httpx.Response(404, json={})
        if request.url.path.endswith("/teams/"):
            return httpx.Response(200, json=[{"slug": "technical"}, {"slug": "nontech"}])
        return httpx.Response(200, json=[], headers={"Link": _link("0:100:0", "false")})

    with pytest.raises(ValueError, match="technical"):
        _team_client(handler, team="prodtoolz").fetch_issues(24)


def test_a_team_with_no_projects_refuses_to_widen_the_scope():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/projects/"):
            return httpx.Response(200, json=[])
        if request.url.path.endswith("/teams/"):
            return httpx.Response(200, json=[{"slug": "prodtools", "projects": []}])
        return httpx.Response(200, json=[], headers={"Link": _link("0:100:0", "false")})

    with pytest.raises(ValueError, match="0 projects"):
        _team_client(handler).fetch_issues(24)


def test_a_leading_hash_on_the_slug_is_accepted():
    """People write it the way Sentry displays it: #prodtools."""
    assert SentryApiClient("https://sentry.io", "t", "o", team="#prodtools")._team == "prodtools"


def test_no_team_configured_sends_no_project_filter():
    """Backwards compatible: the scope stays whatever the token can see."""
    seen: list[list[str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if _is_issue_list(request):
            seen.append(request.url.params.get_list("project"))
            return httpx.Response(200, json=[], headers={"Link": _link("0:100:0", "false")})
        return _no_event_detail(request)

    _client_over(handler).fetch_issues(24)
    assert seen == [[]]


# --------------------------------------------------------------------------
# Environment expansion
#
# Measured on the real org: the literal spelling ["production"] reached 14
# issues; expanding it to every environment that IS production reached 18. The
# four it missed were only reachable through `live`. The org's list also holds
# flock-cron@production, help-web-api@production, flock-websockets@production
# and None@production.
# --------------------------------------------------------------------------

REAL_ENVIRONMENT_NAMES = [
    "production", "prod", "live", "None@production", "flock-cron@production",
    "flock-web-api@production", "flock-websockets@production", "help-cron@production",
    "help-web-api@production", "staging", "testing", "dev", "development",
    "help-web-api-dev@staging", "local-test", "vlaw",
]


def _env_client(handler, environments: list[str]) -> SentryApiClient:
    client = SentryApiClient("https://sentry.io", "tok", "framestore", environments)
    client._client = httpx.Client(
        base_url="https://sentry.io/api/0", transport=httpx.MockTransport(handler)
    )
    return client


def _org_handler(queried: list[str | None], names=REAL_ENVIRONMENT_NAMES):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/environments/"):
            return httpx.Response(200, json=[{"name": n} for n in names])
        if _is_issue_list(request):
            queried.append(request.url.params.get("environment"))
            return httpx.Response(200, json=[], headers={"Link": _link("0:100:0", "false")})
        return _no_event_detail(request)
    return handler


def test_production_expands_to_every_spelling_the_org_uses():
    queried: list[str | None] = []
    _env_client(_org_handler(queried), ["production"]).fetch_issues(24)

    assert "live" in queried, "the four issues only reachable via 'live' would be missed"
    assert "flock-cron@production" in queried
    assert "None@production" in queried
    assert "staging" not in queried and "testing" not in queried


def test_expansion_ignores_environments_that_are_not_production():
    queried: list[str | None] = []
    _env_client(_org_handler(queried), ["production"]).fetch_issues(24)
    assert not {"dev", "development", "local-test", "vlaw"} & set(queried)


def test_a_configured_name_matching_nothing_is_fatal_not_silent():
    """The original failure mode: an allowlist that matches no real environment
    ingests nothing, with no error, and looks like a quiet day.

    (`preprod` would NOT trigger this — it is an alias of `staging`, which the
    org does have. The alias table is doing its job there.)"""
    with pytest.raises(ValueError, match="Refusing to run"):
        _env_client(_org_handler([]), ["qa-sandbox"]).fetch_issues(24)


def test_an_unavailable_environment_list_falls_back_to_the_literal_names():
    """A token without the scope to list environments still gets the old
    behaviour, loudly degraded rather than broken."""
    queried: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/environments/"):
            return httpx.Response(403, json={"detail": "no permission"})
        if _is_issue_list(request):
            queried.append(request.url.params.get("environment"))
            return httpx.Response(200, json=[], headers={"Link": _link("0:100:0", "false")})
        return _no_event_detail(request)

    _env_client(handler, ["production"]).fetch_issues(24)
    assert queried == ["production"]


def test_an_empty_environment_list_means_one_unfiltered_query():
    queried: list[str | None] = []
    _env_client(_org_handler(queried), []).fetch_issues(24)
    assert queried == [None]


def test_the_queried_spelling_is_what_gets_stamped_on_the_alert():
    """The query is the STRONGER claim about environment: it says 'this occurs
    in production', where the event's tag only says 'the last one happened to
    be'. On the real org, 1 issue in 18 fired in production while its latest
    event said testing -- trusting the tag would have let drop_environments
    suppress a production error."""
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/environments/"):
            return httpx.Response(200, json=[{"name": "live"}])
        if _is_issue_list(request):
            return httpx.Response(200, json=[{"id": "1"}],
                                  headers={"Link": _link("0:100:0", "false")})
        # An event whose tag disagrees with the environment we queried.
        return httpx.Response(200, json={"tags": [{"key": "environment", "value": "testing"}]})

    alerts = _env_client(handler, ["production"]).fetch_issues(24)
    assert alerts[0].environment == "live"
    assert "environment:live" in alerts[0].labels
