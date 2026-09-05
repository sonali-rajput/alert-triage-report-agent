"""Regression tests against a real 24h org-issues pull (anonymized).

`fixtures/sentry_org_issues_last_24h_anonymized.json` is an actual
`GET /organizations/{org}/issues/?statsPeriod=24h` response with names, URLs and
IDs replaced. Its value is that it has the SHAPE the hand-written fixtures do
not -- and the shape is where the assumptions broke.

Every test here is anchored to something the hand-written fixtures got wrong:
they carry a top-level `environment` key a real response never has, and their
`count` happens to agree with their 24h stats, which a real one never does.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline.agents.top_issues import _payload as selection_payload
from pipeline.agents.triage import _payload
from pipeline.embeddings import HashEmbedder, cosine_distance
from pipeline.prefilter import Prefilter
from pipeline.sentry_client import PII_BEARING_FIELDS, FixtureSentryClient, issue_to_alert

FIXTURE = Path("fixtures/sentry_org_issues_last_24h_anonymized.json")

pytestmark = pytest.mark.skipif(not FIXTURE.exists(), reason="org-pull fixture not present")


@pytest.fixture(scope="module")
def raw_issues() -> list[dict]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def alerts():
    return FixtureSentryClient(str(FIXTURE), "production").fetch_issues(24)


# --------------------------------------------------------------------------
# Shape of a real response
# --------------------------------------------------------------------------


def test_a_real_response_carries_no_environment_field(raw_issues):
    """The premise behind the per-environment query. If this ever fails, the
    client could read the environment off the payload and stop paying for one
    paged query per environment."""
    assert not any("environment" in issue for issue in raw_issues)


def test_tags_carry_keys_but_no_values(raw_issues):
    """The other half of that premise: `tags` lists which tag keys exist and how
    many values each has, never the values themselves -- so `environment` is
    present as a KEY on every issue and still tells you nothing."""
    for issue in raw_issues:
        for tag in issue.get("tags") or []:
            assert set(tag) <= {"key", "name", "totalValues"}, tag


def test_count_is_the_all_time_total_not_the_window(raw_issues):
    """`count` is NOT a 24h number, and not by a little: the largest issue here
    reads 206,017 against 223 events in the last 24 hours. Anything volume-
    related must use `Alert.events_24h()`."""
    worst = max(raw_issues, key=lambda i: int(i["count"]))
    window = sum(c for _ts, c in worst["stats"]["24h"])
    assert int(worst["count"]) > window * 100


def test_the_fixture_still_carries_the_pii_bearing_fields(raw_issues):
    """The projection is only worth testing against a payload that actually has
    something to leak. Anonymization must not have stripped these."""
    present = {f for f in PII_BEARING_FIELDS if any(issue.get(f) for issue in raw_issues)}
    assert present, "no PII-bearing fields left; this fixture cannot test the projection"


def test_the_projection_drops_every_pii_bearing_field(raw_issues):
    for issue in raw_issues:
        blob = json.dumps(issue_to_alert(issue, "production").model_dump(), default=str)
        for field in PII_BEARING_FIELDS:
            assert f'"{field}"' not in blob


# --------------------------------------------------------------------------
# What the pipeline makes of it
# --------------------------------------------------------------------------


def test_every_alert_gets_the_queried_environment_stamped_on(alerts):
    """Without this the fixture path scores `environment: ""` on every alert,
    silently zeroing a 1.5-weight signal that the live path always sets."""
    assert alerts and all(a.environment == "production" for a in alerts)


def test_events_24h_never_falls_back_to_the_all_time_count(alerts):
    """Every issue in a real pull carries stats, so the `event_count` fallback
    in `events_24h()` should never be reached -- if it were, every volume
    number in the run would silently become the lifetime total."""
    for alert in alerts:
        assert alert.hourly_counts, f"{alert.short_id} has no 24h stats"
        # The window can never exceed the lifetime. Equality is legitimate and
        # expected for an issue first seen inside the window.
        assert alert.events_24h() <= alert.event_count, alert.short_id


def test_prefilter_thresholds_on_the_window_not_the_lifetime_count(alerts):
    """min_event_count is documented as "fewer events than this in the window".
    With a threshold of 50, the issue with 206,017 lifetime events but 223 in
    the window survives, while one with 3,025 lifetime and 1 in the window is
    dropped -- the opposite of what a `count` comparison does."""
    prefilter = Prefilter({"min_event_count": 50})
    kept, _dropped = prefilter.apply(alerts)
    kept_ids = {a.short_id for a in kept}
    assert "DUMMY-PROJ-1" in kept_ids  # 206,017 lifetime / 223 in window
    assert "DUMMY-PROJ-4" not in kept_ids  # 3,025 lifetime / 1 in window


def test_the_model_is_told_which_period_each_count_covers(alerts):
    payload = _payload(alerts[0])
    assert "event_count" not in payload
    assert payload["event_count_all_time"] >= payload["events_24h"]


# --------------------------------------------------------------------------
# What the agents actually see, on real-shaped data
# --------------------------------------------------------------------------


def test_the_selection_payload_carries_the_signals_the_rules_name(alerts):
    """The prompt's rules cite specific fields by name. A field the rules
    reason about but the payload never sends is a rule that silently does
    nothing -- and it looks exactly like a working rule from the outside."""
    payload = selection_payload(alerts[0])
    for signal in (
        "user_count", "events_24h", "event_count_all_time", "environment",
        "substatus", "is_unhandled", "hourly_events", "masking_hits", "similar_past",
    ):
        assert signal in payload, f"the selection prompt reasons about {signal} but never sends it"


def test_lifetime_and_24h_counts_are_sent_as_separate_named_fields(alerts):
    """A real issue in this pull reads 206,017 lifetime events next to 223 in
    the last 24 hours. Handing a model a bare `event_count` beside `events_24h`
    invites it to anchor on the larger number, so the name says the period."""
    loud = max(alerts, key=lambda a: a.event_count)
    payload = selection_payload(loud)
    assert payload["event_count_all_time"] > payload["events_24h"] * 10
    assert "event_count" not in payload


def test_near_identical_errors_embed_closer_than_unrelated_ones(alerts):
    """The premise the whole dedup rests on. This pull contains two
    ConnectionError issues against the same internal host and, separately, two
    unrelated errors -- the first pair has to be measurably closer than the
    second, or `similar_past` is showing the model noise.

    The offline HashEmbedder is lexical, not semantic, so this is a floor:
    Vertex embeddings should do better, never worse.
    """
    by_id = {a.short_id: a for a in alerts}
    pair = ("DUMMY-PROJ-9", "DUMMY-PROJ-14")   # same ConnectionError, same host
    unrelated = ("DUMMY-PROJ-9", "DUMMY-PROJ-12")

    vectors = dict(
        zip(
            [*pair, unrelated[1]],
            HashEmbedder().embed([by_id[s].embedding_text() for s in [*pair, unrelated[1]]]),
        )
    )
    same = cosine_distance(vectors[pair[0]], vectors[pair[1]])
    different = cosine_distance(vectors[unrelated[0]], vectors[unrelated[1]])
    assert same < different / 2, f"same-error distance {same:.3f} vs unrelated {different:.3f}"


def test_an_http_403_is_visible_to_the_model_as_an_auth_failure(alerts):
    """The only auth-shaped issue in this pull is titled "Request failed with
    status code 403" -- no "forbidden", no "unauthorized". A regex-based
    security signal missed it entirely, which is one of the reasons the
    security judgement is now the model's: it reads 403 as an auth failure
    without being told a pattern for it. The pipeline's job is to make sure the
    text reaches the model at all."""
    issue = next(a for a in alerts if a.short_id == "DUMMY-PROJ-2")
    payload = selection_payload(issue)
    assert "403" in f"{payload['title']} {payload['body']}"
