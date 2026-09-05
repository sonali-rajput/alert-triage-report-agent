"""The fixture projection in scripts/capture_sentry_payloads.py.

This is the last thing standing between a real Sentry event and a git commit.
Everything else in the repo can be fixed by editing a file; a payload committed
with a colleague's email in it is in the history forever.

The projection is an ALLOWLIST — it keeps the fields the pipeline reads and
discards the rest — for the same reason `issue_to_alert` is: a denylist only
protects you from the PII you thought of. These tests are written the way that
implies: they assert what SURVIVES, and they feed it an event stuffed with
everything a real one carries.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from pipeline.event_text import event_to_body_text
from pipeline.masking import Masker

SCRIPT = Path("scripts/capture_sentry_payloads.py")


@pytest.fixture(scope="module")
def capture():
    spec = importlib.util.spec_from_file_location("capture_module", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def nasty_event() -> dict:
    """Everything a real event payload carries that a fixture must not."""
    return {
        "id": "6316939353",
        "eventID": "deadbeefdeadbeef",
        "groupID": "77",
        "title": "ConnectionError",
        "platform": "python",
        "dateCreated": "2026-08-18T09:00:00Z",
        "user": {"id": "3416073", "email": "ada.testperson@framestore.com",
                 "ip_address": "10.4.2.19", "username": "ada.testperson",
                 "name": "Ada Testperson"},
        "contexts": {"device": {"name": "ADA-WORKSTATION-07"}},
        "sdk": {"name": "sentry.python", "version": "1.2.3"},
        "packages": {"certifi": "2024.2.2"},
        "tags": [
            {"key": "environment", "value": "production"},
            {"key": "level", "value": "error"},
            {"key": "server_name", "value": "render-node-04.internal.framestore.com"},
            {"key": "user", "value": "id:3416073"},
            {"key": "url", "value": "https://internal.framestore.com/submit?token=abc12345"},
            {"key": "machine_id", "value": "9f2b"},
        ],
        "entries": [
            {"type": "request", "data": {
                "url": "https://internal.framestore.com/api/submit",
                "headers": [["Authorization", "Bearer sk_live_9f8a7b6c5d4e"],
                            ["Cookie", "sessionid=8f3ac91b2e7d4a56"]],
                "data": {"password": "hunter2hunter2"}}},
            {"type": "exception", "data": {"values": [{
                "type": "ConnectionError",
                "value": "could not reach db as ada.testperson@framestore.com",
                "stacktrace": {"frames": [
                    {"filename": "/home/ada.testperson/src/app/db.py", "function": "connect",
                     "lineNo": 42, "inApp": True,
                     "vars": {"password": "hunter2hunter2", "api_key": "sk_live_9f8a7b6c"},
                     "context": [[41, "    token = 'sk_live_9f8a7b6c'"]]},
                    {"filename": "app/pool.py", "function": "acquire", "lineNo": 88,
                     "inApp": True}]}}]}},
            {"type": "breadcrumbs", "data": {"values": [
                {"category": "http", "level": "warning", "timestamp": "2026-08-18T08:59:00Z",
                 "message": "GET https://internal/x failed for ada.testperson@framestore.com",
                 "data": {"url": "https://internal/x?token=abc12345", "user": "ada.testperson"}}]}},
        ],
    }


@pytest.fixture
def projected(capture, nasty_event) -> dict:
    return capture._project_event(nasty_event, Masker())


def blob(payload: dict) -> str:
    return json.dumps(payload)


# --------------------------------------------------------------------------
# What must not survive
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "secret,what",
    [
        ("sk_live_9f8a7b6c5d4e", "an Authorization header from a request entry"),
        ("sessionid=8f3ac91b2e7d4a56", "a session cookie"),
        ("hunter2hunter2", "a password in frame-local variables"),
        ("sk_live_9f8a7b6c", "a token in a frame's source context"),
        ("ada.testperson@framestore.com", "an email address"),
        ("10.4.2.19", "a user's IP"),
        ("ADA-WORKSTATION-07", "a workstation name from contexts"),
        ("render-node-04.internal.framestore.com", "an internal hostname from a tag"),
        ("id:3416073", "a user id from a tag"),
    ],
)
def test_the_projection_drops_what_a_real_event_carries(projected, secret, what):
    assert secret not in blob(projected), f"{what} survived into the fixture"


def test_no_request_entry_survives(projected):
    """Request entries hold headers, cookies and posted bodies. There is no
    version of them worth keeping in a committed fixture."""
    assert {e["type"] for e in projected["entries"]} <= {"exception", "breadcrumbs"}


def test_no_frame_keeps_its_local_variables(projected):
    """`vars` is the single most likely place in a payload to hold a live
    credential: it is the state at the moment of failure."""
    for entry in projected["entries"]:
        for value in entry["data"].get("values", []):
            for frame in value.get("stacktrace", {}).get("frames", []):
                assert "vars" not in frame
                assert "context" not in frame


def test_no_breadcrumb_keeps_its_data(projected):
    """Breadcrumb `data` carries URLs, ids and request payloads."""
    for entry in projected["entries"]:
        if entry["type"] == "breadcrumbs":
            for crumb in entry["data"]["values"]:
                assert "data" not in crumb


def test_only_safe_tags_survive(projected):
    kept = {t["key"] for t in projected["tags"]}
    assert kept == {"environment", "level"}


def test_top_level_pii_objects_are_gone(projected):
    for key in ("user", "contexts", "sdk", "packages"):
        assert key not in projected


def test_identifiers_are_replaced_not_preserved(projected, nasty_event):
    assert projected["id"] != nasty_event["id"]
    assert projected["eventID"] != nasty_event["eventID"]


# --------------------------------------------------------------------------
# What must survive — a fixture nobody can use is not safe, it is useless
# --------------------------------------------------------------------------


def test_the_stack_trace_survives(projected):
    body = event_to_body_text(projected)
    assert "ConnectionError" in body
    assert "app/pool.py:88 in acquire" in body


def test_the_breadcrumb_survives(projected):
    assert "Breadcrumbs" in event_to_body_text(projected)


def test_the_home_path_keeps_its_shape_without_the_username(projected):
    body = event_to_body_text(projected)
    assert "ada.testperson" not in body
    assert "src/app/db.py:42 in connect" in body


def test_the_environment_tag_survives(projected):
    """Q1 of the capture probe exists to answer whether events carry this. A
    fixture that dropped it could not be used to re-test that."""
    assert {"key": "environment", "value": "production"} in projected["tags"]


def test_the_projection_is_idempotent(capture, projected):
    """Re-projecting a fixture must not degrade it — someone will do this by
    accident when regenerating."""
    assert capture._project_event(projected, Masker()) == projected


# --------------------------------------------------------------------------
# Robustness: real payloads vary by platform and a crash here is a lost capture
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "event",
    [
        {},
        {"entries": None},
        {"entries": [{"type": "exception", "data": None}]},
        {"entries": [{"type": "exception", "data": {"values": None}}]},
        {"entries": [{"type": "exception", "data": {"values": ["not a dict"]}}]},
        {"entries": [{"type": "breadcrumbs", "data": {"values": [None]}}]},
        {"entries": [{"type": "exception", "data": {"values": [
            {"type": "E", "value": "v", "stacktrace": None}]}}]},
        {"tags": [{"key": "environment"}, "not a dict", None]},
    ],
)
def test_a_malformed_event_does_not_crash_the_projection(capture, event):
    result = capture._project_event(event, Masker())
    assert isinstance(result, dict) and "entries" in result


# --------------------------------------------------------------------------
# The residual scan
# --------------------------------------------------------------------------


def test_the_residual_scan_knows_the_patterns_that_matter(capture):
    labels = " ".join(capture._RESIDUAL_PII)
    for expected in ("email", "gravatar", "IPv4", "host", "home directory"):
        assert expected in labels


def test_the_residual_scan_ignores_values_the_masker_already_redacted(capture):
    """Reporting `[MASKED_USER]` as a finding trains the reader to skim past
    the section, which is the last thing you want from a PII scan."""
    pattern = capture._RESIDUAL_PII["home directory (a username)"]
    hits = {h for h in pattern.findall("/home/[MASKED_USER]/app.py") if "[MASKED_" not in h}
    assert hits == set()


def test_the_residual_scan_still_catches_a_real_leak(capture):
    pattern = capture._RESIDUAL_PII["home directory (a username)"]
    assert pattern.findall("/home/ada.testperson/app.py")
