"""The masker against the shapes real payloads actually contain.

`tests/test_masking.py` checks the rules on strings written to exercise them.
This file checks them on the shapes that turn up in production error text, and
— just as importantly — on the shapes that must NOT be masked.

Precision matters as much as recall. A credential hit is treated as a security
finding: it drives the report's "applications logging secrets" callout and the
top-issues agent is told to rank it above ordinary errors at any volume. A rule
that fires on a UUID or a git SHA does not merely over-redact, it manufactures
a security incident out of a version string.

Every rule below was added after probing the masker with these inputs and
watching it miss. See FINDINGS.md.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline.masking import Masker
from pipeline.sentry_client import FixtureSentryClient

ORG_FIXTURE = Path("fixtures/sentry_org_issues_last_24h_anonymized.json")


@pytest.fixture(scope="module")
def masker() -> Masker:
    return Masker()


# --------------------------------------------------------------------------
# Secrets that appear in real error text
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,rule,must_not_survive",
    [
        # Connection failures are bread-and-butter Sentry issues, and the DSN
        # goes into the message with the password attached.
        ("could not connect to postgres://svc_user:s3cr3tP4ss@db-01:5432/prod",
         "connection-string-credentials", "s3cr3tP4ss"),
        ("amqp://rabbit:Hunter2Hunter2@broker:5672 refused",
         "connection-string-credentials", "Hunter2Hunter2"),
        ("GET https://admin:hunter22@internal/api failed",
         "connection-string-credentials", "hunter22"),
        # `eyJ` is base64 of `{"`, so this is a JWT and essentially nothing else.
        ("auth failed for eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjMifQ.abc123def456",
         "jwt", "eyJzdWIiOiIxMjMifQ"),
        # The signature IS the credential: it reads the object until it expires.
        ("https://b.s3.amazonaws.com/k?X-Amz-Signature=9f8a7b6c5d4e3f2a1b0c",
         "url-signature", "9f8a7b6c5d4e3f2a1b0c"),
        ("xoxb-2401234567-abcdefghijklmno", "prefixed-vendor-token", "abcdefghijklmno"),
        ("token ghp_A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8", "prefixed-vendor-token",
         "A1b2C3d4E5f6G7h8I9j0"),
        ("Cookie: sessionid=8f3ac91b2e7d4a56", "session-identifier", "8f3ac91b2e7d4a56"),
        ("-----BEGIN RSA PRIVATE KEY-----MIIEpAIBAAKCAQEA", "private-key-block",
         "MIIEpAIBAAKCAQEA"),
    ],
)
def test_a_real_world_secret_shape_is_masked(masker, text, rule, must_not_survive):
    masked, hits = masker.mask_text(text)
    assert rule in hits, f"{rule} did not fire on {text!r}"
    assert must_not_survive not in masked, "the secret survived masking"


def test_the_username_survives_a_masked_connection_string(masker):
    """Which account failed to authenticate is most of the debugging value, and
    the username is not the secret."""
    masked, _ = masker.mask_text("postgres://svc_user:s3cr3t@db-01/prod refused")
    assert "svc_user" in masked and "db-01" in masked


def test_a_windows_share_path_masks_only_the_username(masker):
    """A studio runs Windows workstations; scene files live on UNC shares."""
    masked, hits = masker.mask_text(r"\\RENDER01\Users\ada.testperson\maya\shot_010.ma")
    assert "home-directory-path" in hits
    assert "ada.testperson" not in masked
    assert "maya" in masked and "shot_010.ma" in masked


# --------------------------------------------------------------------------
# Shapes that must NOT be masked
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "upgrade to 1.2.3-beta.4 required",
        "job 550e8400-e29b-41d4-a716-446655440000 failed",
        "at commit a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0",
        "Invalid token: expired",
        "Request failed with status code 403",
        "/usr/lib/python3.12/socket.py",
        "./node_modules/flux/lib/Dispatcher.js",
        "GET https://api.example.com/v2/assets?page=3 returned 502",
        "redis://cache-01:6379/0 timed out",
        r"\\fileserver\projects\show\shot_010.ma",
        "ModuleNotFoundError: No module named 'mytime_events'",
        "DeprecationWarning: imp module is deprecated",
    ],
)
def test_ordinary_error_text_is_not_a_security_finding(masker, text):
    """These fire the report's 'applications logging secrets' callout if a rule
    is too greedy, which turns a version string into an incident."""
    masked, hits = masker.mask_text(text)
    assert hits == [], f"{text!r} was flagged by {hits}"
    assert masked == text


def test_a_url_without_credentials_keeps_its_host(masker):
    """The connection-string rule needs a `user:pass@`; a plain URL must pass
    through, or every HTTP error in the report becomes unreadable."""
    text = "POST https://internal.example.com/api/submit returned 500"
    assert masker.mask_text(text) == (text, [])


# --------------------------------------------------------------------------
# Properties that have to hold on any input
# --------------------------------------------------------------------------


def test_masking_is_idempotent(masker):
    """The pipeline masks a body more than once — it grows when event detail is
    folded in. A rule that rewrites its own output would churn the alert's text
    every run, which changes its embedding and quietly breaks dedup."""
    text = ("postgres://u:p4ssw0rd123@db/prod failed for ada@example.com from 10.4.2.19 "
            "in /home/ada/app.py with eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ4In0.sig12345")
    once, first_hits = masker.mask_text(text)
    twice, second_hits = masker.mask_text(once)
    assert once == twice
    assert set(second_hits) <= set(first_hits)


def test_every_rule_in_the_config_is_exercised_by_some_test():
    """A rule nobody tests is a rule nobody knows still works. This fails when
    someone adds a pattern without a case for it."""
    import yaml

    configured = {r["name"] for r in yaml.safe_load(
        Path("config/masking_patterns.yaml").read_text(encoding="utf-8"))["patterns"]}
    tested = set()
    for path in Path("tests").glob("test_masking*.py"):
        text = path.read_text(encoding="utf-8")
        tested |= {name for name in configured if name in text}
    assert configured - tested == set(), f"untested masking rules: {configured - tested}"


def test_masking_never_silently_empties_a_body(masker):
    """Over-masking to nothing would leave the agents with no error to read at
    all, which is worse than a redacted one."""
    text = "Bearer abcd1234efgh5678 rejected by /home/ada/app.py for ada@example.com"
    masked, _ = masker.mask_text(text)
    assert "rejected by" in masked and len(masked) > 20


# --------------------------------------------------------------------------
# Against the real captured payload
# --------------------------------------------------------------------------


@pytest.mark.skipif(not ORG_FIXTURE.exists(), reason="org-pull fixture not present")
def test_the_real_org_pull_leaves_no_unmasked_secrets(masker):
    """End to end on a real 24h pull: after masking, nothing in any alert should
    still look like a credential, an email, an IP or a home path."""
    import re

    alerts = FixtureSentryClient(str(ORG_FIXTURE), "production").fetch_issues(24)
    masked = masker.mask_alerts(alerts)
    blob = " ".join(f"{a.title} {a.body}" for a in masked)

    leaks = {
        "email": r"[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}",
        "ipv4": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
        "home path": r"(?:/home/|/Users/)[^/\s\[]+",
        "bearer": r"(?i)bearer\s+[a-z0-9._\-]{8,}",
        "jwt": r"\beyJ[A-Za-z0-9_\-]{8,}\.",
    }
    for label, pattern in leaks.items():
        found = [m for m in re.findall(pattern, blob) if "MASKED" not in m]
        assert not found, f"unmasked {label} survived the real pull: {found[:3]}"


@pytest.mark.skipif(not ORG_FIXTURE.exists(), reason="org-pull fixture not present")
def test_masking_the_real_pull_is_stable_across_runs(masker):
    """Same input, same output — the dedup vectors depend on it."""
    alerts = FixtureSentryClient(str(ORG_FIXTURE), "production").fetch_issues(24)
    first = [a.body for a in masker.mask_alerts(alerts)]
    second = [a.body for a in masker.mask_alerts(masker.mask_alerts(alerts))]
    assert first == second


@pytest.mark.skipif(not ORG_FIXTURE.exists(), reason="org-pull fixture not present")
def test_the_real_pull_does_not_trip_the_security_signal(masker):
    """This pull was anonymized before it was committed, so a credential hit
    here means a rule is firing on ordinary VFX-pipeline error prose — the
    false-positive direction that manufactures security findings."""
    from pipeline.agents.providers import _CREDENTIAL_HITS

    alerts = masker.mask_alerts(
        FixtureSentryClient(str(ORG_FIXTURE), "production").fetch_issues(24))
    fired = {h for a in alerts for h in a.masking_hits} & _CREDENTIAL_HITS
    assert not fired, f"credential rules fired on anonymized prose: {fired}"


def test_the_org_fixture_is_json_we_can_actually_read():
    """Guards the two tests above from silently skipping forever."""
    assert ORG_FIXTURE.exists(), "the org-pull fixture has gone missing"
    assert isinstance(json.loads(ORG_FIXTURE.read_text(encoding="utf-8")), list)
