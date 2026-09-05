"""Regression tests against a REAL captured `events/latest/` payload.

`fixtures/sentry_event_latest.sample.json` is an actual event response, run
through the allowlist projection in `scripts/capture_sentry_payloads.py`. It is
the first real event this codebase has ever seen: until it exists, every test
of the detail path uses events I wrote by hand, which means they test my
assumptions about the shape rather than the shape.

These skip until the fixture is captured. See RUNBOOK.md level 4.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline.event_text import event_to_body_text
from pipeline.masking import Masker
from pipeline.sentry_client import FixtureSentryClient

FIXTURE = Path("fixtures/sentry_event_latest.sample.json")

pytestmark = pytest.mark.skipif(
    not FIXTURE.exists(),
    reason="no real event fixture captured yet (scripts/capture_sentry_payloads.py --make-event-fixture)",
)


@pytest.fixture(scope="module")
def event() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_a_real_event_yields_text_for_the_agents(event):
    """Q5 of the capture probe said 8/8 real events produced text. This pins
    that for the one we kept."""
    body = event_to_body_text(event)
    assert body.strip(), "the real payload produced nothing — the shape is not what we parse"


def test_a_real_event_yields_a_stack_trace(event):
    assert "Exception:" in event_to_body_text(event)


def test_frames_render_with_somewhere_to_look(event):
    """A frame that renders as `<unknown> in <unknown>` is worse than no frame:
    it costs tokens and tells the reader nothing."""
    frames = [ln for ln in event_to_body_text(event).splitlines() if " in " in ln]
    assert frames, "no frame lines at all"
    useless = [f for f in frames if "<unknown> in <unknown>" in f]
    assert len(useless) < len(frames), f"every frame is unreadable: {frames[:3]}"


def test_the_fixture_carries_no_obvious_pii(event):
    """The projection is an allowlist, but a fixture is committed forever, so
    this is the cheap standing check that it stayed clean."""
    blob = json.dumps(event)
    for marker in ("@", "gravatar", "/home/", "/Users/", "Authorization", "Cookie"):
        if marker in blob:
            # Masked values are the rules working, not a leak.
            assert "[MASKED_" in blob, f"unmasked {marker!r} in the committed fixture"


def test_the_fixture_holds_no_request_entry(event):
    """Request entries carry headers and cookies. The projection drops them;
    this fails if someone regenerates the fixture without it."""
    types = {e.get("type") for e in event.get("entries") or []}
    assert "request" not in types


def test_frames_carry_no_local_variables(event):
    """Frame `vars` are the local variables at the point of failure — the most
    likely place in a whole payload to find a live credential."""
    for entry in event.get("entries") or []:
        for value in (entry.get("data") or {}).get("values") or []:
            for frame in ((value.get("stacktrace") or {}).get("frames") or []):
                assert "vars" not in frame and "context" not in frame


def test_the_offline_pipeline_can_use_it_as_detail(tmp_path, event):
    """The end the fixture exists for: SENTRY_FIXTURE_DETAIL_PATH pointing at
    it gives an offline run bodies shaped like production's."""
    issues = tmp_path / "issues.json"
    issues.write_text(json.dumps([{"id": "1", "level": "error"}]), encoding="utf-8")

    alerts = FixtureSentryClient(str(issues), "production", str(FIXTURE)).fetch_issues(24)

    assert len(alerts) == 1
    assert "Exception:" in alerts[0].body


def test_masking_the_real_body_is_stable(event):
    """Masking a body that has already been masked must not keep changing it —
    the pipeline masks after folding in detail, and a rule that rewrites its
    own output would churn the vector every run and break dedup."""
    masker = Masker()
    once, _ = masker.mask_text(event_to_body_text(event))
    twice, _ = masker.mask_text(once)
    assert once == twice


def test_the_real_event_has_chained_exceptions(event):
    """What a real payload turned out to contain that no hand-written test
    event did: FOUR chained exceptions (3, 9, 3 and 9 frames), not one stack.

    Every synthetic event in this suite has a single `values` entry, so the
    loop over multiple exceptions was never exercised on real data. A "cause"
    chain is the normal shape for wrapped errors, and the LAST exception is
    usually the one that actually broke."""
    values = [
        value
        for entry in event.get("entries") or []
        for value in (entry.get("data") or {}).get("values") or []
        if entry.get("type") == "exception"
    ]
    assert len(values) > 1, "this fixture no longer covers chained exceptions"

    body = event_to_body_text(event)
    assert body.count("Exception:") == len(values), "a link in the chain was dropped"


def test_frame_truncation_is_per_exception_not_across_the_whole_event(event):
    """This fixture has more frames in total than MAX_FRAMES, but no single
    exception exceeds it, so nothing should be omitted.

    That distinction is load-bearing. Capping frames across the whole event --
    an easy "optimisation" -- would silently drop the innermost exception,
    which is the one that says what actually failed."""
    from pipeline.event_text import MAX_FRAMES

    per_value = [
        len((value.get("stacktrace") or {}).get("frames") or [])
        for entry in event.get("entries") or []
        for value in (entry.get("data") or {}).get("values") or []
    ]
    assert sum(per_value) > MAX_FRAMES, "the fixture no longer has enough frames to matter"
    assert max(per_value) <= MAX_FRAMES

    body = event_to_body_text(event)
    assert "outer frames omitted" not in body, "frames were dropped that should have fit"
    assert len([ln for ln in body.splitlines() if " in " in ln]) >= sum(per_value)


def test_the_real_event_carries_breadcrumbs(event):
    assert "Breadcrumbs (most recent last):" in event_to_body_text(event)
