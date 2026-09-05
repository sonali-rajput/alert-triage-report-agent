"""Flattening a Sentry event into the text the agents read.

This coverage moved here from the deleted enrichment tests. It matters more now
than it did then: every issue gets its event folded in, so a formatting bug no
longer degrades a top-N narrative — it degrades the input to the ranking
decision for the entire run.

Event payloads vary enormously by platform, so most of these tests are about
not exploding on shapes a real SDK produces.
"""

from __future__ import annotations

from pipeline.event_text import (
    MAX_BREADCRUMBS,
    MAX_FRAMES,
    breadcrumb_lines,
    event_to_body_text,
    stack_trace_lines,
)


def exception_event(frames: list[dict], type_="ValueError", value="boom") -> dict:
    return {
        "entries": [
            {
                "type": "exception",
                "data": {"values": [{"type": type_, "value": value,
                                     "stacktrace": {"frames": frames}}]},
            }
        ]
    }


def frame(name: str, **kwargs) -> dict:
    return {"filename": f"app/{name}.py", "function": name, "lineNo": 10, **kwargs}


# --------------------------------------------------------------------------
# Stack traces
# --------------------------------------------------------------------------


def test_renders_the_exception_header_and_frames():
    lines = stack_trace_lines(exception_event([frame("handler", inApp=True)]))
    assert lines[0] == "Exception: ValueError: boom"
    assert "app/handler.py:10 in handler" in lines[1]


def test_in_app_frames_are_marked():
    ours = stack_trace_lines(exception_event([frame("ours", inApp=True)]))[1]
    theirs = stack_trace_lines(exception_event([frame("theirs", inApp=False)]))[1]
    assert ours.strip().startswith("*")
    assert not theirs.strip().startswith("*")


def test_keeps_the_innermost_frames_not_the_outermost():
    """Sentry orders frames oldest-first, so the error actually happened at the
    END of the list. Truncating from the wrong end throws away the only part
    anyone reads."""
    frames = [frame(f"f{i}") for i in range(MAX_FRAMES + 5)]
    text = "\n".join(stack_trace_lines(exception_event(frames)))
    assert f"f{MAX_FRAMES + 4}" in text          # innermost, kept
    assert "f0" not in text                       # outermost, dropped
    assert "5 outer frames omitted" in text


def test_a_frame_without_a_line_number_still_renders():
    lines = stack_trace_lines(exception_event([{"filename": "app/x.py", "function": "go"}]))
    assert "app/x.py in go" in lines[1]


def test_a_frame_with_no_filename_falls_back_to_the_module():
    lines = stack_trace_lines(exception_event([{"module": "app.tasks", "function": "run"}]))
    assert "app.tasks in run" in lines[1]


def test_a_frame_with_nothing_useful_does_not_crash():
    lines = stack_trace_lines(exception_event([{}]))
    assert "<unknown> in <unknown>" in lines[1]


def test_multiple_chained_exceptions_all_render():
    event = {
        "entries": [{
            "type": "exception",
            "data": {"values": [
                {"type": "OSError", "value": "refused", "stacktrace": {"frames": [frame("sock")]}},
                {"type": "RetryError", "value": "gave up", "stacktrace": {"frames": [frame("retry")]}},
            ]},
        }]
    }
    text = "\n".join(stack_trace_lines(event))
    assert "OSError: refused" in text and "RetryError: gave up" in text


def test_an_exception_with_no_stacktrace_still_gives_its_header():
    event = {"entries": [{"type": "exception", "data": {"values": [{"type": "E", "value": "v"}]}}]}
    assert stack_trace_lines(event) == ["Exception: E: v"]


def test_no_exception_entry_gives_nothing():
    assert stack_trace_lines({"entries": [{"type": "request", "data": {}}]}) == []


# --------------------------------------------------------------------------
# Breadcrumbs
# --------------------------------------------------------------------------


def crumbs_event(values: list[dict]) -> dict:
    return {"entries": [{"type": "breadcrumbs", "data": {"values": values}}]}


def test_breadcrumbs_render_newest_last():
    lines = breadcrumb_lines(crumbs_event([
        {"category": "http", "level": "info", "message": "first"},
        {"category": "db", "level": "warning", "message": "last"},
    ]))
    assert lines[0].startswith("Breadcrumbs")
    assert "first" in lines[1] and "last" in lines[2]


def test_only_the_most_recent_breadcrumbs_are_kept():
    values = [{"category": "c", "message": f"m{i}"} for i in range(MAX_BREADCRUMBS + 5)]
    text = "\n".join(breadcrumb_lines(crumbs_event(values)))
    assert f"m{MAX_BREADCRUMBS + 4}" in text
    assert "m0" not in text


def test_a_breadcrumb_with_no_message_falls_back_to_its_data():
    lines = breadcrumb_lines(crumbs_event([{"category": "http", "data": {"url": "/submit"}}]))
    assert "/submit" in lines[1]


def test_a_very_long_breadcrumb_is_truncated():
    lines = breadcrumb_lines(crumbs_event([{"category": "c", "message": "x" * 900}]))
    assert len(lines[1]) <= 300


def test_empty_breadcrumb_values_give_nothing():
    assert breadcrumb_lines(crumbs_event([])) == []
    assert breadcrumb_lines(crumbs_event(["not a dict"])) == []  # type: ignore[list-item]


# --------------------------------------------------------------------------
# The whole event
# --------------------------------------------------------------------------


def test_stack_trace_comes_before_breadcrumbs():
    event = {
        "entries": [
            {"type": "breadcrumbs", "data": {"values": [{"category": "c", "message": "crumb"}]}},
            {"type": "exception", "data": {"values": [
                {"type": "E", "value": "v", "stacktrace": {"frames": [frame("h")]}}]}},
        ]
    }
    text = event_to_body_text(event)
    assert text.index("Exception: E") < text.index("crumb")


def test_junk_payloads_return_empty_rather_than_raising():
    """A malformed event must cost one alert its stack trace, never the run."""
    for junk in (None, "a string", 42, [], {}, {"entries": None}, {"entries": ["x"]},
                 {"entries": [{"type": "exception", "data": None}]},
                 {"entries": [{"type": "exception", "data": {"values": None}}]}):
        assert event_to_body_text(junk) == ""  # type: ignore[arg-type]
