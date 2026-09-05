"""The LLM transport: retries, truncation, and schema validation.

Everything else in the suite talks to MockProvider, so this is the only place
the real `VertexProvider` code path is exercised. Its failure modes all end up
in the same place if they are not handled — an unexplained "Invalid JSON" two
layers up — which is why each one is separated out here.

The google-genai client is replaced with a stub after construction, so no
network, no credentials, and no GCP project.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from pipeline.agents.providers import (
    MAX_OUTPUT_TOKENS,
    LLMError,
    MockProvider,
    VertexProvider,
    build_provider,
    extract_input_json,
)
from shared.models import Priority, TriageOutput

VALID = (
    '[{"alert_id": "abc", "summary": "s", "component": "c", "suspected_cause": "u",'
    ' "security_relevant": false, "security_rationale": "none", "priority": "high",'
    ' "decision": "notify", "reasoning": "r", "clean_title": "t"}]'
)


class StubModels:
    """Stands in for `client.models`. `script` is consumed one call at a time;
    an entry may be a response object or an exception to raise."""

    def __init__(self, script: list):
        self.script = list(script)
        self.calls: list[dict] = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        item = self.script.pop(0) if self.script else self.script
        if isinstance(item, Exception):
            raise item
        return item


def response(text: str | None, finish_reason: str = "STOP", usage=None):
    return SimpleNamespace(
        text=text,
        candidates=[SimpleNamespace(finish_reason=finish_reason)],
        usage_metadata=usage,
    )


def provider_over(script: list) -> tuple[VertexProvider, StubModels]:
    p = VertexProvider.__new__(VertexProvider)   # skip __init__: it builds a real client
    stub = StubModels(script)
    p._client = SimpleNamespace(models=stub)
    p._model = "gemini-test"
    return p, stub


# --------------------------------------------------------------------------
# The happy path, and what it sends
# --------------------------------------------------------------------------


def test_parses_a_valid_structured_response():
    p, _ = provider_over([response(VALID)])
    out = p.generate_list("sys", "prompt", TriageOutput)
    assert len(out) == 1 and out[0].priority == Priority.high


def test_the_request_pins_json_the_schema_and_a_low_temperature():
    p, stub = provider_over([response(VALID)])
    p.generate_list("sys", "prompt", TriageOutput)

    config = stub.calls[0]["config"]
    assert config.response_mime_type == "application/json"
    assert config.system_instruction == "sys"
    assert config.temperature == 0.1
    # Without an explicit ceiling, a truncated reply arrives as an unexplained
    # parse failure rather than as truncation.
    assert config.max_output_tokens == MAX_OUTPUT_TOKENS


# --------------------------------------------------------------------------
# Failure modes that would otherwise all look like "Invalid JSON"
# --------------------------------------------------------------------------


def test_an_empty_response_is_reported_with_its_finish_reason():
    """Truncation and a safety block both return no text. Saying which is the
    difference between a one-minute diagnosis and an afternoon."""
    p, _ = provider_over([response(None, finish_reason="MAX_TOKENS")] * 4)
    with pytest.raises(LLMError) as exc:
        p.generate_list("sys", "prompt", TriageOutput)
    assert "MAX_TOKENS" in str(exc.value)


def test_an_empty_response_is_not_retried_as_a_transport_error():
    """It is deterministic: the same prompt reproduces it, so retrying only
    burns the job's timeout."""
    p, stub = provider_over([response(None)] * 6)
    with pytest.raises(LLMError):
        p.generate_list("sys", "prompt", TriageOutput)
    # One schema-retry loop, no transport retries inside it.
    assert len(stub.calls) == 2


def test_malformed_json_gets_exactly_one_retry_then_gives_up():
    p, stub = provider_over([response("not json at all"), response("still not json")])
    with pytest.raises(LLMError):
        p.generate_list("sys", "prompt", TriageOutput)
    assert len(stub.calls) == 2


def test_a_retry_that_succeeds_is_returned():
    """The case the retry exists for: one bad decode, then a good one."""
    p, stub = provider_over([response("{oops"), response(VALID)])
    out = p.generate_list("sys", "prompt", TriageOutput)
    assert len(out) == 1 and len(stub.calls) == 2


def test_json_that_violates_the_schema_is_rejected():
    """A response the model invented a priority for must not reach the audit
    trail as though it were valid."""
    bad = VALID.replace('"priority": "high"', '"priority": "extremely urgent"')
    p, _ = provider_over([response(bad), response(bad)])
    with pytest.raises(LLMError):
        p.generate_list("sys", "prompt", TriageOutput)


def test_a_transport_error_is_retried_then_surfaces_as_an_llm_error():
    p, stub = provider_over([ConnectionError("reset")] * 8)
    with pytest.raises(LLMError):
        p.generate_list("sys", "prompt", TriageOutput)
    # 2 transport attempts inside each of 2 schema attempts.
    assert len(stub.calls) == 4


def test_a_transport_error_that_clears_on_retry_succeeds():
    p, _ = provider_over([ConnectionError("reset"), response(VALID)])
    assert len(p.generate_list("sys", "prompt", TriageOutput)) == 1


def test_usage_metadata_is_tolerated_when_absent():
    """Cost logging must never be the thing that breaks a run."""
    p, _ = provider_over([response(VALID, usage=None)])
    assert p.generate_list("sys", "prompt", TriageOutput)


def test_usage_metadata_is_read_when_present(caplog):
    usage = SimpleNamespace(prompt_token_count=1234, candidates_token_count=56)
    p, _ = provider_over([response(VALID, usage=usage)])
    with caplog.at_level("INFO"):
        p.generate_list("sys", "prompt", TriageOutput)
    assert "prompt_tokens=1234" in caplog.text


# --------------------------------------------------------------------------
# Prompt plumbing and the factory
# --------------------------------------------------------------------------


def test_extract_input_json_reads_the_last_marker():
    """The marker string can legitimately appear inside an alert body; the real
    payload is always the last one."""
    prompt = 'INPUT_JSON:\n[{"a": 1}]\nINPUT_JSON:\n[{"b": 2}]'
    assert extract_input_json(prompt) == [{"b": 2}]


def test_a_prompt_without_a_marker_is_an_error():
    with pytest.raises(ValueError):
        extract_input_json("no marker here")


def test_build_provider_rejects_an_unknown_name():
    with pytest.raises(ValueError):
        build_provider("openai", "p", "l", "m")


def test_gemini_transport_requires_a_key():
    """Falling back to ADC here would silently switch transports and bill a
    project the caller did not mean to use."""
    with pytest.raises(ValueError):
        build_provider("gemini", "p", "l", "m", api_key="")


def test_build_provider_returns_the_mock_without_touching_the_network():
    assert isinstance(build_provider("mock", "", "", ""), MockProvider)


def test_the_mock_rejects_a_schema_it_cannot_serve():
    """Silently returning [] would look like 'the model omitted everything'."""
    from pydantic import BaseModel

    class Unknown(BaseModel):
        x: int = 0

    with pytest.raises(LLMError):
        MockProvider().generate_list("sys", 'INPUT_JSON:\n[{"alert_id": "a"}]', Unknown)
