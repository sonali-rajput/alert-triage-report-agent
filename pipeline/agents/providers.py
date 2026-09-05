"""LLM providers behind a single structured-output interface.

VertexProvider talks to Vertex AI Gemini via the google-genai SDK with a
strict JSON response schema. MockProvider is a deterministic heuristic
implementation used for tests, local demos, and eval sanity checks.

Prompts embed their payload as a JSON array after an INPUT_JSON: marker so
that providers (and the mock) can rely on a stable structure.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Protocol, TypeVar

from pydantic import BaseModel, TypeAdapter
from tenacity import retry, retry_if_not_exception_type, stop_after_attempt, wait_exponential

from shared.models import (
    AlertSummary,
    Priority,
    SelectedIssue,
    TriageDecision,
    TriageOutput,
    TriageResult,
)

logger = logging.getLogger(__name__)

M = TypeVar("M", bound=BaseModel)

INPUT_MARKER = "INPUT_JSON:"

# Generous ceiling for one call. The largest prompt in the pipeline is a
# top-issues chunk of 25 alerts, whose reply is one short object each; the
# triage call is ten richer objects. Both are well under this. Without an
# explicit limit a truncated response surfaces as an unexplained "Invalid JSON"
# parse failure two layers up.
MAX_OUTPUT_TOKENS = 16384


class LLMError(RuntimeError):
    """Raised when the LLM stage fails permanently; triggers the fallback digest."""


class LLMProvider(Protocol):
    def generate_list(self, system: str, prompt: str, item_schema: type[M]) -> list[M]: ...


def extract_input_json(prompt: str) -> list[dict]:
    idx = prompt.rfind(INPUT_MARKER)
    if idx == -1:
        raise ValueError("prompt has no INPUT_JSON marker")
    return json.loads(prompt[idx + len(INPUT_MARKER) :].strip())


class VertexProvider:
    """Gemini through the google-genai SDK.

    Two transports, one code path. `api_key` selects the **Gemini Developer
    API** (an AI Studio key); without it the client uses **Vertex AI** with ADC,
    which is the production path. The request, the response schema and the
    parsing are identical, so a Developer-API key is a genuine smoke test of
    the production prompt -- it needs no GCP project, no ADC, and no billing
    setup, which is what makes it usable from a laptop.

    The two differ in *model availability*, not behaviour: the Developer API
    has already retired `gemini-2.5-flash` (404 "no longer available to new
    users") while Vertex still serves it. Set GEMINI_MODEL accordingly.
    """

    def __init__(self, project: str, location: str, model: str, api_key: str = ""):
        from google import genai

        if api_key:
            self._client = genai.Client(api_key=api_key)
            logger.info("gemini: Developer API transport (model %s)", model)
        else:
            self._client = genai.Client(vertexai=True, project=project or None, location=location)
            logger.info("gemini: Vertex AI transport (project %s, %s, model %s)", project, location, model)
        self._model = model

    # Two transport attempts, not four: combined with the schema-retry loop in
    # generate_list and the concurrent selection calls, the worst-case retry
    # tower has to fit inside the Cloud Run Job's task timeout. LLMError is excluded
    # because it marks a *deterministic* bad response (truncated, safety-blocked)
    # that a same-prompt replay will reproduce.
    @retry(
        retry=retry_if_not_exception_type(LLMError),
        wait=wait_exponential(multiplier=2, min=2, max=60),
        stop=stop_after_attempt(2),
        reraise=True,
    )
    def _call(self, system: str, prompt: str, item_schema: type[M]) -> str:
        from google.genai import types

        response = self._client.models.generate_content(
            model=self._model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system,
                response_mime_type="application/json",
                response_schema=list[item_schema],  # type: ignore[valid-type]
                temperature=0.1,
                max_output_tokens=MAX_OUTPUT_TOKENS,
            ),
        )

        # Cost is a stated design driver; usage was previously discarded.
        usage = getattr(response, "usage_metadata", None)
        if usage is not None:
            logger.info(
                "llm call: model=%s prompt_tokens=%s output_tokens=%s",
                self._model,
                getattr(usage, "prompt_token_count", None),
                getattr(usage, "candidates_token_count", None),
            )

        text = response.text or ""
        if not text:
            # Without this, truncation and safety blocks all flatten into an
            # unexplained "Invalid JSON" two layers up.
            candidates = getattr(response, "candidates", None) or []
            finish = getattr(candidates[0], "finish_reason", None) if candidates else None
            raise LLMError(f"empty LLM response (finish_reason={finish}); truncation or safety block")
        return text

    def generate_list(self, system: str, prompt: str, item_schema: type[M]) -> list[M]:
        adapter = TypeAdapter(list[item_schema])  # type: ignore[valid-type]
        last_error: Exception | None = None
        # Schema-validation failures get exactly one retry, then we give up
        # and let the caller degrade to the fallback digest.
        for attempt in range(2):
            try:
                raw = self._call(system, prompt, item_schema)
                return adapter.validate_json(raw)
            except Exception as exc:  # transport errors already retried in _call
                last_error = exc
                logger.warning("LLM structured output attempt %d failed: %s", attempt + 1, exc)
        raise LLMError(f"LLM call failed after retries: {last_error}")


_CRITICAL_RE = re.compile(
    r"outage|data loss|exhausted|all (users|sites|jobs)|cannot (login|authenticate)|down\b", re.IGNORECASE
)
_HIGH_RE = re.compile(r"production|main branch|deploy|failed pipeline|unhandled|spike", re.IGNORECASE)
_IGNORE_RE = re.compile(r"deprecat|flaky|cosmetic|wontfix", re.IGNORECASE)
_LOW_RE = re.compile(r"warning|staging|typo|cosmetic", re.IGNORECASE)
_SECURITY_RE = re.compile(
    r"auth|login|ldap|sso|credential|token|unauthori[sz]ed|forbidden|permission denied|certificate",
    re.IGNORECASE,
)
# A neighbour this close is textually near-identical. Only the mock uses a
# threshold at all -- the real agent is asked to judge, not to compare a float.
MOCK_DUPLICATE_DISTANCE = 0.05
MOCK_TOP_N = 10
# Masking rules that mean a CREDENTIAL was logged, as opposed to personal data.
# A hit here is a finding about the application, so the report calls it out.
# `home-directory-path` is deliberately absent: a username in a stack frame is
# personal data, not a leaked secret, and treating it as one would fire the
# "applications logging secrets" callout on ordinary traces.
_CREDENTIAL_HITS = {
    "bearer-token", "api-key-assignment", "gitlab-pat", "aws-access-key",
    "env-var-secretish", "connection-string-credentials", "jwt",
    "private-key-block", "prefixed-vendor-token", "url-signature",
    "session-identifier",
}


class MockProvider:
    """Deterministic keyword-heuristic provider. No network, no cost."""

    def generate_list(self, system: str, prompt: str, item_schema: type[M]) -> list[M]:
        items = extract_input_json(prompt)
        if item_schema is TriageOutput:
            return [self._triage_output(item) for item in items]  # type: ignore[misc]
        if item_schema is AlertSummary:
            return [self._summarize(item) for item in items]  # type: ignore[misc]
        if item_schema is TriageResult:
            return [self._triage(item) for item in items]  # type: ignore[misc]
        if item_schema is SelectedIssue:
            return self._select(items)  # type: ignore[return-value]
        raise LLMError(f"mock provider does not support schema {item_schema}")

    @staticmethod
    def _select(items: list[dict]) -> list[SelectedIssue]:
        """Stand-in for the top-issues agent.

        Ranks on users, then 24h events -- the two signals the real prompt is
        told to weigh hardest -- and calls an alert a duplicate when its nearest
        historical neighbour is very close. It is a heuristic, not a model, but
        it produces the same *shape* of answer, which is what lets the
        orchestrator and the eval harness run with no network.
        """
        def impact(item: dict) -> tuple[int, int]:
            return int(item.get("user_count", 0) or 0), int(item.get("events_24h", 0) or 0)

        duplicates: set[str] = set()
        out: dict[str, SelectedIssue] = {}
        for item in items:
            alert_id = item.get("alert_id", "")
            nearest = (item.get("similar_past") or [{}])[0]
            distance = float(nearest.get("distance", 1.0))
            if nearest.get("alert_id") and distance <= MOCK_DUPLICATE_DISTANCE:
                duplicates.add(alert_id)
                out[alert_id] = SelectedIssue(
                    alert_id=alert_id,
                    is_duplicate=True,
                    duplicate_of=str(nearest.get("alert_id", "")),
                    reason=f"[mock] near-identical to a past alert (distance {distance})",
                    selected=False,
                    rank=0,
                )

        ranked = sorted(
            (i for i in items if i.get("alert_id") not in duplicates), key=impact, reverse=True
        )
        for position, item in enumerate(ranked, start=1):
            alert_id = item.get("alert_id", "")
            users, events = impact(item)
            out[alert_id] = SelectedIssue(
                alert_id=alert_id,
                is_duplicate=False,
                duplicate_of="",
                reason=f"[mock] {users} users, {events} events in 24h",
                selected=position <= MOCK_TOP_N,
                rank=position if position <= MOCK_TOP_N else 0,
            )
        # Input order, so the caller sees one object per input alert exactly as
        # the real prompt demands.
        return [out[i.get("alert_id", "")] for i in items if i.get("alert_id", "") in out]

    @classmethod
    def _triage_output(cls, item: dict) -> TriageOutput:
        """The merged shape. Reuses the two existing heuristics so mock
        behaviour is unchanged by the merge, which is what lets the eval
        scorecards be compared before and after."""
        summary = cls._summarize(item)
        triage = cls._triage(item)

        hits = {str(h).lower() for h in item.get("masking_hits") or []}
        leaked = _CREDENTIAL_HITS & hits
        text = f"{item.get('title', '')} {item.get('body', '')}"
        if leaked:
            security, rationale = True, f"[mock] masking rule(s) fired: {', '.join(sorted(leaked))}"
        elif _SECURITY_RE.search(text):
            security, rationale = True, "[mock] matches security keywords"
        else:
            security, rationale = False, "[mock] no security signals"

        return TriageOutput(
            alert_id=item.get("alert_id", ""),
            summary=summary.summary,
            component=summary.component,
            suspected_cause=summary.suspected_cause,
            security_relevant=security,
            security_rationale=rationale,
            priority=triage.priority,
            decision=triage.decision,
            reasoning=triage.reasoning,
            clean_title=summary.title,
        )

    @staticmethod
    def _summarize(item: dict) -> AlertSummary:
        title = item.get("title", "untitled")
        body = item.get("body", "")
        return AlertSummary(
            alert_id=item.get("alert_id") or item.get("fingerprint", ""),
            title=title[:120],
            summary=(f"{title}. {body[:160]}".strip() or "No details available."),
            component=item.get("project", "unknown"),
            suspected_cause="unknown (mock provider)",
        )

    @staticmethod
    def _triage(item: dict) -> TriageResult:
        text = f"{item.get('title', '')} {item.get('summary', '')} {item.get('body', '')}"
        event_count = int(item.get("event_count", 1))
        if _IGNORE_RE.search(text):
            priority, decision, why = Priority.low, TriageDecision.ignore, "matches known-noise keywords"
        elif _CRITICAL_RE.search(text) or event_count >= 500:
            priority, decision, why = Priority.critical, TriageDecision.notify, "outage-class keywords or very high event volume"
        elif _HIGH_RE.search(text) or event_count >= 100:
            priority, decision, why = Priority.high, TriageDecision.notify, "production-impact keywords or high event volume"
        elif _LOW_RE.search(text):
            priority, decision, why = Priority.low, TriageDecision.notify, "low-impact keywords"
        else:
            priority, decision, why = Priority.medium, TriageDecision.notify, "no strong signals; defaulting to medium"
        return TriageResult(
            alert_id=item.get("alert_id", ""),
            priority=priority,
            decision=decision,
            reasoning=f"[mock] {why}",
        )


def build_provider(
    provider: str, project: str, location: str, model: str, api_key: str = ""
) -> LLMProvider:
    if provider == "vertex":
        return VertexProvider(project, location, model)
    if provider == "gemini":
        # Developer API (AI Studio key). Local smoke tests against a real model.
        if not api_key:
            raise ValueError("LLM_PROVIDER=gemini requires GEMINI_API_KEY")
        return VertexProvider(project, location, model, api_key)
    if provider == "mock":
        return MockProvider()
    raise ValueError(f"unknown LLM provider: {provider}")
