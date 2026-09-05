"""Shared Pydantic schemas passed between the pipeline stages.

Scope is v1: a single all-cloud service that pulls Sentry directly, so there
is no cross-process collector contract to version anymore (the old
PipelineRequest/PipelineResponse/TicketAction envelopes are gone). These
models are still kept separate from the pipeline code because the eval
harness and tests depend on them."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from enum import Enum

from pydantic import BaseModel, Field


class AlertSource(str, Enum):
    # gitlab is retained for the v2 second-source work and the golden dataset;
    # v1 only ingests sentry.
    gitlab = "gitlab"
    sentry = "sentry"


class Priority(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"


class TriageDecision(str, Enum):
    notify = "notify"
    ignore = "ignore"


class Alert(BaseModel):
    """A single normalized alert (already masked before it reaches the LLM).

    Everything past `labels` comes from the Sentry issue payload's allowlisted
    projection (see `pipeline.sentry_client.issue_to_alert`). All of it is
    optional with a default so the golden dataset and older fixtures, which
    predate these fields, keep validating.
    """

    source: AlertSource
    source_id: str
    kind: str = ""  # e.g. "sentry_issue"
    title: str
    body: str = ""
    url: str = ""
    project: str = ""
    event_count: int = 1
    first_seen: datetime | None = None
    last_seen: datetime | None = None
    labels: list[str] = Field(default_factory=list)
    # Days this alert has been continuously observed; annotated by the
    # dedup stage, not the ingestion stage.
    ongoing_days: int = 0

    # --- Impact signals from the Sentry issue payload ---------------------
    short_id: str = ""  # human-readable e.g. "PRODTOOLS-4X"
    user_count: int = 0
    environment: str = ""
    substatus: str = ""  # new | regressed | escalating | ongoing | ...
    is_unhandled: bool = False
    level: str = ""
    sentry_priority: str = ""  # Sentry's own priority, used as a prior
    platform: str = ""
    seer_fixability: float | None = None
    # stats.24h as [unix_ts, count] pairs. Sum these for "events in the last
    # 24h" -- `count`/`event_count` reflects whatever statsPeriod was queried
    # and is NOT a 24h number.
    hourly_counts: list[tuple[int, int]] = Field(default_factory=list)
    first_release: str = ""
    last_release: str = ""
    last_release_date: datetime | None = None

    # Names of the masking rules that fired on this alert. A hit means an
    # application is logging something it should not; surfaced in the report.
    masking_hits: list[str] = Field(default_factory=list)

    # --- Dedup / historical context, filled by the BigQuery vector search --
    # The nearest historical alerts by embedding cosine distance, as compact
    # dicts (alert_id, title, run_date, priority, distance). This is what the
    # top-issues agent reads to decide "is this the same error we already
    # triaged, or something new". No hash comparison is involved anywhere:
    # a stack trace that shifts by one frame produces a different hash but a
    # near-identical vector, and it was the hash that was wrong.
    similar_past: list[dict] = Field(default_factory=list)
    # The nearest alerts in THIS run. Sentry files the same underlying failure
    # as several issues whenever the stack shape differs, so two rows of today's
    # list can be one incident. `similar_past` cannot show this -- it looks only
    # at previous runs -- and without it the agent has to notice from the titles.
    similar_today: list[dict] = Field(default_factory=list)
    # Set by the top-issues agent, not by the ingestion stage.
    is_duplicate: bool = False
    duplicate_of: str = ""
    selection_reason: str = ""

    def fingerprint(self) -> str:
        """Stable identity of one Sentry issue across days. This is an identity,
        not a dedup decision -- two different Sentry issues can be the same
        underlying error, and that call is the vector search's to make."""
        raw = f"{self.source.value}:{self.source_id}:{self.kind}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]

    def embedding_text(self) -> str:
        """The text that gets vectorized for the BigQuery vector search.

        Title, culprit-ish body and level -- the parts that describe *what*
        broke. Deliberately no counters, timestamps or run ids: two runs of the
        same error must land on top of each other in vector space, and anything
        that changes daily would push them apart. The body is truncated because
        the embedding model has an input limit and the informative part of a
        stack trace is the top of it.
        """
        return f"{self.title}\n{self.project}\n{self.body}"[:8000]

    def events_24h(self) -> int:
        """Events in the last 24h, summed from the stats buckets. Falls back to
        `event_count` when the payload carried no stats."""
        if not self.hourly_counts:
            return self.event_count
        return sum(count for _ts, count in self.hourly_counts)


class AlertSummary(BaseModel):
    """Output of the summarization agent for one alert."""

    alert_id: str = Field(description="Fingerprint of the alert being summarized")
    title: str = Field(description="Short human-readable title, cleaned up")
    summary: str = Field(description="2-3 sentence plain-English summary of the alert")
    component: str = Field(description="Affected service/component, best effort")
    suspected_cause: str = Field(description="Most likely root cause, or 'unknown'")


class TriageResult(BaseModel):
    """Output of the triage agent for one summarized alert.

    No owner/routing field. Deciding which team should act is a second job with
    its own failure mode -- the model has to know an org chart it was never
    given -- and it is not what this pipeline is for. The report's job is to say
    what broke, how bad it is, and where to look; a human picks it up from
    there. What replaced it is the alert's Sentry permalink, which is a fact
    rather than a guess.
    """

    alert_id: str = Field(description="Fingerprint of the alert being triaged")
    priority: Priority
    decision: TriageDecision
    reasoning: str = Field(description="Brief chain-of-thought for the decision (audit trail)")


class TriageOutput(BaseModel):
    """Output of the single merged triage agent for one alert.

    **Field order is load-bearing.** Generation is autoregressive: the model
    writes these fields in declaration order, so everything analytical
    (summary, component, suspected cause, security assessment) comes before
    anything conclusive (priority, decision). The model has to reason in the
    open before it commits to a verdict, which recovers most of the benefit of
    a separate summarize->triage pass inside a single call. `reasoning` sits
    last so it explains a verdict that already exists rather than rationalizing
    one the model is about to invent.

    Do not reorder these fields casually -- see eval/VARIANTS.md.
    """

    alert_id: str = Field(description="Fingerprint of the alert; copy the input alert_id EXACTLY")
    summary: str = Field(description="2-3 plain-English sentences: what broke, where, and the impact")
    component: str = Field(description="Affected service/component, best effort")
    suspected_cause: str = Field(description="Most likely root cause, or 'unknown'")
    security_relevant: bool = Field(description="True if this looks security-relevant")
    security_rationale: str = Field(description="Why it is or is not security-relevant, one line")
    priority: Priority
    decision: TriageDecision
    reasoning: str = Field(description="Brief justification for the verdict (audit trail)")
    clean_title: str = Field(default="", description="Cleaned-up one-line title")

    def to_summary(self) -> AlertSummary:
        """View onto the summarization half, so downstream code that predates
        the merge (TriagedAlert, the report renderer, chat_notify)
        keeps working unchanged."""
        return AlertSummary(
            alert_id=self.alert_id,
            title=self.clean_title or self.summary[:120],
            summary=self.summary,
            component=self.component,
            suspected_cause=self.suspected_cause,
        )

    def to_triage_result(self) -> TriageResult:
        """View onto the triage half."""
        return TriageResult(
            alert_id=self.alert_id,
            priority=self.priority,
            decision=self.decision,
            reasoning=self.reasoning,
        )


class SelectedIssue(BaseModel):
    """One decision of the top-issues agent about one alert.

    **Field order is load-bearing** for the same reason as TriageOutput:
    generation is autoregressive, so the model states what it observed and
    whether it is a repeat before it commits to a rank.
    """

    alert_id: str = Field(description="Fingerprint of the alert; copy the input alert_id EXACTLY")
    is_duplicate: bool = Field(
        description="True if this is the same underlying error as one of its similar_past entries"
    )
    duplicate_of: str = Field(
        default="", description="alert_id of the historical alert this duplicates, or empty"
    )
    reason: str = Field(description="One or two sentences: why this ranks where it does")
    selected: bool = Field(description="True if this belongs in today's top issues")
    rank: int = Field(default=0, description="1 = most important. 0 when not selected.")


class TriagedAlert(BaseModel):
    """Everything the pipeline knows about one alert after triage.

    Alert metadata is denormalized here so the report renderer and Chat
    notifier don't need to join back against the original payload.
    """

    alert_id: str
    source: AlertSource
    title: str
    url: str = ""
    project: str = ""
    event_count: int = 1
    ongoing_days: int = 0
    summary: AlertSummary
    triage: TriageResult

    # --- how the top-issues agent ranked this, and why ---
    # rank 1 is the most important issue of the run. Selection and triage are
    # two independent LLM judgements over the same alert, so a rank-1 issue the
    # triage agent then calls "low" is a real disagreement worth surfacing --
    # see `disagreement` below. It replaces the old numeric impact_score, whose
    # band thresholds had to be recalibrated by hand every time a weight moved.
    rank: int = 0
    selection_reason: str = ""
    # True when the top-issues agent put this near the top but the triage agent
    # called it low. With no human feedback loop, this remains the pipeline's
    # only self-check.
    disagreement: bool = False
    security_relevant: bool = False
    security_rationale: str = ""

    # --- impact signals, denormalized for the report and card ---
    # Copied off the Alert rather than joined back to it, for the same reason
    # the summary and triage are: the renderer and the Chat notifier should not
    # need the original payload.
    environment: str = ""
    user_count: int = 0
    substatus: str = ""
    is_unhandled: bool = False
    level: str = ""
    sentry_priority: str = ""
    events_24h: int = 0
    hourly_counts: list[tuple[int, int]] = Field(default_factory=list)
    masking_hits: list[str] = Field(default_factory=list)
    similar_past_count: int = 0


class RunStats(BaseModel):
    ingested: int = 0
    prefiltered: int = 0  # dropped by the noise prefilter before selection
    deduped: int = 0  # judged a repeat of a historical alert by the LLM
    considered: int = 0  # reached the top-issues agent
    processed: int = 0  # selected and triaged
    ignored: int = 0
    notified: int = 0
    by_priority: dict[str, int] = Field(default_factory=dict)

    @property
    def noise_ratio(self) -> float:
        if self.ingested == 0:
            return 0.0
        return (self.prefiltered + self.deduped + self.ignored) / self.ingested


class RunResult(BaseModel):
    """Result of one daily triage run, returned by the /run endpoint."""

    run_id: str
    run_date: str
    degraded: bool = False  # true when the LLM fallback digest path was used
    stats: RunStats = Field(default_factory=RunStats)
    results: list[TriagedAlert] = Field(default_factory=list)
    pdf_url: str = ""


def utcnow() -> datetime:
    return datetime.now(UTC)
