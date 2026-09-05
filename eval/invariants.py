"""Metamorphic checks: properties that must hold whatever the right answer is.

This is the highest-value evaluation in the repo per unit of human effort,
because it needs **no labels at all**. Each check builds two alerts that differ
in exactly one field, runs the real agent over them, and asserts the relation
the rules in `config/priority_matrix.yaml` promise. If RULE 2 says production
outranks staging, then an alert that is identical apart from its environment
must not rank below its own staging twin -- and that is checkable without
anyone deciding what its priority ought to be.

What this catches that a golden dataset does not:

  * a prompt edit that silently drops a rule (the dataset may not contain a case
    that exercises it);
  * the model anchoring on the wrong number -- lifetime events instead of 24h
    is the specific failure this pipeline has already been bitten by;
  * dedup suppressing an escalation, which is the expensive mistake and is
    invisible in production by construction;
  * a bias nobody asked for, e.g. ranking by which project reported the issue.

What it does NOT catch: being uniformly, consistently wrong. A model that rates
everything `low` passes most of these. Invariants are necessary, not
sufficient -- pair them with the golden dataset and the outcome backtest
(see EVALUATION.md).
"""

from __future__ import annotations

import time

from eval.harness import Outcome, Report, make_alert, rank_of
from pipeline.agents.providers import LLMProvider
from pipeline.agents.top_issues import select_top_issues
from pipeline.agents.triage import triage_alerts
from shared.models import Alert

# Small top_n so "selected" is a real decision rather than "everything fits".
TOP_N = 2


def _selection_order(
    provider: LLMProvider, expected_first: Alert, expected_second: Alert
) -> tuple[int | None, int | None, list[Alert]]:
    """Runs selection with the EXPECTED WINNER SECOND in the input array, so a
    pass cannot come from the model simply preserving input order."""
    alerts = [expected_second, expected_first]
    selected, _verdicts = select_top_issues(alerts, provider, top_n=TOP_N, chunk_size=25)
    return (
        rank_of(selected, expected_first.source_id),
        rank_of(selected, expected_second.source_id),
        selected,
    )


def _outranks(
    provider: LLMProvider,
    name: str,
    why: str,
    winner: Alert,
    loser: Alert,
) -> Outcome:
    winner_rank, loser_rank, _ = _selection_order(provider, winner, loser)
    if winner_rank is None:
        return Outcome(name, False, "the expected winner was not selected at all", why)
    if loser_rank is None:
        return Outcome(name, True, f"winner ranked #{winner_rank}; the other was not selected", why)
    ok = winner_rank < loser_rank
    return Outcome(name, ok, f"winner #{winner_rank} vs other #{loser_rank}", why)


# --------------------------------------------------------------------------
# Selection invariants
# --------------------------------------------------------------------------


def users_outrank_volume(provider: LLMProvider) -> Outcome:
    return _outranks(
        provider,
        "users outrank raw volume",
        "RULE 1. Ten thousand events from one user is a retry loop; fifty events "
        "from fifty users is an outage. Inverting this is how a report fills up "
        "with noisy-but-harmless retry storms.",
        winner=make_alert("broad", "Checkout submission failed", user_count=400,
                          hourly_counts=[(i, 2) for i in range(24)]),
        loser=make_alert("loud", "Thumbnail fetch retry failed", user_count=1,
                         hourly_counts=[(i, 500) for i in range(24)]),
    )


def production_outranks_development(provider: LLMProvider) -> Outcome:
    return _outranks(
        provider,
        "production outranks development",
        "RULE 2. Identical error, identical numbers, different environment. If "
        "this fails, a developer's laptop can push a production outage out of "
        "the report.",
        winner=make_alert("prod", "TypeError in submission handler", environment="production"),
        loser=make_alert("dev", "TypeError in submission handler", environment="development"),
    )


def recent_volume_outranks_lifetime_total(provider: LLMProvider) -> Outcome:
    return _outranks(
        provider,
        "24h volume outranks lifetime total",
        "The failure this pipeline has actually been bitten by: a real issue "
        "reads 206,017 lifetime events next to 223 in the last 24h. A model "
        "anchoring on the big number ranks a years-old steady error above "
        "today's incident.",
        winner=make_alert("today", "New timeout in publish service", event_count=60,
                          hourly_counts=[(i, 25) for i in range(24)]),
        loser=make_alert("ancient", "Legacy warning in reporting job", event_count=206_017,
                         hourly_counts=[(i, 1) for i in range(4)]),
    )


def security_outranks_ordinary_noise(provider: LLMProvider) -> Outcome:
    return _outranks(
        provider,
        "credential leak outranks a noisier ordinary error",
        "RULE 3. A masking hit like 'bearer-token' means an application is "
        "logging credentials into its own error messages -- a finding in its "
        "own right, at any volume.",
        winner=make_alert("leak", "Auth request failed: token [MASKED_BEARER_TOKEN] rejected",
                          user_count=2, masking_hits=["bearer-token"],
                          hourly_counts=[(i, 1) for i in range(24)]),
        loser=make_alert("busy", "Image resize failed", user_count=30,
                         hourly_counts=[(i, 40) for i in range(24)]),
    )


def new_outranks_ongoing(provider: LLMProvider) -> Outcome:
    return _outranks(
        provider,
        "regressed outranks ongoing",
        "RULE 4. Same error, same numbers; one has come back and one has been "
        "there all along. A system that cannot tell those apart re-reports the "
        "furniture and misses the change.",
        winner=make_alert("regressed", "Asset publish failed", substatus="regressed"),
        loser=make_alert("ongoing", "Asset publish failed", substatus="ongoing"),
    )


def unhandled_outranks_handled(provider: LLMProvider) -> Outcome:
    return _outranks(
        provider,
        "unhandled outranks handled",
        "RULE 5. is_unhandled means nobody caught it and a user very likely saw "
        "a failure.",
        winner=make_alert("uncaught", "Submission failed", is_unhandled=True),
        loser=make_alert("caught", "Submission failed", is_unhandled=False),
    )


# --------------------------------------------------------------------------
# Dedup invariants -- the expensive-mistake direction
# --------------------------------------------------------------------------


def exact_repeat_is_deduplicated(provider: LLMProvider) -> Outcome:
    """The cheap direction: a repeat that has not changed should not be
    re-reported. Being wrong here costs a reader ten seconds."""
    alert = make_alert(
        "repeat", "ConnectionError: could not reach internal-db-01",
        similar_past=[{
            "alert_id": "yesterday", "title": "ConnectionError: could not reach internal-db-01",
            "project": "asset-service", "last_seen_run": "2026-08-17",
            "past_priority": "medium", "past_decision": "notify", "distance": 0.01,
        }],
    )
    _selected, verdicts = select_top_issues([alert], provider, top_n=TOP_N)
    verdict = verdicts.get(alert.fingerprint())
    if verdict is None:
        return Outcome("an unchanged repeat is deduplicated", False, "no verdict returned")
    return Outcome(
        "an unchanged repeat is deduplicated",
        verdict.is_duplicate,
        f"is_duplicate={verdict.is_duplicate} — {verdict.reason[:110]}",
        "Without this the report re-prints yesterday's furniture and people stop reading it.",
    )


def escalated_repeat_is_not_deduplicated(provider: LLMProvider) -> Outcome:
    """The expensive direction, and the single most important check in this
    file. A duplicate call suppresses an alert entirely, so a wrong one is
    invisible in production by construction -- nobody sees what was hidden.
    `dedup_guidance` exists mostly to make this case come out right."""
    alert = make_alert(
        "escalated", "ConnectionError: could not reach internal-db-01",
        substatus="regressed", user_count=500,
        hourly_counts=[(i, 5) for i in range(20)] + [(i, 400) for i in range(20, 24)],
        similar_past=[{
            "alert_id": "yesterday", "title": "ConnectionError: could not reach internal-db-01",
            "project": "asset-service", "last_seen_run": "2026-08-17",
            "past_priority": "low", "past_decision": "notify", "distance": 0.02,
        }],
    )
    _selected, verdicts = select_top_issues([alert], provider, top_n=TOP_N)
    verdict = verdicts.get(alert.fingerprint())
    if verdict is None:
        return Outcome("an escalated repeat is NOT deduplicated", False, "no verdict returned")
    return Outcome(
        "an escalated repeat is NOT deduplicated",
        not verdict.is_duplicate,
        f"is_duplicate={verdict.is_duplicate} — {verdict.reason[:110]}",
        "A wrong duplicate suppresses an alert nobody then sees. This is the "
        "failure mode with no natural feedback signal, so it has to be tested.",
    )


# --------------------------------------------------------------------------
# Triage invariants
# --------------------------------------------------------------------------


def project_identity_does_not_swing_priority(provider: LLMProvider) -> Outcome:
    """The same error in two differently-named projects must not swing priority
    by more than one band.

    This started as a strict equality check and the real model failed it:
    `asset-service` -> high, `internal-scratch-tool` -> medium. Reading that
    failure carefully, the model was right and the check was wrong. The rules
    themselves invoke core-ness -- `priority_rules` defines CRITICAL as "a CORE
    prodtools service is down", and selection RULE 6 puts core services above
    scheduled jobs and one-off scripts. A crash in a scratch tool genuinely is
    less consequential than the same crash in asset publishing.

    What was deliberately removed from this system was a hand-maintained
    per-project criticality TABLE -- it never separated anything, because
    almost every project belongs to the same team, and it made the ranking a
    statement about ownership rather than severity. Judging consequence from
    what a service evidently does is a different thing.

    So the guard is against a project name SWINGING the verdict -- low to
    critical on identical evidence -- not against it mattering at all. One band
    is judgement; two is a criticality table growing back.
    """
    order = {"low": 0, "medium": 1, "high": 2, "critical": 3}
    a = make_alert("p1", "TypeError: cannot read property 'id' of undefined", project="asset-service")
    b = make_alert("p2", "TypeError: cannot read property 'id' of undefined", project="internal-scratch-tool")
    outputs = triage_alerts([a, b], provider)
    by_id = {o.alert_id: o for o in outputs}
    pa = by_id[a.fingerprint()].priority.value
    pb = by_id[b.fingerprint()].priority.value
    gap = abs(order[pa] - order[pb])
    return Outcome(
        "the project's name does not swing priority",
        gap <= 1,
        f"{a.project} -> {pa}, {b.project} -> {pb} ({gap} band(s) apart)",
        "A two-band swing on identical evidence is a per-project criticality "
        "table growing back by another name.",
    )


def priority_rises_with_impact(provider: LLMProvider) -> Outcome:
    """Monotonicity: the same error hitting far more people, in production,
    uncaught, must not be triaged *lower*."""
    order = {"low": 0, "medium": 1, "high": 2, "critical": 3}
    small = make_alert("small", "Publish request failed", user_count=1, is_unhandled=False,
                       environment="staging", hourly_counts=[(i, 1) for i in range(24)])
    big = make_alert("big", "Publish request failed", user_count=900, is_unhandled=True,
                     environment="production", substatus="regressed",
                     hourly_counts=[(i, 300) for i in range(24)])
    outputs = triage_alerts([small, big], provider)
    by_id = {o.alert_id: o for o in outputs}
    ps = by_id[small.fingerprint()].priority.value
    pb = by_id[big.fingerprint()].priority.value
    return Outcome(
        "priority is monotonic in impact",
        order[pb] >= order[ps],
        f"1 user/staging -> {ps}, 900 users/production/regressed/unhandled -> {pb}",
        "If a bigger version of the same error triages lower, the priority is "
        "not tracking impact at all.",
    )


ALL = [
    users_outrank_volume,
    production_outranks_development,
    recent_volume_outranks_lifetime_total,
    security_outranks_ordinary_noise,
    new_outranks_ongoing,
    unhandled_outranks_handled,
    exact_repeat_is_deduplicated,
    escalated_repeat_is_not_deduplicated,
    project_identity_does_not_swing_priority,
    priority_rises_with_impact,
]


def run(provider: LLMProvider, pace: float = 0.0) -> Report:
    """Run every invariant. `pace` sleeps between checks.

    Pacing exists because the free tier allows 5 requests a minute and each
    check makes one or two. Without it, most of the suite reports a rate limit
    rather than a verdict -- and a suite that mostly reports infrastructure
    noise is a suite people stop reading.
    """
    report = Report("INVARIANTS (no labels needed)")
    for n, check in enumerate(ALL):
        if pace and n:
            time.sleep(pace)
        try:
            report.add(check(provider))
        except Exception as exc:
            # NOT a failure: the model was never asked. Saying "FAIL" here would
            # claim the model broke a rule when the truth is a dead key or a
            # quota.
            message = str(exc)
            hint = " (rate limit — raise --pace)" if "RESOURCE_EXHAUSTED" in message else ""
            report.add(Outcome(check.__name__, False, f"could not run{hint}: {message[:160]}",
                               errored=True))
    return report
