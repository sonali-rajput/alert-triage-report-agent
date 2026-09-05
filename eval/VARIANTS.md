# Triage variants: what we did not build, and how to settle it

> ⚠️ **Partly out of date.** This file describes experiments measured with
> `eval/run_eval.py` against `eval/golden_dataset.json` — both removed from the
> MVP (see EVALUATION.md §5 for why, and how to revive them). It also predates
> the rework, so it discusses the deterministic impact score and hash dedup,
> which no longer exist.
>
> The *experiments* it records are still worth reading: they are the questions
> nobody has settled, and each states how to settle it. The harness it names is
> not the one you would use — that is `eval/run_checks.py` now.

v1 ships **one LLM call** over an issue-payload projection, with the top 15 alerts
enriched with a stack trace. Several plausible alternatives were considered and
deliberately not built. This file records them so they are not rediscovered from
scratch — and so anyone who wants to argue for one has a defined experiment to
run rather than an opinion to assert.

**Headline metric throughout: High/Critical recall.** Missing a critical is worse
than over-flagging a low, so the metric is asymmetric on purpose. `run_eval`
reports two forms:

- **exact recall** — predicted the same band the human labelled.
- **caught-at-all** — a critical predicted as high (or vice versa) still counts.
  This is the one that matters operationally: it means the alert got escalated.

Accuracy is the tiebreak, never the headline.

---

## ⚠️ Blocker: the golden dataset predates Phase 0

`eval/golden_dataset.json` was built before the impact signals existed. **0 of 24
entries carry `sentry_priority`, `user_count`, `environment`, `substatus`,
`is_unhandled` or `stats.24h`** — the fields the whole Phase 1 design rests on.

Consequences, which apply to nearly every experiment below:

1. **The two zero-cost baselines cannot run.** `run_eval` detects the missing
   coverage and skips them with an explanation rather than printing a
   meaningless number.
2. **The eval understates the pipeline.** The model is being scored on title,
   body, project and event count alone — none of the signals the prompt now
   tells it to weigh most heavily.

**Fix before running any experiment here:** re-capture the golden dataset through
the current `issue_to_alert` projection so every entry carries the full field
set, then re-label. Roughly: pull a day of real issues, run them through the
projection, and have someone label priority plus a one-line "why".

Until that happens, treat every number below as provisional.

---

## Baselines to beat (blocked on the above)

| Baseline | What it is | Why it matters |
|---|---|---|
| **Sentry's own `priority`** | the `priority` field, straight from the payload | Costs nothing. If the pipeline cannot beat it on High/Critical recall, the LLM is not earning its place *on the priority decision* — though it still earns it on routing, the notify/ignore call, and the audit-trail reasoning, none of which Sentry provides. |
| **Impact score alone** | `ImpactScorer.implied_priority`, no LLM | Isolates how much the LLM adds over arithmetic. If the gap is small, the honest conclusion is that the score is doing the work. |

Run both with `python -m eval.run_eval --baselines-only` — no LLM calls, no cost.

---

## Recorded results

### Single call vs. two-stage — **settled, no regression**

| | Accuracy | H/C exact | H/C caught | Misclassifications |
|---|---|---|---|---|
| Two-stage (pre-merge) | 21/24 = 88% | — | — | 3 |
| Single call (post-merge) | 21/24 = 88% | 8/10 = 80% | 9/10 = 90% | 3 (identical) |

Provider: `mock`. Identical because `MockProvider._triage_output` deliberately
reuses the two existing heuristics, which is what makes the comparison valid as
a *structural* check: the merge did not lose or reorder any alert.

> **This does not tell us whether the merge helps or hurts a real model.** The
> mock is a keyword heuristic; it does not read the prompt. The prompt rewrite,
> the response-schema ordering and the richer payload are all invisible to it.
> **Re-run with `LLM_PROVIDER=vertex` before drawing any conclusion.** That needs
> `gcloud auth application-default login` and was not run here.

---

## Variants not built

### 1. Two-stage summarize → triage

**Hypothesis.** A separate summarization pass gives the model room to reason
before judging, producing better priorities than one call with ordered fields.

**Against.** Twice the latency and roughly twice the tokens for one judgement.
The old split also actively *lost* information: `_triage_payload` never forwarded
`alert.body`, so the triage stage judged priority from the summarizer's
paraphrase and never saw the original error.

**Toggle.** `git revert` the merge commit, or reconstruct from
`shared.models.TriageOutput`'s two view methods.

**Settles it.** High/Critical recall on Vertex. If two-stage wins by less than
~5 points, the single call wins on cost and latency.

---

### 2. Cascade: cheap pass over everything, rich pass over the top N

**Hypothesis.** Flash over all alerts to sort noise from signal, then Pro over
only the top ~15 for the alerts humans actually read. Better quality where it
matters, at close to Flash cost.

**Against.** Two model configs, two prompts, two failure paths, and a
hand-off between them — a lot of machinery for a demo-grade target. The
deterministic score already does the "which ones matter" job for free.

**Toggle.** Not implemented. Would need a second provider instance and a split in
`triage_alerts`.

**Settles it.** High/Critical recall on the top 15 only, versus single-Flash, at
measured cost per run. Worth it only if the recall gain is material *and* the
per-run cost stays under the single-Pro alternative.

---

### 3. Issue-only vs. `enrich_top_n=15`

**Hypothesis.** Stack traces and breadcrumbs improve `suspected_cause` and
`component` accuracy enough to justify one API call per enriched issue.

**Prior.** The payload finding suggests enrichment helps the **narrative** but
not the **ranking** — priority is driven by numbers that are already in the issue
payload. Expect `suspected_cause` to improve and priority to barely move.

**Toggle.** `ENRICH_TOP_N=0` vs `ENRICH_TOP_N=15`. Deliberately a single config
flip, with no code path differences, precisely so this is cheap to test.

**Settles it.** Priority recall should be *unchanged* — if it moves much, the
score weights are wrong, not the enrichment. Judge `suspected_cause` quality by
reading 15 of them side by side; there is no automated metric for it and
pretending otherwise would be false precision.

---

### 4. Response-schema field ordering

**Hypothesis.** Generating `summary` → `component` → `suspected_cause` →
`security_relevant` before `priority` → `decision` makes the model reason before
committing, recovering most of the two-stage benefit inside one call.

**Why it is the default.** Generation is autoregressive: tokens already emitted
condition everything after them. A model that has just written "affects 480 users
in production, unhandled" is measurably less likely to then emit `low`.

**Toggle.** Reorder the fields in `shared.models.TriageOutput`. Note this also
requires confirming Gemini honours `propertyOrdering` from the Pydantic-derived
schema — **assumed, not yet verified.** `tests/test_agents.py` pins the field
order so the ordering cannot be lost by accident, but that test only checks our
schema, not what the provider does with it.

**Settles it.** Same prompt, both orderings, on Vertex. Compare High/Critical
recall and read a sample of `reasoning` fields — the giveaway for a bad ordering
is reasoning that reads as post-hoc justification.

---

### 5. Promoting repeated corrections into YAML ("Level 2")

**Hypothesis.** If humans keep correcting the same class of alert, that pattern
belongs in `component_criticality` or the priority matrix, not in a per-alert
override.

**Against.** Requires a human feedback loop, which v1 does not have — Sentry is
read-only for this project and there is no developer-input channel. See the
deferred section of `COMPONENTS.md`.

**Toggle.** N/A.

**Settles it.** N/A until corrections exist. Note that the *manual* version of
this is already the intended workflow: if the report ranks something wrong,
edit `component_criticality` in `config/priority_matrix.yaml`.

---

### 6. Impact-score weights

Not a model variant, but the highest-leverage knob in the system, and the one
most likely to be wrong on day one — the weights in
`config/priority_matrix.yaml` are a considered first guess, not a fitted result.

**Settles it.** Once the golden dataset carries the impact signals, the score
becomes fittable: grid-search or hand-tune the weights against
`implied_priority` accuracy with no LLM in the loop at all. That is the cheapest
accuracy work available in this codebase and should probably be done **before**
any prompt tuning.
