"""Does it say the same thing twice?

Run the identical input through the agents more than once and measure how much
the answer moves. No labels needed, and the number it produces is a **ceiling
on every other measurement in the suite**: if the selection changes by 40%
between two runs of the same input, an 80%-accuracy figure from the golden
dataset means very little, because a re-run would have produced a different 80%.

This is the number that replaced the old scorer's determinism. The previous
pipeline ranked with arithmetic, so the ordering was reproducible by
construction; handing ranking to a model bought better judgement and sold that
guarantee. This measures what was sold.

Sampling is at temperature 0.1 (the pipeline's own setting), so this measures
the residual non-determinism of a near-greedy decode, not creative variance.
"""

from __future__ import annotations

from eval.harness import Outcome, Report
from pipeline.agents.providers import LLMProvider
from pipeline.agents.top_issues import select_top_issues
from pipeline.agents.triage import triage_alerts
from shared.models import Alert

# Below these the pipeline is too unstable for anyone to trust a day-to-day
# comparison of its reports. They are judgement calls, set where a shifting
# report would start to look broken to a reader rather than merely different.
MIN_SELECTION_OVERLAP = 0.8
MIN_PRIORITY_AGREEMENT = 0.8


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def run(alerts: list[Alert], provider: LLMProvider, runs: int = 2, top_n: int = 10) -> Report:
    report = Report(f"STABILITY ({runs} runs of identical input)")
    if len(alerts) < 2:
        report.add(Outcome("stability", True, "need at least 2 alerts", skipped=True))
        return report

    selections: list[set[str]] = []
    orderings: list[list[str]] = []
    priorities: list[dict[str, str]] = []

    for _ in range(runs):
        selected, _verdicts = select_top_issues(alerts, provider, top_n=top_n)
        selections.append({a.fingerprint() for a in selected})
        orderings.append([a.fingerprint() for a in selected])
        outputs = triage_alerts(selected, provider)
        priorities.append({o.alert_id: o.priority.value for o in outputs})

    overlaps = [_jaccard(selections[0], s) for s in selections[1:]]
    worst_overlap = min(overlaps) if overlaps else 1.0
    report.add(Outcome(
        "the same input selects the same issues",
        worst_overlap >= MIN_SELECTION_OVERLAP,
        f"worst-case selection overlap {worst_overlap:.0%} (threshold {MIN_SELECTION_OVERLAP:.0%})",
        "This is a ceiling on every accuracy number in the suite: a measurement "
        "cannot be more meaningful than the process it measures is repeatable.",
    ))

    identical_order = all(o == orderings[0] for o in orderings[1:])
    report.add(Outcome(
        "rank order is reproducible",
        True,  # informational: order churn inside the same set is tolerable
        "identical across runs" if identical_order else
        f"order moved between runs (sets still overlap {worst_overlap:.0%})",
        "",
    ))

    shared = set.intersection(*(set(p) for p in priorities)) if priorities else set()
    if not shared:
        report.add(Outcome("priority agreement across runs", False,
                           "no alert was selected by every run — selection is too unstable to compare"))
        return report

    agreed = sum(1 for fp in shared if len({p[fp] for p in priorities}) == 1)
    agreement = agreed / len(shared)
    report.add(Outcome(
        "the same alert gets the same priority",
        agreement >= MIN_PRIORITY_AGREEMENT,
        f"{agreed}/{len(shared)} commonly-selected alerts agreed ({agreement:.0%}, "
        f"threshold {MIN_PRIORITY_AGREEMENT:.0%})",
        "A priority that changes between identical runs is not a priority, and "
        "the audit trail will record whichever one the run happened to get.",
    ))
    return report
