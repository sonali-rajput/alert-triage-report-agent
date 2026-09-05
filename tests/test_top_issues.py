"""The top-issues agent: dedup, rank, select."""

from __future__ import annotations

from pipeline.agents.providers import LLMError, MockProvider, extract_input_json
from pipeline.agents.top_issues import (
    SELECTION_PROMPT_VERSION,
    _payload,
    build_system_prompt,
    duplicate_count,
    select_top_issues,
)
from shared.models import Alert, AlertSource, SelectedIssue


def make_alert(title: str, **kwargs) -> Alert:
    base = dict(
        source=AlertSource.sentry, source_id=title, kind="sentry_issue",
        title=title, body="details", project="asset-service",
    )
    base.update(kwargs)
    return Alert(**base)


class Scripted:
    """Returns a fixed verdict per alert_id, and records every call."""

    def __init__(self, verdicts: dict[str, dict]):
        self._verdicts = verdicts
        self.calls: list[list[str]] = []

    def generate_list(self, system, prompt, item_schema):
        items = extract_input_json(prompt)
        self.calls.append([i["alert_id"] for i in items])
        out = []
        for item in items:
            spec = self._verdicts.get(item["alert_id"], {})
            out.append(
                SelectedIssue(
                    alert_id=item["alert_id"],
                    is_duplicate=spec.get("is_duplicate", False),
                    duplicate_of=spec.get("duplicate_of", ""),
                    reason=spec.get("reason", "because"),
                    selected=spec.get("selected", True),
                    rank=spec.get("rank", 0),
                )
            )
        return out


def test_empty_input_makes_no_call():
    class Exploding:
        def generate_list(self, *a, **kw):
            raise AssertionError("should not be called")

    assert select_top_issues([], Exploding()) == ([], {})


def test_returns_the_selected_alerts_in_rank_order():
    alerts = [make_alert(f"e{i}") for i in range(3)]
    ids = [a.fingerprint() for a in alerts]
    provider = Scripted({
        ids[0]: {"selected": True, "rank": 3},
        ids[1]: {"selected": True, "rank": 1},
        ids[2]: {"selected": True, "rank": 2},
    })

    selected, _verdicts = select_top_issues(alerts, provider)
    assert [a.title for a in selected] == ["e1", "e2", "e0"]


def test_ranks_are_renumbered_densely():
    """The model's own ranks can have gaps or repeats, and the report prints
    these as "#1..#10", which has to mean position."""
    alerts = [make_alert("a"), make_alert("b")]
    ids = [a.fingerprint() for a in alerts]
    provider = Scripted({ids[0]: {"rank": 4}, ids[1]: {"rank": 9}})

    _selected, verdicts = select_top_issues(alerts, provider)
    assert sorted(v.rank for v in verdicts.values()) == [1, 2]


def test_an_unranked_selection_sorts_last_rather_than_first():
    """rank 0 on a selected alert means the model forgot to number it. Sorting
    it naively would let it jump the queue ahead of rank 1."""
    alerts = [make_alert("forgot"), make_alert("numbered")]
    ids = [a.fingerprint() for a in alerts]
    provider = Scripted({ids[0]: {"rank": 0}, ids[1]: {"rank": 1}})

    selected, _ = select_top_issues(alerts, provider)
    assert [a.title for a in selected] == ["numbered", "forgot"]


def test_duplicates_are_excluded_from_the_report_but_kept_in_the_verdicts():
    """A duplicate is an alert nobody will see, so the audit trail is the only
    place its reasoning is ever recorded."""
    alerts = [make_alert("repeat"), make_alert("fresh")]
    ids = [a.fingerprint() for a in alerts]
    provider = Scripted({
        ids[0]: {"is_duplicate": True, "duplicate_of": "yesterday", "selected": True, "rank": 1},
        ids[1]: {"rank": 2},
    })

    selected, verdicts = select_top_issues(alerts, provider)
    assert [a.title for a in selected] == ["fresh"]
    assert duplicate_count(verdicts) == 1
    assert verdicts[ids[0]].duplicate_of == "yesterday"


def test_a_verdict_for_an_unknown_alert_id_is_discarded():
    """A garbled or invented alert_id would otherwise put a row in the audit
    trail describing an alert that does not exist — and inflate the 'deduped'
    number on the report's funnel, because that count reads this dict."""
    alert = make_alert("real")

    class InventsAnId(Scripted):
        def generate_list(self, system, prompt, item_schema):
            outputs = super().generate_list(system, prompt, item_schema)
            return outputs + [
                SelectedIssue(alert_id="not-an-alert-we-sent", is_duplicate=True,
                              duplicate_of="x", reason="hallucinated", selected=False, rank=0)
            ]

    provider = InventsAnId({alert.fingerprint(): {"rank": 1}})
    selected, verdicts = select_top_issues([alert], provider)

    assert [a.source_id for a in selected] == ["real"]
    assert set(verdicts) == {alert.fingerprint()}
    assert duplicate_count(verdicts) == 0


def test_selection_is_capped_at_top_n():
    alerts = [make_alert(f"e{i}") for i in range(8)]
    provider = Scripted({a.fingerprint(): {"rank": i + 1} for i, a in enumerate(alerts)})
    selected, _ = select_top_issues(alerts, provider, top_n=3)
    assert len(selected) == 3


def test_the_selection_reason_travels_with_the_alert():
    """It goes into the report and into the triage agent's payload, so it has
    to survive the hand-off."""
    alert = make_alert("only")
    provider = Scripted({alert.fingerprint(): {"reason": "480 users in production", "rank": 1}})
    selected, _ = select_top_issues([alert], provider)
    assert selected[0].selection_reason == "480 users in production"


# --------------------------------------------------------------------------
# Map/reduce over a large run
# --------------------------------------------------------------------------


def test_a_large_run_is_chunked_then_re_ranked():
    """The map/reduce is what keeps the prompt small enough to stay accurate at
    a few hundred issues: chunks shortlist, then one call ranks the shortlist."""
    alerts = [make_alert(f"e{i}") for i in range(20)]
    provider = Scripted({a.fingerprint(): {"rank": 1} for a in alerts})

    select_top_issues(alerts, provider, top_n=5, chunk_size=5)

    # 4 chunks of 5 in the map step, then the reduce step over the shortlist.
    assert [len(c) for c in provider.calls[:4]] == [5, 5, 5, 5]
    assert len(provider.calls) > 4


def test_the_final_ranking_is_always_a_single_call():
    """The bug this guards: two ranking calls each return their own "rank 1",
    and sorting the union interleaves two independent rankings — so the
    report's "#1..#10" silently stops meaning anything. It only bites on a big
    org, which is exactly where the ranking matters most."""
    alerts = [make_alert(f"e{i}") for i in range(400)]
    # Every alert survives every round, so the shortlist stays huge and the
    # reduce step is forced to deal with it.
    provider = Scripted({a.fingerprint(): {"rank": 1} for a in alerts})

    selected, _ = select_top_issues(alerts, provider, top_n=10, chunk_size=25)

    reduce_size = 25 * 2
    assert len(provider.calls[-1]) <= reduce_size, "the final ranking call was oversized"
    # And every selected alert came from that one call, so the ranks are
    # comparable with each other.
    final_ids = set(provider.calls[-1])
    assert {a.fingerprint() for a in selected} <= final_ids
    assert len(selected) == 10


def test_the_reduce_tournament_terminates_when_it_cannot_narrow():
    """A model that selects everything must not send the loop round forever —
    that turns a bad answer into an expensive bad answer."""
    alerts = [make_alert(f"e{i}") for i in range(300)]
    provider = Scripted({a.fingerprint(): {"selected": True, "rank": 1} for a in alerts})

    selected, _ = select_top_issues(alerts, provider, top_n=10, chunk_size=25)

    # 12 map chunks + at most MAX_REDUCE_ROUNDS worth of narrowing + 1 final.
    assert len(provider.calls) < 12 + 4 * 12 + 2
    assert len(selected) == 10


def test_every_alert_dropped_by_a_reduce_round_still_has_a_verdict():
    """Truncation and narrowing both remove alerts from the ranking. The audit
    trail has to explain them anyway — 'why was this not in the report' is the
    question it exists to answer."""
    alerts = [make_alert(f"e{i}") for i in range(120)]
    provider = Scripted({a.fingerprint(): {"rank": 1} for a in alerts})

    _selected, verdicts = select_top_issues(alerts, provider, top_n=10, chunk_size=25)

    assert len(verdicts) == len(alerts)
    assert all(v.reason for v in verdicts.values())


def test_a_small_run_skips_the_map_step_entirely():
    """Below one chunk the single reduce call is already the whole job; a
    map/reduce would pay for two passes over the same alerts."""
    alerts = [make_alert(f"e{i}") for i in range(4)]
    provider = Scripted({a.fingerprint(): {"rank": 1} for a in alerts})

    select_top_issues(alerts, provider, top_n=3, chunk_size=25)
    assert len(provider.calls) == 1


def test_a_failed_chunk_passes_its_alerts_through_rather_than_dropping_them():
    """The worst case has to be a longer shortlist, not a critical issue
    vanishing because one call timed out."""

    class FailsFirstChunk(Scripted):
        def generate_list(self, system, prompt, item_schema):
            if not self.calls:
                self.calls.append([])
                raise LLMError("garbage back from the model")
            return super().generate_list(system, prompt, item_schema)

    alerts = [make_alert(f"e{i}") for i in range(10)]
    provider = FailsFirstChunk({a.fingerprint(): {"rank": 1} for a in alerts})

    selected, _ = select_top_issues(alerts, provider, top_n=10, chunk_size=5)
    assert len(selected) == 10


def test_an_omitted_alert_survives_to_the_reduce_step():
    """Omission is not a judgement. An alert the model simply did not mention
    must not be dropped on its silence."""

    class Omits(Scripted):
        def generate_list(self, system, prompt, item_schema):
            outputs = super().generate_list(system, prompt, item_schema)
            return outputs[1:] if len(self.calls) == 1 else outputs

    alerts = [make_alert(f"e{i}") for i in range(10)]
    provider = Omits({a.fingerprint(): {"rank": 1} for a in alerts})

    selected, _ = select_top_issues(alerts, provider, top_n=10, chunk_size=5)
    assert alerts[0].fingerprint() in {a.fingerprint() for a in selected}


# --------------------------------------------------------------------------
# The payload and the prompt
# --------------------------------------------------------------------------


def test_payload_carries_the_history_the_dedup_decision_needs():
    alert = make_alert("x", similar_past=[{"alert_id": "old", "distance": 0.02}])
    assert _payload(alert)["similar_past"][0]["alert_id"] == "old"


def test_payload_sends_the_hourly_shape_not_just_the_total():
    """A sharp rise in the final buckets is happening right now; the same total
    spread evenly across the day is not. A single number cannot say which."""
    alert = make_alert("x", hourly_counts=[(1, 1), (2, 1), (3, 90)])
    assert _payload(alert)["hourly_events"] == [1, 1, 90]


def test_payload_separates_lifetime_from_24h_counts():
    alert = make_alert("x", event_count=206017, hourly_counts=[(1, 223)])
    payload = _payload(alert)
    assert (payload["events_24h"], payload["event_count_all_time"]) == (223, 206017)
    assert "event_count" not in payload


def test_the_prompt_carries_the_rules_and_the_dedup_guidance():
    prompt = build_system_prompt()
    assert "RULE 1" in prompt
    assert "similar_past" in prompt
    assert "duplicate" in prompt.lower()


def test_the_rules_are_prose_not_a_score():
    """A weight or a threshold creeping back into the prompt is a calibration
    exercise creeping back with it."""
    prompt = build_system_prompt()
    assert "impact_score" not in prompt
    assert "weight" not in prompt.lower()


def test_prompt_version_is_set():
    assert SELECTION_PROMPT_VERSION


# --------------------------------------------------------------------------
# The mock provider's stand-in behaviour
# --------------------------------------------------------------------------


def test_the_mock_ranks_on_users_then_events():
    alerts = [
        make_alert("quiet", user_count=1, hourly_counts=[(1, 5000)]),
        make_alert("broad", user_count=400, hourly_counts=[(1, 10)]),
    ]
    selected, _ = select_top_issues(alerts, MockProvider())
    assert selected[0].title == "broad"


def test_the_mock_calls_a_very_close_neighbour_a_duplicate():
    alerts = [make_alert("repeat", similar_past=[{"alert_id": "old", "distance": 0.0}])]
    selected, verdicts = select_top_issues(alerts, MockProvider())
    assert selected == []
    assert duplicate_count(verdicts) == 1


def test_payload_separates_todays_siblings_from_history():
    """Two different questions: `similar_past` decides `is_duplicate`,
    `similar_today` decides how many report slots one incident gets."""
    alert = make_alert(
        "x",
        similar_past=[{"alert_id": "yesterday", "distance": 0.02}],
        similar_today=[{"alert_id": "sibling", "distance": 0.027}],
    )
    payload = _payload(alert)
    assert payload["similar_past"][0]["alert_id"] == "yesterday"
    assert payload["similar_today"][0]["alert_id"] == "sibling"


def test_the_prompt_keeps_the_two_neighbour_sets_apart():
    """The dangerous confusion: treating a same-run sibling as evidence of a
    repeat would let one of today's alerts suppress another, and nothing in
    today's list has been reported yet."""
    prompt = build_system_prompt()
    assert "similar_today" in prompt
    # The prompt's own glossary has to say it outright. (Asserting on the first
    # occurrence would test the RULE 7 text from the YAML instead, which the
    # config owns.)
    assert "These NEVER" in prompt
    assert "has been reported yet" in prompt
    # And RULE 7 is what the field exists to serve.
    assert "similar_today" in prompt.split("RULE 7")[1][:900]
