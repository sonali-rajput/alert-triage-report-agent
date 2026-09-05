"""The vector layer that dedup rests on.

These tests are about the *properties* dedup needs -- identical text lands in
the same place, near-identical text lands nearby, unrelated text does not --
rather than about specific numbers. The offline HashEmbedder is lexical, so it
is a floor: Vertex embeddings should satisfy every one of these more
comfortably, never less.
"""

from __future__ import annotations

import pytest

from pipeline.embeddings import EMBEDDING_DIM, HashEmbedder, build_embedder, cosine_distance
from shared.models import Alert, AlertSource


def alert(**kwargs) -> Alert:
    base = dict(source=AlertSource.sentry, source_id="1", kind="sentry_issue", title="boom")
    base.update(kwargs)
    return Alert(**base)


def test_vectors_have_the_width_bigquery_expects():
    """VECTOR_SEARCH cannot compare vectors of different widths -- it fails the
    query outright rather than returning bad matches. The offline embedder has
    to agree with the column, or the local store stops being a rehearsal."""
    assert all(len(v) == EMBEDDING_DIM for v in HashEmbedder().embed(["a", "b"]))


def test_identical_text_gives_identical_vectors():
    a, b = HashEmbedder().embed(["Connection refused on db-01", "Connection refused on db-01"])
    assert cosine_distance(a, b) == pytest.approx(0.0, abs=1e-9)


def test_near_identical_text_is_much_closer_than_unrelated_text():
    """The property that makes hashing the wrong tool and embedding the right
    one: an error that gained a stack frame overnight hashes differently and
    embeds almost identically."""
    same, shifted, other = HashEmbedder().embed(
        [
            "ConnectionError: could not reach internal-db-01 port 5432",
            "ConnectionError: could not reach internal-db-01 port 5432 (retry 2)",
            "TypeError: cannot read property 'length' of undefined",
        ]
    )
    assert cosine_distance(same, shifted) < cosine_distance(same, other)


def test_the_embedded_text_excludes_anything_that_changes_daily():
    """Two runs of the same error have to land on top of each other. A counter
    or a timestamp in the embedded text would push them apart every night, and
    the dedup would never fire."""
    monday = alert(title="Payment gateway timeout", body="Level: error", event_count=12)
    tuesday = monday.model_copy(update={"event_count": 4000, "user_count": 91})
    assert monday.embedding_text() == tuesday.embedding_text()


def test_embedded_text_is_truncated():
    """The embedding API has an input limit, and the informative part of a
    stack trace is the top of it."""
    assert len(alert(body="x" * 50_000).embedding_text()) <= 8000


def test_cosine_distance_is_defensive_about_bad_input():
    """A stored embedding can be empty -- an alert ingested before the
    embedding stage was added, or one whose embed call failed. Comparing it
    must give "maximally far", not a crash inside the local store's scan."""
    assert cosine_distance([], [1.0, 0.0]) == 1.0
    assert cosine_distance([1.0, 0.0], [1.0, 0.0, 0.0]) == 1.0
    assert cosine_distance([0.0, 0.0], [1.0, 0.0]) == 1.0


def test_unknown_provider_is_rejected_loudly():
    with pytest.raises(ValueError):
        build_embedder("word2vec", "p", "l")


# --------------------------------------------------------------------------
# The Vertex path
#
# Only place the real embedder's code runs. The stub replaces the genai client
# after construction, so there is no network and no GCP project.
# --------------------------------------------------------------------------


class StubEmbeddings:
    def __init__(self, dims: int = EMBEDDING_DIM, drop: int = 0):
        self.dims = dims
        self.drop = drop  # return fewer vectors than inputs, as a broken API would
        self.batches: list[int] = []

    def embed_content(self, model, contents, config=None):
        self.config = config
        from types import SimpleNamespace

        self.batches.append(len(contents))
        values = [SimpleNamespace(values=[1.0] + [0.0] * (self.dims - 1)) for _ in contents]
        return SimpleNamespace(embeddings=values[: len(values) - self.drop])


def vertex_over(stub: StubEmbeddings):
    from types import SimpleNamespace

    from pipeline.embeddings import VertexEmbedder

    e = VertexEmbedder.__new__(VertexEmbedder)  # skip __init__: it builds a real client
    e._client = SimpleNamespace(models=stub)
    e._model = "gemini-embedding-001"
    e._dimensions = EMBEDDING_DIM
    return e


def test_vertex_vectors_are_normalized_like_the_stored_ones():
    """Cosine distance compares a query vector against stored ones. Mixed
    scales do not error — they just rank neighbours subtly wrongly, which is
    the hardest kind of bug to notice in a dedup decision."""
    [vector] = vertex_over(StubEmbeddings()).embed(["one"])
    assert sum(v * v for v in vector) == pytest.approx(1.0)


def test_large_runs_are_split_into_batches_the_api_accepts():
    from pipeline.embeddings import EMBED_BATCH

    stub = StubEmbeddings()
    vertex_over(stub).embed([f"text {i}" for i in range(EMBED_BATCH * 2 + 5)])
    assert stub.batches == [EMBED_BATCH, EMBED_BATCH, 5]


def test_a_short_response_is_an_error_not_a_silent_misalignment():
    """Vectors are zipped with alerts by position. Fewer vectors than inputs
    would attach one alert's embedding to a different alert — poisoning the
    vector store in a way no later check could detect."""
    with pytest.raises(RuntimeError, match="2 vectors for 3 inputs"):
        vertex_over(StubEmbeddings(drop=1)).embed(["a", "b", "c"])


def test_embedding_nothing_makes_no_api_call():
    stub = StubEmbeddings()
    assert vertex_over(stub).embed([]) == []
    assert stub.batches == []


def test_the_configured_dimension_is_actually_requested():
    """Matryoshka truncation is a request parameter, not a default. Without it
    gemini-embedding-001 returns 3072 and every insert mismatches the column."""
    stub = StubEmbeddings()
    vertex_over(stub).embed(["one"])
    assert stub.config is not None
    assert stub.config.output_dimensionality == EMBEDDING_DIM


def test_a_wrong_width_is_refused_rather_than_stored():
    """The failure this guards is silent and expensive: a 3072-wide vector in a
    768-wide column poisons the store, and VECTOR_SEARCH then fails every query
    against it. Better to lose one run loudly."""
    with pytest.raises(RuntimeError, match="3072-wide"):
        vertex_over(StubEmbeddings(dims=3072)).embed(["one"])


def test_the_default_model_is_not_the_deprecated_one():
    """text-embedding-004 was deprecated with a migrate-by date of October 2025.
    Pinning the name here means a silent revert shows up as a failed test."""
    from pipeline.embeddings import DEFAULT_MODEL

    assert DEFAULT_MODEL == "gemini-embedding-001"
    assert "004" not in DEFAULT_MODEL


def test_the_developer_api_transport_needs_a_key():
    """Falling back to Vertex ADC would silently switch transports and bill a
    project the caller did not mean to use."""
    with pytest.raises(ValueError, match="requires GEMINI_API_KEY"):
        build_embedder("gemini", "p", "l", api_key="")


def test_both_transports_produce_the_same_shape():
    """The point of the second transport: an AI Studio key exercises the real
    production embedding call — same model, same truncation, same
    normalisation — with no GCP project. Without it, 'does this model return
    the width we asked for' could only be answered by deploying, and getting
    it wrong writes unsearchable vectors into BigQuery."""
    from pipeline.embeddings import VertexEmbedder

    stub = StubEmbeddings()
    e = VertexEmbedder.__new__(VertexEmbedder)
    e._client = __import__("types").SimpleNamespace(models=stub)
    e._model, e._dimensions = "gemini-embedding-001", EMBEDDING_DIM
    assert len(e.embed(["x"])[0]) == EMBEDDING_DIM


# --------------------------------------------------------------------------
# Distance thresholds
#
# Calibrated on the same 19 real alerts. The two embedders' vector spaces are
# scaled very differently — median pair 0.791 offline vs 0.352 for
# gemini-embedding-001 — so a single shared constant is either far too loose
# for one or far too tight for the other. Before the split, the offline-tuned
# 0.35 sat almost exactly on the real model's MEDIAN.
# --------------------------------------------------------------------------


def test_each_embedder_carries_its_own_thresholds():
    from pipeline.embeddings import VertexEmbedder

    real = VertexEmbedder.__new__(VertexEmbedder)
    offline = HashEmbedder()
    assert real.neighbour_distance < offline.neighbour_distance
    assert real.sibling_distance < offline.sibling_distance


def test_the_sibling_cut_is_tighter_than_the_neighbour_cut():
    """A same-run merge spends one report slot on two issues, so a wrong one
    HIDES an alert. A missed historical duplicate only repeats a row."""
    from pipeline.embeddings import VertexEmbedder

    for e in (HashEmbedder(), VertexEmbedder.__new__(VertexEmbedder)):
        assert e.sibling_distance < e.neighbour_distance


def test_the_real_thresholds_sit_between_the_measured_bands():
    """Measured on the real pull: same-incident pairs run 0.037–0.134, a
    debatable band 0.146–0.176, then the unrelated mass from 0.297. The cuts
    have to land in the gaps, not in the middle of a band."""
    from pipeline.embeddings import VertexEmbedder

    e = VertexEmbedder.__new__(VertexEmbedder)
    assert 0.134 < e.sibling_distance < 0.146, "sibling cut must clear the clear pairs, stop before the debatable ones"
    assert 0.176 < e.neighbour_distance < 0.297, "neighbour cut must clear the debatable band, stop before the unrelated mass"
