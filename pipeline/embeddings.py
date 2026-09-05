"""Text embeddings for the BigQuery vector search.

One vector per alert, generated from `Alert.embedding_text()`. The vectors are
what dedup is built on: a hash comparison classes "same error, one extra stack
frame" as a new issue and "same error, one more event" as a change, which is
exactly backwards. Cosine distance in embedding space does not care about a
shifted line number.

Two implementations behind one protocol, matching the LLM provider split:

  * VertexEmbedder -- `gemini-embedding-001` through the google-genai SDK. The
    production path.
  * HashEmbedder   -- a deterministic bag-of-words hash projection. No network,
    no cost, and *good enough to be meaningful*: identical text gives identical
    vectors and near-identical text gives near-identical vectors, so the whole
    dedup path is exercisable offline. It is not semantic -- it will not see
    that "connection refused" and "could not connect" are the same thing -- so
    it is for tests and local runs only.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
from typing import Any, Protocol, cast

logger = logging.getLogger(__name__)

# `text-embedding-004` was the original choice here and it was wrong: Google
# deprecated it and told users to migrate by the end of October 2025.
# `gemini-embedding-001` is the current recommendation on Vertex AI.
#
# Changing this is not a config tweak. The model decides the vector width, and
# a stored history embedded with one model cannot be compared against a query
# vector from another -- VECTOR_SEARCH fails the query rather than returning
# bad matches, which is the good failure mode. Re-embedding the history is the
# migration; there is no in-place fix.
DEFAULT_MODEL = "gemini-embedding-001"

# The stored vector width. The BigQuery table column and the
# HashEmbedder both have to agree with it, because VECTOR_SEARCH compares a
# query vector against stored ones and mismatched widths are a query error,
# not a bad result.
# gemini-embedding-001 defaults to 3072 and supports Matryoshka truncation to
# 128..3072, with 768 / 1536 / 3072 the sizes Google recommends. 768 is chosen
# here because the vector search is a brute-force scan of the whole history
# table until it earns an index -- a quarter of the width is a quarter of the
# bytes scanned every run -- and because error text is short enough that the
# extra resolution buys little.
#
# The BigQuery `embedding` column, the offline HashEmbedder and this constant
# must all agree: VECTOR_SEARCH cannot compare vectors of different widths, and
# fails the query outright rather than returning bad matches.
EMBEDDING_DIM = 768

# The API rejects oversized batches; 100 is well inside every documented limit.
EMBED_BATCH = 100


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...

    # Distance thresholds belong to the EMBEDDER, not to the store, because
    # they are properties of the vector space it produces. Measured on the same
    # 19 real alerts, the offline embedder's median pair sits at 0.791 and
    # gemini-embedding-001's at 0.352 -- a single shared constant is therefore
    # either far too loose for one or far too tight for the other. Before this
    # split, 0.35 was tuned against the offline embedder and would have sat
    # almost exactly on the real model's MEDIAN: roughly half of all pairs
    # offered to the agent as "similar", which is noise wearing evidence's
    # clothes.
    @property
    def neighbour_distance(self) -> float:
        """Historical matches shown to the agent as `similar_past`."""
        ...

    @property
    def sibling_distance(self) -> float:
        """Same-run matches shown as `similar_today`, feeding RULE 7."""
        ...


def _normalize(vector: list[float]) -> list[float]:
    """Unit-length, so cosine distance is a dot product and every stored vector
    is on the same scale."""
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0:
        return vector
    return [v / norm for v in vector]


class VertexEmbedder:
    """Gemini embeddings through the google-genai SDK.

    Two transports, one code path -- the same arrangement as VertexProvider,
    and for the same reason. `api_key` selects the **Gemini Developer API** (an
    AI Studio key); without it the client uses **Vertex AI** with ADC, which is
    the deployed path. The request, the truncation and the normalisation are
    identical, so a Developer-API key is a genuine test of the production
    embedding call: it needs no GCP project, no ADC and no billing.

    That matters more here than it does for the LLM. Without it, "does this
    model name exist, and does it return the width we asked for" could only be
    answered by deploying -- and getting that wrong writes vectors of the wrong
    shape into a table that then cannot be searched at all.
    """

    def __init__(
        self,
        project: str,
        location: str,
        model: str = DEFAULT_MODEL,
        dimensions: int = EMBEDDING_DIM,
        api_key: str = "",
    ):
        from google import genai

        if api_key:
            self._client = genai.Client(api_key=api_key)
            logger.info("embeddings: Developer API transport (model %s)", model)
        else:
            self._client = genai.Client(vertexai=True, project=project or None, location=location)
            logger.info("embeddings: Vertex AI transport (project %s, %s)", project, model)
        self._model = model
        self._dimensions = dimensions

    # Calibrated on the real 19-issue org pull with gemini-embedding-001.
    # The distribution is compressed: min 0.037, p5 0.176, p10 0.297,
    # median 0.352, max 0.455.
    #
    #   0.037-0.134  the five pairs that are unarguably the same incident
    #   0.146-0.176  a debatable band -- a network failure and its downstream
    #                "Failed to get an ID" symptom
    #   0.297+       the dense mass of genuinely unrelated pairs
    @property
    def neighbour_distance(self) -> float:
        # Well clear of a true repeat (a same error a day later lands near 0)
        # and below where the unrelated mass begins. Being loose here costs
        # prompt noise; being tight costs a missed repeat, so it leans loose.
        return 0.25

    @property
    def sibling_distance(self) -> float:
        # Deliberately below the debatable band. A same-run merge spends one
        # report slot on two issues, so a wrong one HIDES an alert -- the
        # expensive direction. The five clear pairs are all under 0.14.
        return 0.14

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        out: list[list[float]] = []
        for start in range(0, len(texts), EMBED_BATCH):
            chunk = texts[start : start + EMBED_BATCH]
            # The SDK's `contents` overloads do not name a plain list[str],
            # which it accepts and documents; the cast keeps mypy honest about
            # everything else in this call.
            from google.genai import types

            response = self._client.models.embed_content(
                model=self._model,
                contents=cast(Any, chunk),
                # Matryoshka truncation. gemini-embedding-001 does NOT normalize
                # a truncated vector for you -- Google's docs are explicit that
                # non-3072 outputs must be normalized by the caller, or cosine
                # similarity comes out wrong. `_normalize` below does it, which
                # makes that call load-bearing rather than belt-and-braces.
                config=types.EmbedContentConfig(output_dimensionality=self._dimensions),
                # RETRIEVAL_DOCUMENT, not the default: these vectors are stored
                # and searched against, which is what that task type is for.
                # Both sides of the comparison are documents here -- today's
                # alert is matched against yesterday's stored alerts -- so the
                # same task type is used for writing and querying.
            )
            embeddings = getattr(response, "embeddings", None) or []
            if len(embeddings) != len(chunk):
                # Also the guard against a model that AGGREGATES a batch into one
                # vector instead of embedding each input -- gemini-embedding-2
                # does exactly that unless each input is wrapped in a Content
                # object, and silently returning one vector for the whole run
                # would attach one alert's meaning to every other.
                raise RuntimeError(
                    f"embedding API returned {len(embeddings)} vectors for {len(chunk)} inputs"
                )
            vectors = [_normalize(list(e.values)) for e in embeddings]
            wrong = {len(v) for v in vectors} - {self._dimensions}
            if wrong:
                raise RuntimeError(
                    f"{self._model} returned {', '.join(str(w) for w in sorted(wrong))}-wide "
                    f"vectors, expected "
                    f"{self._dimensions}. The BigQuery column and every stored embedding "
                    "assume that width; VECTOR_SEARCH cannot compare across widths."
                )
            out.extend(vectors)
        logger.info("embedded %d texts with %s", len(out), self._model)
        return out


_TOKEN = re.compile(r"[a-z0-9_]+")


class HashEmbedder:
    """Deterministic offline embedder. Hashing-trick bag of words.

    Each token is hashed into one of EMBEDDING_DIM buckets and adds its weight
    there; the result is L2-normalized. Documents sharing most of their tokens
    end up close together, which is all the dedup path needs to be testable
    without a GCP project.
    """

    def __init__(self, dim: int = EMBEDDING_DIM):
        self._dim = dim

    # Calibrated on the same 19 alerts: min 0.086, p5 0.402, median 0.791.
    # Lexical matching spreads everything much further apart than semantic
    # matching does, so these are roughly double the real model's.
    @property
    def neighbour_distance(self) -> float:
        return 0.35

    @property
    def sibling_distance(self) -> float:
        return 0.15

    def _one(self, text: str) -> list[float]:
        vector = [0.0] * self._dim
        for token in _TOKEN.findall(text.lower()):
            digest = hashlib.sha1(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % self._dim
            # Sign from a second byte so unrelated tokens can cancel rather
            # than only ever accumulate, which keeps unrelated documents from
            # drifting towards a shared "everything is positive" direction.
            vector[bucket] += 1.0 if digest[4] % 2 else -1.0
        return _normalize(vector)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]


def cosine_distance(a: list[float], b: list[float]) -> float:
    """1 - cosine similarity, matching BigQuery's COSINE distance_type, so the
    local store and BigQuery rank neighbours identically."""
    if not a or not b or len(a) != len(b):
        return 1.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 1.0
    return 1.0 - dot / (na * nb)


def build_embedder(
    provider: str,
    project: str,
    location: str,
    model: str = DEFAULT_MODEL,
    dimensions: int = EMBEDDING_DIM,
    api_key: str = "",
) -> Embedder:
    if provider == "vertex":
        return VertexEmbedder(project, location, model, dimensions)
    if provider == "gemini":
        # Developer API (AI Studio key). The real model, no GCP.
        if not api_key:
            raise ValueError("EMBEDDING_PROVIDER=gemini requires GEMINI_API_KEY")
        return VertexEmbedder(project, location, model, dimensions, api_key)
    if provider in ("hash", "mock"):
        return HashEmbedder()
    raise ValueError(f"unknown embedding provider: {provider}")
