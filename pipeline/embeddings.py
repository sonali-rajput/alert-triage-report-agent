"""Text embeddings for the BigQuery vector search.

One vector per alert, generated from `Alert.embedding_text()`. The vectors are
what dedup is built on: a hash comparison classes "same error, one extra stack
frame" as a new issue and "same error, one more event" as a change, which is
exactly backwards. Cosine distance in embedding space does not care about a
shifted line number.

Two implementations behind one protocol, matching the LLM provider split:

  * VertexEmbedder -- `text-embedding-004` through the google-genai SDK. The
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

# text-embedding-004's native width. The BigQuery table column and the
# HashEmbedder both have to agree with it, because VECTOR_SEARCH compares a
# query vector against stored ones and mismatched widths are a query error,
# not a bad result.
EMBEDDING_DIM = 768

# The API rejects oversized batches; 100 is well inside every documented limit.
EMBED_BATCH = 100


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


def _normalize(vector: list[float]) -> list[float]:
    """Unit-length, so cosine distance is a dot product and every stored vector
    is on the same scale."""
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0:
        return vector
    return [v / norm for v in vector]


class VertexEmbedder:
    """Vertex AI embeddings via google-genai (ADC in Cloud Run)."""

    def __init__(self, project: str, location: str, model: str = "text-embedding-004"):
        from google import genai

        self._client = genai.Client(vertexai=True, project=project or None, location=location)
        self._model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        out: list[list[float]] = []
        for start in range(0, len(texts), EMBED_BATCH):
            chunk = texts[start : start + EMBED_BATCH]
            # The SDK's `contents` overloads do not name a plain list[str],
            # which it accepts and documents; the cast keeps mypy honest about
            # everything else in this call.
            response = self._client.models.embed_content(
                model=self._model, contents=cast(Any, chunk)
            )
            embeddings = getattr(response, "embeddings", None) or []
            if len(embeddings) != len(chunk):
                raise RuntimeError(
                    f"embedding API returned {len(embeddings)} vectors for {len(chunk)} inputs"
                )
            out.extend(_normalize(list(e.values)) for e in embeddings)
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


def build_embedder(provider: str, project: str, location: str, model: str = "text-embedding-004") -> Embedder:
    if provider == "vertex":
        return VertexEmbedder(project, location, model)
    if provider in ("hash", "mock"):
        return HashEmbedder()
    raise ValueError(f"unknown embedding provider: {provider}")
