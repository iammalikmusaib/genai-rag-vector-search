"""Mocks for the Vertex AI SDK classes used by the assessment."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer


@dataclass(frozen=True)
class TextEmbedding:
    """Small stand-in for Vertex AI's embedding response object."""

    values: list[float]


class _LocalHashingEncoder:
    """Deterministic local encoder that behaves like an embedding model."""

    def __init__(self, dimensions: int = 384) -> None:
        self._vectorizer = HashingVectorizer(
            n_features=dimensions,
            alternate_sign=False,
            norm="l2",
            lowercase=True,
            ngram_range=(1, 2),
        )

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        matrix = self._vectorizer.transform(list(texts)).astype("float32")
        return matrix.toarray()


class TextEmbeddingModel:
    """Mock of vertexai.language_models.TextEmbeddingModel."""

    def __init__(self, encoder: _LocalHashingEncoder | None = None) -> None:
        self._encoder = encoder or _LocalHashingEncoder()

    @classmethod
    def from_pretrained(cls, model_name: str) -> "TextEmbeddingModel":
        if not model_name:
            raise ValueError("model_name is required")
        return cls()

    def get_embeddings(self, texts: list[str]) -> list[TextEmbedding]:
        vectors = self._encoder.encode(texts)
        return [TextEmbedding(values=row.tolist()) for row in vectors]


@dataclass(frozen=True)
class GenerationResponse:
    text: str


class GenerativeModel:
    """Mock of vertexai.generative_models.GenerativeModel for query expansion."""

    _EXPANSIONS = {
        "peak load": (
            "peak traffic demand spikes autoscaling horizontal scaling replicas "
            "queue depth request rate p95 latency capacity backpressure"
        ),
        "system handle": "operational behavior capacity controls resilience",
        "confidential tenant": "tenant isolation leakage access control audit encryption retrieval filters",
        "tenant data": "tenant isolation leakage access control audit encryption retrieval filters",
        "leaking": "cross-tenant leakage access control audit encryption isolation",
        "vector search": "embeddings semantic similarity nearest neighbors matching engine index endpoint",
        "security": "access control audit encryption tenant data isolation compliance",
        "search quality": "retrieval evaluation recall at k mean reciprocal rank benchmark retrieved chunks relevance",
        "measured": "metrics recall at k mean reciprocal rank side-by-side review benchmark",
        "evaluation": "recall at k mean reciprocal rank benchmark retrieved chunks quality",
    }

    def __init__(self, model_name: str = "gemini-mock") -> None:
        self.model_name = model_name

    def generate_content(self, prompt: str) -> GenerationResponse:
        query = self._extract_query(prompt)
        expanded_terms = [
            expansion
            for trigger, expansion in self._EXPANSIONS.items()
            if trigger in query.lower()
        ]
        rewritten = " ".join([query, *expanded_terms]).strip()
        return GenerationResponse(text=rewritten)

    @staticmethod
    def _extract_query(prompt: str) -> str:
        match = re.search(r"query:\s*(.+)", prompt, flags=re.IGNORECASE | re.DOTALL)
        return match.group(1).strip() if match else prompt.strip()
