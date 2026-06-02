"""A small NumPy vector store using cosine similarity."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StoredChunk:
    chunk_id: str
    title: str
    text: str


@dataclass(frozen=True)
class SearchHit:
    chunk: StoredChunk
    score: float


class NumpyVectorStore:
    """In-memory vector index for normalized embeddings."""

    def __init__(self) -> None:
        self._chunks: list[StoredChunk] = []
        self._vectors: np.ndarray | None = None

    def add(self, chunks: list[StoredChunk], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        self._chunks = list(chunks)
        self._vectors = np.asarray(vectors, dtype=np.float32)

    def search(self, query_vector: np.ndarray, top_k: int = 3) -> list[SearchHit]:
        if self._vectors is None or not self._chunks:
            raise RuntimeError("vector store has not been populated")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        scores = self._vectors @ query
        indices = np.argsort(scores)[::-1][:top_k]
        return [
            SearchHit(chunk=self._chunks[index], score=float(scores[index]))
            for index in indices
        ]
