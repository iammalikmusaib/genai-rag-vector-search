"""Embedding adapters for local RAG retrieval."""

from __future__ import annotations

import numpy as np

from .vertex_mocks import TextEmbeddingModel


class EmbeddingService:
    """Produces normalized embedding vectors through a Vertex-style interface."""

    def __init__(
        self,
        model_name: str = "textembedding-gecko-mock",
        model_cls: type[TextEmbeddingModel] = TextEmbeddingModel,
    ) -> None:
        self.model = model_cls.from_pretrained(model_name)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            raise ValueError("texts must not be empty")
        embeddings = self.model.get_embeddings(texts)
        vectors = np.asarray([item.values for item in embeddings], dtype=np.float32)
        return self._normalize(vectors)

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms
