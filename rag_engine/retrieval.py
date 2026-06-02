"""RAG orchestration and retrieval strategies."""

from __future__ import annotations

from dataclasses import dataclass

from .data import TECHNICAL_PARAGRAPHS
from .embeddings import EmbeddingService
from .vector_store import NumpyVectorStore, StoredChunk
from .vertex_mocks import GenerativeModel


@dataclass(frozen=True)
class RetrievalResult:
    chunk_id: str
    title: str
    score: float
    text: str


class QueryExpander:
    """Uses a mocked generative model to rewrite a query for embedding search."""

    def __init__(self, model: GenerativeModel | None = None) -> None:
        self.model = model or GenerativeModel("gemini-query-expansion-mock")

    def expand(self, query: str) -> str:
        prompt = (
            "Rewrite this user query with operational synonyms and retrieval "
            f"keywords while preserving intent.\nQuery: {query}"
        )
        return self.model.generate_content(prompt).text


class RAGPipeline:
    """Coordinates corpus ingestion, embedding, search, and query expansion."""

    def __init__(
        self,
        embedding_service: EmbeddingService | None = None,
        vector_store: NumpyVectorStore | None = None,
        query_expander: QueryExpander | None = None,
    ) -> None:
        self.embedding_service = embedding_service or EmbeddingService()
        self.vector_store = vector_store or NumpyVectorStore()
        self.query_expander = query_expander or QueryExpander()

    def ingest(self, dataset: list[dict[str, str]] | None = None) -> None:
        records = dataset or TECHNICAL_PARAGRAPHS
        chunks = [
            StoredChunk(
                chunk_id=record["id"],
                title=record["title"],
                text=record["text"],
            )
            for record in records
        ]
        vectors = self.embedding_service.embed_texts([chunk.text for chunk in chunks])
        self.vector_store.add(chunks, vectors)

    def search_raw(self, query: str, top_k: int = 3) -> list[RetrievalResult]:
        return self._search(query=query, top_k=top_k)

    def search_expanded(self, query: str, top_k: int = 3) -> tuple[str, list[RetrievalResult]]:
        expanded_query = self.query_expander.expand(query)
        return expanded_query, self._search(query=expanded_query, top_k=top_k)

    def compare(self, queries: list[str], top_k: int = 3) -> list[dict[str, object]]:
        report = []
        for query in queries:
            expanded_query, expanded_results = self.search_expanded(query, top_k=top_k)
            report.append(
                {
                    "query": query,
                    "expanded_query": expanded_query,
                    "strategy_a_raw_vector_search": self.search_raw(query, top_k=top_k),
                    "strategy_b_ai_enhanced_retrieval": expanded_results,
                }
            )
        return report

    def _search(self, query: str, top_k: int) -> list[RetrievalResult]:
        query_vector = self.embedding_service.embed_texts([query])[0]
        hits = self.vector_store.search(query_vector, top_k=top_k)
        return [
            RetrievalResult(
                chunk_id=hit.chunk.chunk_id,
                title=hit.chunk.title,
                score=round(hit.score, 4),
                text=hit.chunk.text,
            )
            for hit in hits
        ]
