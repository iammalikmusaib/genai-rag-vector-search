from rag_engine.embeddings import EmbeddingService
from rag_engine.retrieval import QueryExpander, RAGPipeline
from rag_engine.vertex_mocks import GenerativeModel, TextEmbeddingModel


def test_pipeline_ingests_and_returns_top_k_results():
    pipeline = RAGPipeline()
    pipeline.ingest()

    results = pipeline.search_raw("How does the system handle peak load?", top_k=3)

    assert len(results) == 3
    assert all(result.score >= 0 for result in results)
    assert {result.chunk_id for result in results}


def test_query_expansion_improves_peak_load_retrieval():
    pipeline = RAGPipeline()
    pipeline.ingest()

    raw_results = pipeline.search_raw("How does the system handle peak load?", top_k=3)
    expanded_query, expanded_results = pipeline.search_expanded(
        "How does the system handle peak load?",
        top_k=3,
    )

    assert "autoscaling" in expanded_query
    assert expanded_results[0].chunk_id == "scale-01"
    assert raw_results != expanded_results


def test_mock_text_embedding_model_is_used_by_embedding_service():
    service = EmbeddingService(model_cls=TextEmbeddingModel)

    vectors = service.embed_texts(["one search query", "another search query"])

    assert vectors.shape[0] == 2
    assert vectors.shape[1] == 384


def test_mock_generative_model_expands_domain_language():
    expander = QueryExpander(model=GenerativeModel())

    expanded = expander.expand("How does the system handle peak load?")

    assert "peak traffic" in expanded
    assert "queue depth" in expanded
