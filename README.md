# Context-Aware Retrieval Engine

Local RAG/vector-search assessment implementation based on `GenAI_RAG_VectorSearch_Assessment.pdf`.

## What It Builds

- Ingests a small technical corpus of 8 paragraphs.
- Generates local deterministic embeddings through a mocked `TextEmbeddingModel`.
- Stores normalized vectors in a lightweight NumPy vector store.
- Retrieves with cosine similarity.
- Compares:
  - Strategy A: raw vector search.
  - Strategy B: mocked generative query expansion, then vector search.
- Writes `retrieval_benchmark.md` with Markdown tables and JSON output.

## Run

```powershell
python -m rag_engine.benchmark
```

## Test

```powershell
python -m pytest
```

If pytest is not installed in the active Python environment, install the project requirements first.

<img width="1903" height="988" alt="Screenshot 2026-06-02 195607" src="https://github.com/user-attachments/assets/512a85da-3392-4bd6-8657-def58e94c0a3" />


## Files

- `rag_engine/embeddings.py`: Vertex-style embedding adapter.
- `rag_engine/vector_store.py`: NumPy cosine similarity store.
- `rag_engine/retrieval.py`: ingestion, raw retrieval, expanded retrieval.
- `rag_engine/vertex_mocks.py`: mocked `TextEmbeddingModel` and `GenerativeModel`.
- `rag_engine/benchmark.py`: benchmark/report generation.
- `tests/test_retrieval.py`: pytest coverage for retrieval and mocks.
