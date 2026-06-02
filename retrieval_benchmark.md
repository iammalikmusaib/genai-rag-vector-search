# Retrieval Benchmark: Strategy A vs Strategy B

This report compares direct embedding search against mocked AI query expansion before vector search.

## Similarity Metric

The local vector store uses cosine similarity. Embeddings are L2-normalized before storage and lookup, so cosine similarity is a dot product over direction rather than magnitude. This is preferred here because semantic embedding distance is usually driven by angular similarity; Euclidean distance can overemphasize vector length unless every vector is normalized consistently.

## Production Migration to Vertex AI Vector Search

To migrate this implementation to Vertex AI Vector Search, replace the local `NumpyVectorStore` with a Matching Engine index. Keep the embedding contract stable, export normalized document embeddings with metadata IDs, create and deploy a Vertex AI Vector Search index endpoint, then route query embeddings through the endpoint with the same top-k and access-filter semantics used locally. Query expansion can move from the mocked `GenerativeModel` to Gemini while `TextEmbeddingModel` maps to the production Vertex embedding model.

## Comparison

### Query 1: How does the system handle peak load?

Expanded query: `How does the system handle peak load? peak traffic demand spikes autoscaling horizontal scaling replicas queue depth request rate p95 latency capacity backpressure operational behavior capacity controls resilience`

| Rank | Strategy A: Raw Vector Search | Score | Strategy B: AI-Enhanced Retrieval | Score |
| --- | --- | ---: | --- | ---: |
| 1 | scale-01: Peak traffic and autoscaling | 0.3519 | scale-01: Peak traffic and autoscaling | 0.4582 |
| 2 | rag-01: Retrieval augmented generation | 0.2816 | rag-01: Retrieval augmented generation | 0.2934 |
| 3 | arch-01: Service topology | 0.1317 | vertex-01: Production vector search | 0.1956 |

### Query 2: What prevents confidential tenant data from leaking during retrieval?

Expanded query: `What prevents confidential tenant data from leaking during retrieval? tenant isolation leakage access control audit encryption retrieval filters tenant isolation leakage access control audit encryption retrieval filters cross-tenant leakage access control audit encryption isolation`

| Rank | Strategy A: Raw Vector Search | Score | Strategy B: AI-Enhanced Retrieval | Score |
| --- | --- | ---: | --- | ---: |
| 1 | security-01: Security and compliance | 0.1805 | security-01: Security and compliance | 0.3104 |
| 2 | cache-01: Caching strategy | 0.1482 | cache-01: Caching strategy | 0.2887 |
| 3 | rag-01: Retrieval augmented generation | 0.1478 | arch-01: Service topology | 0.1320 |

### Query 3: How should search quality be measured before production release?

Expanded query: `How should search quality be measured before production release? retrieval evaluation recall at k mean reciprocal rank benchmark retrieved chunks relevance metrics recall at k mean reciprocal rank side-by-side review benchmark`

| Rank | Strategy A: Raw Vector Search | Score | Strategy B: AI-Enhanced Retrieval | Score |
| --- | --- | ---: | --- | ---: |
| 1 | eval-01: Retrieval evaluation | 0.1482 | eval-01: Retrieval evaluation | 0.5827 |
| 2 | vertex-01: Production vector search | 0.1478 | rag-01: Retrieval augmented generation | 0.2045 |
| 3 | arch-01: Service topology | 0.1439 | security-01: Security and compliance | 0.1446 |

## JSON Output

```json
[
  {
    "query": "How does the system handle peak load?",
    "expanded_query": "How does the system handle peak load? peak traffic demand spikes autoscaling horizontal scaling replicas queue depth request rate p95 latency capacity backpressure operational behavior capacity controls resilience",
    "strategy_a_raw_vector_search": [
      {
        "chunk_id": "scale-01",
        "title": "Peak traffic and autoscaling",
        "score": 0.3519,
        "text": "During peak traffic the system handles load by horizontally scaling API workers, adding retrieval replicas, and keeping p95 latency under 250 ms. Autoscaling policies react to CPU, queue depth, and request rate so the cluster absorbs demand spikes without manual intervention."
      },
      {
        "chunk_id": "rag-01",
        "title": "Retrieval augmented generation",
        "score": 0.2816,
        "text": "The RAG workflow chunks source documents, embeds each chunk, stores vectors in an index, retrieves the nearest passages, and sends only the grounded context to the generator. Citations are attached to every answer."
      },
      {
        "chunk_id": "arch-01",
        "title": "Service topology",
        "score": 0.1317,
        "text": "The platform is composed of stateless API workers, an event gateway, and a retrieval service. Each service publishes health metrics and structured traces so requests can be followed across the full stack."
      }
    ],
    "strategy_b_ai_enhanced_retrieval": [
      {
        "chunk_id": "scale-01",
        "title": "Peak traffic and autoscaling",
        "score": 0.4582,
        "text": "During peak traffic the system handles load by horizontally scaling API workers, adding retrieval replicas, and keeping p95 latency under 250 ms. Autoscaling policies react to CPU, queue depth, and request rate so the cluster absorbs demand spikes without manual intervention."
      },
      {
        "chunk_id": "rag-01",
        "title": "Retrieval augmented generation",
        "score": 0.2934,
        "text": "The RAG workflow chunks source documents, embeds each chunk, stores vectors in an index, retrieves the nearest passages, and sends only the grounded context to the generator. Citations are attached to every answer."
      },
      {
        "chunk_id": "vertex-01",
        "title": "Production vector search",
        "score": 0.1956,
        "text": "A production deployment can migrate from the local NumPy index to Vertex AI Vector Search by exporting normalized embeddings, creating a Matching Engine index, deploying it to an index endpoint, and routing queries through the same embedding and filtering contract."
      }
    ]
  },
  {
    "query": "What prevents confidential tenant data from leaking during retrieval?",
    "expanded_query": "What prevents confidential tenant data from leaking during retrieval? tenant isolation leakage access control audit encryption retrieval filters tenant isolation leakage access control audit encryption retrieval filters cross-tenant leakage access control audit encryption isolation",
    "strategy_a_raw_vector_search": [
      {
        "chunk_id": "security-01",
        "title": "Security and compliance",
        "score": 0.1805,
        "text": "Sensitive data is encrypted in transit and at rest. Access control lists are enforced before retrieval, and audit events record which principal queried which indexed document."
      },
      {
        "chunk_id": "cache-01",
        "title": "Caching strategy",
        "score": 0.1482,
        "text": "Frequent semantic queries are cached with their normalized embedding vectors. Cache keys include tenant, model version, and retrieval filters, which prevents cross-tenant leakage while reducing vector database reads."
      },
      {
        "chunk_id": "rag-01",
        "title": "Retrieval augmented generation",
        "score": 0.1478,
        "text": "The RAG workflow chunks source documents, embeds each chunk, stores vectors in an index, retrieves the nearest passages, and sends only the grounded context to the generator. Citations are attached to every answer."
      }
    ],
    "strategy_b_ai_enhanced_retrieval": [
      {
        "chunk_id": "security-01",
        "title": "Security and compliance",
        "score": 0.3104,
        "text": "Sensitive data is encrypted in transit and at rest. Access control lists are enforced before retrieval, and audit events record which principal queried which indexed document."
      },
      {
        "chunk_id": "cache-01",
        "title": "Caching strategy",
        "score": 0.2887,
        "text": "Frequent semantic queries are cached with their normalized embedding vectors. Cache keys include tenant, model version, and retrieval filters, which prevents cross-tenant leakage while reducing vector database reads."
      },
      {
        "chunk_id": "arch-01",
        "title": "Service topology",
        "score": 0.132,
        "text": "The platform is composed of stateless API workers, an event gateway, and a retrieval service. Each service publishes health metrics and structured traces so requests can be followed across the full stack."
      }
    ]
  },
  {
    "query": "How should search quality be measured before production release?",
    "expanded_query": "How should search quality be measured before production release? retrieval evaluation recall at k mean reciprocal rank benchmark retrieved chunks relevance metrics recall at k mean reciprocal rank side-by-side review benchmark",
    "strategy_a_raw_vector_search": [
      {
        "chunk_id": "eval-01",
        "title": "Retrieval evaluation",
        "score": 0.1482,
        "text": "Search quality is measured with recall at k, mean reciprocal rank, and side-by-side review of retrieved chunks. Evaluation queries cover ambiguous language, operational incidents, and domain-specific synonyms."
      },
      {
        "chunk_id": "vertex-01",
        "title": "Production vector search",
        "score": 0.1478,
        "text": "A production deployment can migrate from the local NumPy index to Vertex AI Vector Search by exporting normalized embeddings, creating a Matching Engine index, deploying it to an index endpoint, and routing queries through the same embedding and filtering contract."
      },
      {
        "chunk_id": "arch-01",
        "title": "Service topology",
        "score": 0.1439,
        "text": "The platform is composed of stateless API workers, an event gateway, and a retrieval service. Each service publishes health metrics and structured traces so requests can be followed across the full stack."
      }
    ],
    "strategy_b_ai_enhanced_retrieval": [
      {
        "chunk_id": "eval-01",
        "title": "Retrieval evaluation",
        "score": 0.5827,
        "text": "Search quality is measured with recall at k, mean reciprocal rank, and side-by-side review of retrieved chunks. Evaluation queries cover ambiguous language, operational incidents, and domain-specific synonyms."
      },
      {
        "chunk_id": "rag-01",
        "title": "Retrieval augmented generation",
        "score": 0.2045,
        "text": "The RAG workflow chunks source documents, embeds each chunk, stores vectors in an index, retrieves the nearest passages, and sends only the grounded context to the generator. Citations are attached to every answer."
      },
      {
        "chunk_id": "security-01",
        "title": "Security and compliance",
        "score": 0.1446,
        "text": "Sensitive data is encrypted in transit and at rest. Access control lists are enforced before retrieval, and audit events record which principal queried which indexed document."
      }
    ]
  }
]
```
