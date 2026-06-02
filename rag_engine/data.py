"""Seed technical corpus for the local RAG assessment."""

TECHNICAL_PARAGRAPHS = [
    {
        "id": "arch-01",
        "title": "Service topology",
        "text": (
            "The platform is composed of stateless API workers, an event gateway, "
            "and a retrieval service. Each service publishes health metrics and "
            "structured traces so requests can be followed across the full stack."
        ),
    },
    {
        "id": "scale-01",
        "title": "Peak traffic and autoscaling",
        "text": (
            "During peak traffic the system handles load by horizontally scaling "
            "API workers, adding retrieval replicas, and keeping p95 latency under "
            "250 ms. Autoscaling policies react to CPU, queue depth, and request "
            "rate so the cluster absorbs demand spikes without manual intervention."
        ),
    },
    {
        "id": "queue-01",
        "title": "Backpressure controls",
        "text": (
            "The ingestion path uses bounded queues and backpressure. When upstream "
            "publishers exceed downstream capacity, the gateway slows producers, "
            "prioritizes interactive requests, and sheds noncritical batch jobs."
        ),
    },
    {
        "id": "cache-01",
        "title": "Caching strategy",
        "text": (
            "Frequent semantic queries are cached with their normalized embedding "
            "vectors. Cache keys include tenant, model version, and retrieval "
            "filters, which prevents cross-tenant leakage while reducing vector "
            "database reads."
        ),
    },
    {
        "id": "rag-01",
        "title": "Retrieval augmented generation",
        "text": (
            "The RAG workflow chunks source documents, embeds each chunk, stores "
            "vectors in an index, retrieves the nearest passages, and sends only "
            "the grounded context to the generator. Citations are attached to every "
            "answer."
        ),
    },
    {
        "id": "security-01",
        "title": "Security and compliance",
        "text": (
            "Sensitive data is encrypted in transit and at rest. Access control "
            "lists are enforced before retrieval, and audit events record which "
            "principal queried which indexed document."
        ),
    },
    {
        "id": "eval-01",
        "title": "Retrieval evaluation",
        "text": (
            "Search quality is measured with recall at k, mean reciprocal rank, "
            "and side-by-side review of retrieved chunks. Evaluation queries cover "
            "ambiguous language, operational incidents, and domain-specific synonyms."
        ),
    },
    {
        "id": "vertex-01",
        "title": "Production vector search",
        "text": (
            "A production deployment can migrate from the local NumPy index to "
            "Vertex AI Vector Search by exporting normalized embeddings, creating "
            "a Matching Engine index, deploying it to an index endpoint, and routing "
            "queries through the same embedding and filtering contract."
        ),
    },
]
