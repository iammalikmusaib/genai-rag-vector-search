"""Benchmark Strategy A versus Strategy B and write the assessment report."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .retrieval import RAGPipeline, RetrievalResult


BENCHMARK_QUERIES = [
    "How does the system handle peak load?",
    "What prevents confidential tenant data from leaking during retrieval?",
    "How should search quality be measured before production release?",
]


def run_benchmark() -> list[dict[str, object]]:
    pipeline = RAGPipeline()
    pipeline.ingest()
    return pipeline.compare(BENCHMARK_QUERIES, top_k=3)


def result_to_dict(result: RetrievalResult) -> dict[str, object]:
    return asdict(result)


def report_as_json(report: list[dict[str, object]]) -> str:
    serializable = []
    for row in report:
        serializable.append(
            {
                "query": row["query"],
                "expanded_query": row["expanded_query"],
                "strategy_a_raw_vector_search": [
                    result_to_dict(result)
                    for result in row["strategy_a_raw_vector_search"]
                ],
                "strategy_b_ai_enhanced_retrieval": [
                    result_to_dict(result)
                    for result in row["strategy_b_ai_enhanced_retrieval"]
                ],
            }
        )
    return json.dumps(serializable, indent=2)


def report_as_markdown(report: list[dict[str, object]]) -> str:
    lines = [
        "# Retrieval Benchmark: Strategy A vs Strategy B",
        "",
        "This report compares direct embedding search against mocked AI query expansion before vector search.",
        "",
        "## Similarity Metric",
        "",
        "The local vector store uses cosine similarity. Embeddings are L2-normalized before storage and lookup, so cosine similarity is a dot product over direction rather than magnitude. This is preferred here because semantic embedding distance is usually driven by angular similarity; Euclidean distance can overemphasize vector length unless every vector is normalized consistently.",
        "",
        "## Production Migration to Vertex AI Vector Search",
        "",
        "To migrate this implementation to Vertex AI Vector Search, replace the local `NumpyVectorStore` with a Matching Engine index. Keep the embedding contract stable, export normalized document embeddings with metadata IDs, create and deploy a Vertex AI Vector Search index endpoint, then route query embeddings through the endpoint with the same top-k and access-filter semantics used locally. Query expansion can move from the mocked `GenerativeModel` to Gemini while `TextEmbeddingModel` maps to the production Vertex embedding model.",
        "",
        "## Comparison",
        "",
    ]
    for index, row in enumerate(report, 1):
        lines.extend(
            [
                f"### Query {index}: {row['query']}",
                "",
                f"Expanded query: `{row['expanded_query']}`",
                "",
                "| Rank | Strategy A: Raw Vector Search | Score | Strategy B: AI-Enhanced Retrieval | Score |",
                "| --- | --- | ---: | --- | ---: |",
            ]
        )
        raw_results = row["strategy_a_raw_vector_search"]
        expanded_results = row["strategy_b_ai_enhanced_retrieval"]
        for rank, (raw, expanded) in enumerate(zip(raw_results, expanded_results), 1):
            lines.append(
                "| {rank} | {raw_id}: {raw_title} | {raw_score:.4f} | {expanded_id}: {expanded_title} | {expanded_score:.4f} |".format(
                    rank=rank,
                    raw_id=raw.chunk_id,
                    raw_title=raw.title,
                    raw_score=raw.score,
                    expanded_id=expanded.chunk_id,
                    expanded_title=expanded.title,
                    expanded_score=expanded.score,
                )
            )
        lines.append("")
    lines.extend(["## JSON Output", "", "```json", report_as_json(report), "```", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="retrieval_benchmark.md",
        help="Path to write the Markdown benchmark report.",
    )
    args = parser.parse_args()
    report = run_benchmark()
    output_path = Path(args.output)
    output_path.write_text(report_as_markdown(report), encoding="utf-8")
    print(report_as_json(report))


if __name__ == "__main__":
    main()
