from ranking.hybrid_retriever import hybrid_rank

from ranking.mock_data.mock_job_requirements import (
    mock_job_requirements
)

from ranking.mock_data.mock_candidates import (
    mock_candidates
)


results = hybrid_rank(
    mock_job_requirements,
    mock_candidates,
    top_k=10
)


print("\nHYBRID TOP 10")
print("=" * 70)

for index, result in enumerate(results, start=1):

    print(
        f"{index}. "
        f"{result['candidate_id']} | "
        f"BM25: {result['bm25_score']:.4f} | "
        f"Semantic: {result['semantic_score']:.4f} | "
        f"Hybrid: {result['hybrid_score']:.4f}"
    )