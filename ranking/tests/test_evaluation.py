from ranking.pipeline import run_retrieval_pipeline

from ranking.evaluator import (
    precision_at_k,
    recall_at_k,
    ndcg_at_k
)

from ranking.mock_data.mock_job_requirements import (
    mock_job_requirements
)

from ranking.mock_data.mock_candidates import (
    mock_candidates
)

from ranking.evaluation_data import ground_truth


results = run_retrieval_pipeline(
    mock_job_requirements,
    mock_candidates,
    top_k=10
)


precision = precision_at_k(
    results,
    ground_truth,
    k=10
)

recall = recall_at_k(
    results,
    ground_truth,
    k=10
)

ndcg = ndcg_at_k(
    results,
    ground_truth,
    k=10
)


print("\nHYBRID RANKING")
print("=" * 70)

for position, result in enumerate(results, start=1):

    candidate_id = result["candidate_id"]

    label = ground_truth.get(
        candidate_id,
        0
    )

    print(
        f"{position}. "
        f"{candidate_id} | "
        f"Ground Truth: {label} | "
        f"BM25: {result['bm25_score']:.4f} | "
        f"Semantic: {result['semantic_score']:.4f} | "
        f"Hybrid: {result['hybrid_score']:.4f}"
    )


print("\nEVALUATION")
print("=" * 50)

print(f"Precision@10: {precision:.2f}")
print(f"Recall@10:    {recall:.2f}")
print(f"NDCG@10:      {ndcg:.2f}")