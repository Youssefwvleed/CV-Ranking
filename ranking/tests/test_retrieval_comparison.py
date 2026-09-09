from ranking.hard_filter import hard_filter

from ranking.bm25_retriever import rank_candidates as rank_bm25
from ranking.semantic_retriever import rank_candidates as rank_semantic
from ranking.hybrid_retriever import hybrid_rank

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


# ==========================================
# 1. HARD FILTER
# ==========================================

qualified_candidates = []

for candidate in mock_candidates:

    result = hard_filter(
        mock_job_requirements,
        candidate
    )

    if result["qualified"]:
        qualified_candidates.append(candidate)


print(
    f"Hard Filter: "
    f"{len(qualified_candidates)} / "
    f"{len(mock_candidates)} qualified"
)


# ==========================================
# 2. BM25
# ==========================================

bm25_results = rank_bm25(
    mock_job_requirements,
    qualified_candidates
)


# ==========================================
# 3. SEMANTIC
# ==========================================

semantic_results = rank_semantic(
    mock_job_requirements,
    qualified_candidates
)


# ==========================================
# 4. HYBRID
# ==========================================

hybrid_results = hybrid_rank(
    mock_job_requirements,
    qualified_candidates,
    top_k=10
)


# ==========================================
# Evaluation Helper
# ==========================================

def evaluate(name, results):

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

    print(f"\n{name}")
    print("-" * 40)
    print(f"Precision@10: {precision:.2f}")
    print(f"Recall@10:    {recall:.2f}")
    print(f"NDCG@10:      {ndcg:.2f}")


# ==========================================
# Compare
# ==========================================

evaluate(
    "BM25",
    bm25_results
)

evaluate(
    "Semantic",
    semantic_results
)

evaluate(
    "Hybrid",
    hybrid_results
)