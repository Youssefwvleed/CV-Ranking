from .hard_filter import hard_filter
from .hybrid_retriever import hybrid_rank


def run_retrieval_pipeline(job, candidates, top_k=10):

    # =========================
    # 1. HARD FILTER
    # =========================

    qualified_candidates = []

    for candidate in candidates:

        result = hard_filter(
            job,
            candidate
        )

        if result["qualified"]:
            qualified_candidates.append(candidate)

    print(
        f"Hard Filter: "
        f"{len(qualified_candidates)} / {len(candidates)} qualified"
    )

    # =========================
    # 2. HYBRID RETRIEVAL
    # =========================

    if not qualified_candidates:
        return []

    results = hybrid_rank(
        job,
        qualified_candidates,
        top_k=top_k
    )

    return results
