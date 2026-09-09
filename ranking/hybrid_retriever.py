from .bm25_retriever import rank_candidates as rank_bm25
from .semantic_retriever import rank_candidates as rank_semantic


BM25_WEIGHT = 0.4
SEMANTIC_WEIGHT = 0.6


def normalize_scores(scores):
    if not scores:
        return {}

    min_score = min(scores.values())
    max_score = max(scores.values())

    if max_score == min_score:
        return {
            candidate_id: 1.0
            for candidate_id in scores
        }

    return {
        candidate_id: (score - min_score) / (max_score - min_score)
        for candidate_id, score in scores.items()
    }


def hybrid_rank(job, candidates, top_k=15):

    # 1. BM25
    bm25_results = rank_bm25(job, candidates)

    bm25_scores = {
        result["candidate_id"]: result["bm25_score"]
        for result in bm25_results
    }

    # 2. Semantic
    semantic_results = rank_semantic(job, candidates)

    semantic_scores = {
        result["candidate_id"]: result["semantic_score"]
        for result in semantic_results
    }

    # 3. Normalize BM25
    normalized_bm25 = normalize_scores(bm25_scores)

    # 4. Combine
    results = []

    for candidate in candidates:

        candidate_id = candidate["candidate_id"]

        bm25_score = normalized_bm25.get(candidate_id, 0.0)
        semantic_score = semantic_scores.get(candidate_id, 0.0)

        hybrid_score = (
            BM25_WEIGHT * bm25_score
            + SEMANTIC_WEIGHT * semantic_score
        )

        results.append({
            "candidate_id": candidate_id,
            "candidate": candidate,
            "bm25_score": bm25_score,
            "semantic_score": semantic_score,
            "hybrid_score": hybrid_score
        })

    # 5. Rank
    results.sort(
        key=lambda x: x["hybrid_score"],
        reverse=True
    )

    # 6. Return only Top K
    return results[:top_k]