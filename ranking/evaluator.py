import math


def precision_at_k(results, ground_truth, k=10):

    top_k = results[:k]

    if not top_k:
        return 0.0

    relevant_count = sum(
        1
        for result in top_k
        if ground_truth.get(
            result["candidate_id"],
            0
        ) > 0
    )

    return relevant_count / len(top_k)


def recall_at_k(results, ground_truth, k=10):

    top_k = results[:k]

    total_relevant = sum(
        1 for score in ground_truth.values()
        if score > 0
    )

    if total_relevant == 0:
        return 0.0

    relevant_retrieved = sum(
        1
        for result in top_k
        if ground_truth.get(
            result["candidate_id"],
            0
        ) > 0
    )

    return relevant_retrieved / total_relevant

def ndcg_at_k(results, ground_truth, k=10):
    top_k = results[:k]

    dcg = 0.0

    for position, result in enumerate(top_k, start=1):
        relevance = ground_truth.get(
            result["candidate_id"],
            0
        )

        dcg += relevance / math.log2(position + 1)

    ideal_relevances = sorted(
        ground_truth.values(),
        reverse=True
    )[:k]

    idcg = 0.0

    for position, relevance in enumerate(
        ideal_relevances,
        start=1
    ):
        idcg += relevance / math.log2(position + 1)

    if idcg == 0:
        return 0.0

    return dcg / idcg