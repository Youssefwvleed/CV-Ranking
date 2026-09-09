from rank_bm25 import BM25Okapi

from .tests.text_builder import build_job_text, build_candidate_text


def tokenize(text):
    return text.lower().split()


def rank_candidates(job, candidates):
    job_text = build_job_text(job)

    candidate_texts = [
        build_candidate_text(candidate)
        for candidate in candidates
    ]

    tokenized_candidates = [
        tokenize(text)
        for text in candidate_texts
    ]

    bm25 = BM25Okapi(tokenized_candidates)

    query = tokenize(job_text)

    scores = bm25.get_scores(query)

    results = []

    for candidate, score in zip(candidates, scores):
        results.append({
            "candidate_id": candidate["candidate_id"],
            "bm25_score": float(score)
        })

    results.sort(
        key=lambda x: x["bm25_score"],
        reverse=True
    )

    return results