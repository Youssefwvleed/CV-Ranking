from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from .tests.text_builder import build_job_text, build_candidate_text


MODEL_NAME = "BAAI/bge-base-en-v1.5"

model = SentenceTransformer(MODEL_NAME)


def rank_candidates(job, candidates):

    job_text = build_job_text(job)

    candidate_texts = [
        build_candidate_text(candidate)
        for candidate in candidates
    ]

    job_embedding = model.encode(
        [job_text],
        normalize_embeddings=True
    )

    candidate_embeddings = model.encode(
        candidate_texts,
        normalize_embeddings=True
    )

    scores = cosine_similarity(
        job_embedding,
        candidate_embeddings
    )[0]

    results = []

    for candidate, score in zip(candidates, scores):
        results.append({
            "candidate_id": candidate["candidate_id"],
            "semantic_score": float(score)
        })

    results.sort(
        key=lambda x: x["semantic_score"],
        reverse=True
    )

    return results