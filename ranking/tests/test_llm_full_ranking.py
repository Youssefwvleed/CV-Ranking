import json

from ranking.pipeline import run_retrieval_pipeline
from ranking.ollama_explainer import verify_and_explain_ranking

from ranking.mock_data.mock_job_requirements import (
    mock_job_requirements
)

from ranking.mock_data.mock_candidates import (
    mock_candidates
)


# ============================================================
# 1. RETRIEVAL PIPELINE
# ============================================================

print("\n")
print("=" * 80)
print("CV RANKING PIPELINE TEST")
print("=" * 80)

print(f"\nTotal Mock CVs: {len(mock_candidates)}")

results = run_retrieval_pipeline(
    mock_job_requirements,
    mock_candidates,
    top_k=10
)


# ============================================================
# 2. RETRIEVAL RESULTS
# ============================================================

print("\n")
print("=" * 80)
print("HYBRID RANKING")
print("=" * 80)

for position, result in enumerate(results, start=1):

    print(
        f"{position:2}. "
        f"{result['candidate_id']} | "
        f"BM25: {result['bm25_score']:.4f} | "
        f"Semantic: {result['semantic_score']:.4f} | "
        f"Hybrid: {result['hybrid_score']:.4f}"
    )


# ============================================================
# 3. PREPARE TOP 10 FOR LLM
# ============================================================

ranked_candidates = []

for result in results:

    candidate_id = result["candidate_id"]

    candidate = next(
        (
            cv
            for cv in mock_candidates
            if cv["candidate_id"] == candidate_id
        ),
        None
    )

    if candidate is None:
        continue

    ranked_candidates.append({
        "candidate_id": candidate_id,
        "candidate": candidate,
        "bm25_score": result["bm25_score"],
        "semantic_score": result["semantic_score"],
        "hybrid_score": result["hybrid_score"]
    })


# ============================================================
# 4. LLM VERIFICATION + EXPLANATION
# ============================================================

print("\n")
print("=" * 80)
print("LLM VERIFICATION + EXPLANATION")
print("=" * 80)

print("\nAnalyzing Top 10 with LLM...")

llm_result = verify_and_explain_ranking(
    mock_job_requirements,
    ranked_candidates
)


# ============================================================
# 5. RANKING VERIFICATION
# ============================================================

print("\n")
print("=" * 80)
print("RANKING VERIFICATION")
print("=" * 80)

ranking_valid = llm_result.get(
    "ranking_valid"
)

print(
    f"\nRanking Valid: {ranking_valid}"
)

corrected_order = llm_result.get(
    "corrected_order",
    []
)

if corrected_order:

    print("\nCorrected Order:")

    for position, candidate_id in enumerate(
        corrected_order,
        start=1
    ):
        print(
            f"{position:2}. {candidate_id}"
        )

else:

    print("\nNo ranking correction needed.")


# ============================================================
# 6. INDEX LLM EXPLANATIONS
# ============================================================

explanations = {
    item["candidate_id"]: item
    for item in llm_result.get(
        "candidates",
        []
    )
}


# ============================================================
# 7. FINAL CANDIDATE SUMMARIES
# ============================================================

print("\n")
print("=" * 80)
print("FINAL CANDIDATE SUMMARIES")
print("=" * 80)

for position, result in enumerate(
    ranked_candidates,
    start=1
):

    candidate_id = result["candidate_id"]

    explanation = explanations.get(
        candidate_id,
        {}
    )

    print("\n")
    print("-" * 80)

    print(
        f"RANK #{position} | {candidate_id}"
    )

    print("-" * 80)

    print(
        f"Hybrid Score: "
        f"{result['hybrid_score']:.4f}"
    )

    print("\nMatching Skills:")

    for skill in explanation.get(
        "matching_skills",
        []
    ):
        print(f"  + {skill}")

    print("\nMissing Skills:")

    for skill in explanation.get(
        "missing_skills",
        []
    ):
        print(f"  - {skill}")

    print("\nPros:")

    for pro in explanation.get(
        "pros",
        []
    ):
        print(f"  + {pro}")

    print("\nCons:")

    for con in explanation.get(
        "cons",
        []
    ):
        print(f"  - {con}")

    print("\nExperience Summary:")

    print(
        explanation.get(
            "experience_summary",
            ""
        )
    )

    print("\nRecommendation:")

    print(
        explanation.get(
            "recommendation",
            ""
        )
    )


# ============================================================
# 8. OPTIONAL RAW JSON
# ============================================================

print("\n")
print("=" * 80)
print("RAW LLM RESULT")
print("=" * 80)

print(
    json.dumps(
        llm_result,
        indent=2,
        ensure_ascii=False
    )
)