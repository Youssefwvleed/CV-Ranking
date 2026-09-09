import json

from ranking.pipeline import run_retrieval_pipeline

from ranking.llm_explainer import (
    verify_and_explain_ranking
)

from ranking.mock_data.mock_job_requirements import (
    mock_job_requirements
)

from ranking.mock_data.mock_candidates import (
    mock_candidates
)


# ==========================================
# 1. RUN RETRIEVAL PIPELINE
# ==========================================

results = run_retrieval_pipeline(
    mock_job_requirements,
    mock_candidates,
    top_k=10
)


print("\nTOP 10 CANDIDATES")
print("=" * 80)


# ==========================================
# 2. PREPARE CANDIDATES FOR LLM
# ==========================================

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


# ==========================================
# 3. SEND TOP 10 TO LLM
# ==========================================

print("\nAnalyzing Top 10 with LLM...")

llm_result = verify_and_explain_ranking(
    mock_job_requirements,
    ranked_candidates
)


# ==========================================
# 4. PRINT RANKING VERIFICATION
# ==========================================

print("\n")
print("=" * 80)
print("LLM RANKING VERIFICATION")
print("=" * 80)

print(
    f"\nRanking Valid: "
    f"{llm_result.get('ranking_valid')}"
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
            f"{position}. {candidate_id}"
        )

else:

    print("\nNo ranking correction needed.")


# ==========================================
# 5. INDEX EXPLANATIONS BY CANDIDATE ID
# ==========================================

explanations = {
    item["candidate_id"]: item
    for item in llm_result.get(
        "candidates",
        []
    )
}


# ==========================================
# 6. PRINT CANDIDATE EXPLANATIONS
# ==========================================

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
    print("=" * 80)

    print(
        f"RANK #{position} | "
        f"{candidate_id}"
    )

    print("=" * 80)

    print(
        f"Hybrid Score: "
        f"{result['hybrid_score']:.4f}"
    )

    print(
        f"BM25 Score: "
        f"{result['bm25_score']:.4f}"
    )

    print(
        f"Semantic Score: "
        f"{result['semantic_score']:.4f}"
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


# ==========================================
# 7. OPTIONAL: RAW JSON
# ==========================================

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