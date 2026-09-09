"""
Full, step-by-step walkthrough of the ranking pipeline using the mock
data (mock_job_requirements + mock_candidates).

Unlike test_pipeline.py (which only prints a final score summary), this
prints what happens at EVERY stage: who gets excluded by the hard
filter and why, then the BM25/semantic/hybrid scores for everyone who
survives, then the final ranking and the top 3.

Run from the project root:
    python -m ranking.tests.test_full_pipeline_verbose
"""

from ranking.hard_filter import (
    hard_filter,
    calculate_skill_coverage,
    calculate_title_match,
)
from ranking.bm25_retriever import rank_candidates as rank_bm25
from ranking.semantic_retriever import rank_candidates as rank_semantic
from ranking.hybrid_retriever import hybrid_rank

from ranking.mock_data.mock_job_requirements import mock_job_requirements
from ranking.mock_data.mock_candidates import mock_candidates


def print_header(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# =========================================================
# STEP 1 -- HARD FILTER (verbose: who's excluded and why)
# =========================================================

print_header(f"STEP 1: HARD FILTER  ({len(mock_candidates)} candidates)")

qualified_candidates = []

for candidate in mock_candidates:

    skill_coverage = calculate_skill_coverage(
        mock_job_requirements["skills"],
        candidate["skills"]
    )

    title_match = calculate_title_match(
        mock_job_requirements["related_titles"],
        candidate["job_titles"]
    )

    experience_ok = (
        candidate["experience_years"]
        >= mock_job_requirements["experience_years"]
    )

    result = hard_filter(mock_job_requirements, candidate)

    if result["qualified"]:
        qualified_candidates.append(candidate)
        decision = "QUALIFIED"
        reason = (
            f"skill coverage {skill_coverage:.0%}, "
            f"title match: {title_match}, "
            f"experience ok: {experience_ok}"
        )
    else:
        if skill_coverage < 0.30 and not title_match:
            reason = (
                f"skill coverage only {skill_coverage:.0%} "
                "and no related title match"
            )
        else:
            reason = (
                f"skill coverage only {skill_coverage:.0%} "
                f"and experience {candidate['experience_years']}y "
                f"< required {mock_job_requirements['experience_years']}y"
            )
        decision = "EXCLUDED"

    print(
        f"{candidate['candidate_id']:8} | {decision:10} | "
        f"{candidate['job_titles'][0]:28} | {reason}"
    )

print(
    f"\n-> {len(qualified_candidates)} / {len(mock_candidates)} "
    "candidates passed the hard filter."
)


# =========================================================
# STEP 2 -- BM25 RETRIEVAL (on qualified candidates only)
# =========================================================

print_header(f"STEP 2: BM25 SCORES  ({len(qualified_candidates)} candidates)")

bm25_results = rank_bm25(mock_job_requirements, qualified_candidates)

for position, result in enumerate(bm25_results, start=1):
    print(f"{position:2}. {result['candidate_id']:8} | BM25: {result['bm25_score']:.4f}")


# =========================================================
# STEP 3 -- SEMANTIC RETRIEVAL
# =========================================================

print_header(f"STEP 3: SEMANTIC SCORES  ({len(qualified_candidates)} candidates)")

semantic_results = rank_semantic(mock_job_requirements, qualified_candidates)

for position, result in enumerate(semantic_results, start=1):
    print(f"{position:2}. {result['candidate_id']:8} | Semantic: {result['semantic_score']:.4f}")


# =========================================================
# STEP 4 -- HYBRID RANKING (final combined score)
# =========================================================

print_header(f"STEP 4: FINAL HYBRID RANKING  ({len(qualified_candidates)} candidates)")

hybrid_results = hybrid_rank(
    mock_job_requirements,
    qualified_candidates,
    top_k=len(qualified_candidates)
)

for position, result in enumerate(hybrid_results, start=1):
    print(
        f"{position:2}. {result['candidate_id']:8} | "
        f"BM25: {result['bm25_score']:.4f} | "
        f"Semantic: {result['semantic_score']:.4f} | "
        f"Hybrid: {result['hybrid_score']:.4f}"
    )


# =========================================================
# STEP 5 -- TOP 3
# =========================================================

print_header("STEP 5: TOP 3")

for position, result in enumerate(hybrid_results[:3], start=1):

    candidate = result["candidate"]

    print(
        f"\n#{position}  {candidate['candidate_id']} -- "
        f"{candidate['job_titles'][0]}"
    )
    print(f"    Hybrid score: {result['hybrid_score']:.4f}")
    print(f"    Experience:   {candidate['experience_years']} years")
    print(f"    Skills:       {', '.join(candidate['skills'])}")


print("\n" + "=" * 70)
print("DONE")
print("=" * 70)