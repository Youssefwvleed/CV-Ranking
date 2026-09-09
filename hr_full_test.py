import json
import time

from pathlib import Path
from datetime import datetime


# ============================================================
# PROJECT IMPORTS
# ============================================================

from main import extract_text_from_pdf

from parser_engine import (
    parse_resume_with_fallback
)

from candidate_adapter import (
    adapt_candidate
)

from ranking.hard_filter import (
    hard_filter
)

from ranking.hybrid_retriever import (
    hybrid_rank
)

from ranking.llm_explainer import (
    verify_and_explain_ranking
)


# ============================================================
# CONFIG
# ============================================================

CV_FOLDER = Path("test_cvs")

RESULTS_FOLDER = Path("results")

TOP_K = 10


# ============================================================
# HR JOB
# ============================================================

JOB = {

    "title": "Sales Executive",

    "description": """
    We are looking for a Sales Executive to join our sales team.

    The candidate will be responsible for identifying new business
    opportunities, communicating with prospective and existing
    customers, presenting products and services, following up on
    leads, negotiating with clients, closing sales, and maintaining
    strong customer relationships.

    Relevant experience may include B2B or B2C sales, retail sales,
    account management, business development, territory sales,
    customer-facing sales, lead generation, sales support,
    customer retention, and CRM systems.

    Strong communication, customer service, negotiation,
    relationship building, persuasion, and sales skills
    are valuable for this role.
    """,

    "skills": [
        "Sales",
        "Customer Service",
        "Communication",
        "Negotiation",
        "Lead Generation",
        "Business Development",
        "CRM"
    ],

    "related_titles": [
        "Sales Executive",
        "Sales Representative",
        "Sales Associate",
        "Sales Specialist",
        "Sales Consultant",
        "Sales Advisor",
        "Sales Coordinator",
        "Sales Manager",
        "Account Executive",
        "Account Manager",
        "Business Development Representative",
        "Business Development Executive",
        "Business Development Specialist",
        "Territory Sales Representative",
        "Retail Sales Associate",
        "Sales Engineer"
    ],

    "experience_years": 1,

    "experience_level": "entry_mid"
}


# ============================================================
# DISPLAY HELPERS
# ============================================================

def header(title):

    print()

    print(
        "=" * 70
    )

    print(title)

    print(
        "=" * 70
    )


# ============================================================
# FIND CV FILES
# ============================================================

def find_cv_files():

    if not CV_FOLDER.exists():

        raise FileNotFoundError(
            f"CV folder not found: {CV_FOLDER}"
        )

    files = sorted(
        CV_FOLDER.glob("*.pdf")
    )

    if not files:

        raise FileNotFoundError(
            f"No PDF CVs found inside: {CV_FOLDER}"
        )

    return files


# ============================================================
# PROCESS ONE CV
# ============================================================

def process_cv(
    pdf_path,
    position,
    total
):

    print()

    print(
        f"[{position}/{total}]"
    )

    print()

    print(
        "-" * 70
    )

    print(
        f"Processing: {pdf_path.name}"
    )

    start_time = time.time()

    try:

        # ====================================================
        # 1. PDF -> TEXT
        # ====================================================

        pdf_bytes = pdf_path.read_bytes()

        text = extract_text_from_pdf(
            pdf_bytes
        )

        if not text or not text.strip():

            raise RuntimeError(
                "No text extracted from PDF"
            )

        print(
            f"  Text extracted: "
            f"{len(text)} characters"
        )

        # ====================================================
        # 2. LOCAL RESUME PARSER
        # ====================================================

        parsed_resume, parser_used = (
            parse_resume_with_fallback(
                text
            )
        )

        # ====================================================
        # 3. ADAPT TO RANKING SCHEMA
        # ====================================================

        candidate = adapt_candidate(

            parsed_resume=parsed_resume,

            candidate_id=pdf_path.stem
        )

        elapsed = (
            time.time()
            - start_time
        )

        print(
            f"  Parser: {parser_used}"
        )

        print(
            f"  Candidate ID: "
            f"{candidate['candidate_id']}"
        )

        print(
            f"  Experience: "
            f"{candidate['experience_years']}"
        )

        print(
            f"  Job Titles: "
            f"{candidate['job_titles']}"
        )

        print(
            f"  Skills: "
            f"{candidate['skills']}"
        )

        print(
            f"  Time: "
            f"{elapsed:.2f}s"
        )

        return {
            "success": True,

            "filename": pdf_path.name,

            "candidate": candidate,

            "parser": parser_used,

            "processing_time": round(
                elapsed,
                2
            ),

            "error": None
        }

    except Exception as e:

        elapsed = (
            time.time()
            - start_time
        )

        print(
            f"  FAILED: {e}"
        )

        return {
            "success": False,

            "filename": pdf_path.name,

            "candidate": None,

            "parser": None,

            "processing_time": round(
                elapsed,
                2
            ),

            "error": str(e)
        }


# ============================================================
# PROCESS ALL CVS
# ============================================================

def process_all_cvs(
    files
):

    header(
        "LOCAL CV PARSING"
    )

    results = []

    total = len(files)

    for position, pdf_path in enumerate(
        files,
        start=1
    ):

        result = process_cv(
            pdf_path,
            position,
            total
        )

        results.append(
            result
        )

    return results


# ============================================================
# HARD FILTER - EXACTLY ONCE
# ============================================================

def run_hard_filter(
    candidates
):

    header(
        "HARD FILTER"
    )

    qualified = []

    rejected = []

    for candidate in candidates:

        result = hard_filter(
            JOB,
            candidate
        )

        if result["qualified"]:

            qualified.append(
                candidate
            )

        else:

            rejected.append({
                "candidate": candidate,
                "filter": result
            })

    print()

    print(
        f"Total Candidates: "
        f"{len(candidates)}"
    )

    print(
        f"Qualified: "
        f"{len(qualified)}"
    )

    print(
        f"Rejected: "
        f"{len(rejected)}"
    )

    # ========================================================
    # SHOW REJECTED
    # ========================================================

    if rejected:

        print()

        print(
            "REJECTED CANDIDATES"
        )

        print(
            "-" * 70
        )

    for item in rejected:

        candidate = item[
            "candidate"
        ]

        result = item[
            "filter"
        ]

        print()

        print(
            candidate[
                "candidate_id"
            ]
        )

        print(
            f"  Experience: "
            f"{candidate.get('experience_years')}"
        )

        print(
            f"  Job titles: "
            f"{candidate.get('job_titles', [])}"
        )

        print(
            f"  Skill coverage: "
            f"{result['skill_coverage']:.1%}"
        )

        print(
            f"  Title match: "
            f"{result['title_match']}"
        )

        print(
            f"  Reason: "
            f"{result['reason']}"
        )

    return (
        qualified,
        rejected
    )


# ============================================================
# HYBRID RANKING
# ============================================================

def run_ranking(
    qualified
):

    header(
        "HYBRID RANKING"
    )

    if not qualified:

        print(
            "No qualified candidates."
        )

        return []

    start_time = time.time()

    # IMPORTANT:
    #
    # We call hybrid_rank DIRECTLY.
    #
    # We DO NOT call run_retrieval_pipeline()
    # because we already performed Hard Filter above.

    ranked = hybrid_rank(

        JOB,

        qualified,

        top_k=TOP_K
    )

    elapsed = (
        time.time()
        - start_time
    )

    print()

    print(
        f"Qualified candidates: "
        f"{len(qualified)}"
    )

    print(
        f"Returned Top K: "
        f"{len(ranked)}"
    )

    print(
        f"Ranking time: "
        f"{elapsed:.2f}s"
    )

    print()

    print(
        "CURRENT HYBRID RANKING"
    )

    print(
        "-" * 70
    )

    for rank, result in enumerate(
        ranked,
        start=1
    ):

        candidate = result[
            "candidate"
        ]

        print()

        print(
            f"#{rank} "
            f"{candidate['candidate_id']}"
        )

        print(
            f"   Hybrid: "
            f"{result['hybrid_score']:.4f}"
        )

        print(
            f"   BM25: "
            f"{result['bm25_score']:.4f}"
        )

        print(
            f"   Semantic: "
            f"{result['semantic_score']:.4f}"
        )

        print(
            f"   Experience: "
            f"{candidate.get('experience_years')}"
        )

        print(
            f"   Titles: "
            f"{candidate.get('job_titles', [])}"
        )

    return ranked


# ============================================================
# FINAL LLM STAGE
# ============================================================

def run_final_llm(
    ranked
):

    header(
        "FINAL LLM VERIFICATION + EXPLANATION"
    )

    if not ranked:

        print(
            "No ranked candidates to send to LLM."
        )

        return None

    print()

    print(
        f"Sending ONLY the Top "
        f"{len(ranked)} candidates to the LLM."
    )

    print()

    print(
        "No LLM was used during CV parsing."
    )

    start_time = time.time()

    try:

        result = verify_and_explain_ranking(
            JOB,
            ranked
        )

        elapsed = (
            time.time()
            - start_time
        )

        print()

        print(
            f"LLM finished in "
            f"{elapsed:.2f}s"
        )

        print(
            f"Ranking valid: "
            f"{result.get('ranking_valid')}"
        )

        corrected_order = (
            result.get(
                "corrected_order",
                []
            )
        )

        if corrected_order:

            print()

            print(
                "LLM CORRECTED ORDER"
            )

            print(
                "-" * 70
            )

            for rank, candidate_id in enumerate(
                corrected_order,
                start=1
            ):

                print(
                    f"{rank}. "
                    f"{candidate_id}"
                )

        return result

    except Exception as e:

        print()

        print(
            f"LLM stage failed: {e}"
        )

        return {
            "ranking_valid": None,
            "corrected_order": [],
            "candidates": [],
            "error": str(e)
        }


# ============================================================
# PRINT FINAL CANDIDATE EXPLANATIONS
# ============================================================

def print_llm_explanations(
    llm_result
):

    if not llm_result:

        return

    candidates = llm_result.get(
        "candidates",
        []
    )

    if not candidates:

        return

    header(
        "FINAL HR CANDIDATE EXPLANATIONS"
    )

    for index, candidate in enumerate(
        candidates,
        start=1
    ):

        print()

        print(
            "-" * 70
        )

        print(
            f"Candidate: "
            f"{candidate.get('candidate_id')}"
        )

        print()

        print(
            "Matching Skills:"
        )

        for skill in candidate.get(
            "matching_skills",
            []
        ):

            print(
                f"  + {skill}"
            )

        print()

        print(
            "Missing Skills:"
        )

        for skill in candidate.get(
            "missing_skills",
            []
        ):

            print(
                f"  - {skill}"
            )

        print()

        print(
            "Pros:"
        )

        for item in candidate.get(
            "pros",
            []
        ):

            print(
                f"  + {item}"
            )

        print()

        print(
            "Cons:"
        )

        for item in candidate.get(
            "cons",
            []
        ):

            print(
                f"  - {item}"
            )

        print()

        print(
            "Experience Summary:"
        )

        print(
            candidate.get(
                "experience_summary",
                ""
            )
        )

        recommendation = candidate.get(
            "recommendation"
        )

        if recommendation:

            print()

            print(
                "Recommendation:"
            )

            print(
                recommendation
            )


# ============================================================
# SAVE FULL REPORT
# ============================================================

def save_report(
    processing_results,
    qualified,
    rejected,
    ranked,
    llm_result
):

    RESULTS_FOLDER.mkdir(
        parents=True,
        exist_ok=True
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    output_path = (
        RESULTS_FOLDER
        / f"hr_test_{timestamp}.json"
    )

    successful = [
        item
        for item in processing_results
        if item["success"]
    ]

    failed = [
        item
        for item in processing_results
        if not item["success"]
    ]

    report = {

        "timestamp": (
            datetime.now().isoformat()
        ),

        "job": JOB,

        "statistics": {

            "total_cvs": len(
                processing_results
            ),

            "parsed_successfully": len(
                successful
            ),

            "parse_failed": len(
                failed
            ),

            "qualified": len(
                qualified
            ),

            "rejected": len(
                rejected
            ),

            "top_k": len(
                ranked
            ),

            "parser": (
                "local_parser"
            ),

            "llm_used_only_for_final_stage": True
        },

        "processing": (
            processing_results
        ),

        "rejected_candidates": (
            rejected
        ),

        "qualified_candidates": (
            qualified
        ),

        "hybrid_ranking": (
            ranked
        ),

        "llm_result": (
            llm_result
        )
    }

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=2,
            default=str
        )

    return output_path


# ============================================================
# FINAL SUMMARY
# ============================================================

def print_summary(
    processing_results,
    qualified,
    rejected,
    ranked,
    output_path
):

    header(
        "FINAL HR TEST SUMMARY"
    )

    successful = sum(

        1
        for item in processing_results
        if item["success"]
    )

    failed = (
        len(processing_results)
        - successful
    )

    print()

    print(
        f"Job: "
        f"{JOB['title']}"
    )

    print()

    print(
        f"CVs submitted: "
        f"{len(processing_results)}"
    )

    print(
        f"Parsed locally: "
        f"{successful}"
    )

    print(
        f"Parsing failures: "
        f"{failed}"
    )

    print()

    print(
        f"Qualified: "
        f"{len(qualified)}"
    )

    print(
        f"Rejected: "
        f"{len(rejected)}"
    )

    print()

    print(
        f"Top candidates sent to LLM: "
        f"{len(ranked)}"
    )

    print()

    print(
        "FINAL HYBRID TOP 10"
    )

    print(
        "-" * 70
    )

    for rank, result in enumerate(
        ranked,
        start=1
    ):

        print(
            f"{rank:2}. "
            f"{result['candidate_id']:<20} "
            f"{result['hybrid_score']:.4f}"
        )

    print()

    print(
        f"Full report saved to:"
    )

    print(
        output_path
    )


# ============================================================
# MAIN
# ============================================================

def main():

    header(
        "FULL END-TO-END HR RECRUITMENT TEST"
    )

    print()

    print(
        f"Job: {JOB['title']}"
    )

    print(
        f"Parser: LOCAL ONLY"
    )

    print(
        f"Final LLM: TOP {TOP_K} ONLY"
    )

    # ========================================================
    # 1. FIND CVS
    # ========================================================

    files = find_cv_files()

    print()

    print(
        f"CV files found: "
        f"{len(files)}"
    )

    # ========================================================
    # 2. LOCAL PARSING
    # ========================================================

    processing_results = (
        process_all_cvs(
            files
        )
    )

    candidates = [

        item["candidate"]

        for item in processing_results

        if (
            item["success"]
            and item["candidate"]
        )
    ]

    # ========================================================
    # 3. HARD FILTER - ONCE
    # ========================================================

    qualified, rejected = (
        run_hard_filter(
            candidates
        )
    )

    # ========================================================
    # 4. HYBRID RANKING
    # ========================================================

    ranked = run_ranking(
        qualified
    )

    # ========================================================
    # 5. LLM - ONLY HERE
    # ========================================================

    llm_result = run_final_llm(
        ranked
    )

    # ========================================================
    # 6. DISPLAY LLM RESULT
    # ========================================================

    print_llm_explanations(
        llm_result
    )

    # ========================================================
    # 7. SAVE
    # ========================================================

    output_path = save_report(

        processing_results,

        qualified,

        rejected,

        ranked,

        llm_result
    )

    # ========================================================
    # 8. SUMMARY
    # ========================================================

    print_summary(

        processing_results,

        qualified,

        rejected,

        ranked,

        output_path
    )


if __name__ == "__main__":

    main()