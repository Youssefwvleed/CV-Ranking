def build_llm_input(job, candidate, retrieval_result):
    return {
        "job": {
            "title": job["title"],
            "description": job.get("description", ""),
            "skills": job.get("skills", []),
            "experience_years": job.get("experience_years")
        },

        "candidate": {
            "candidate_id": candidate["candidate_id"],
            "job_titles": candidate.get("job_titles", []),
            "skills": candidate.get("skills", []),
            "experience_years": candidate.get(
                "experience_years"
            )
        },

        "retrieval": {
            "bm25_score": retrieval_result["bm25_score"],
            "semantic_score": retrieval_result["semantic_score"],
            "hybrid_score": retrieval_result["hybrid_score"]
        }
    }