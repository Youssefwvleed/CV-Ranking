from .schemas import CandidateProfile, Education


def build_candidate_profile(candidate_data: dict) -> CandidateProfile:
    education = [
        Education(
            degree=edu["degree"],
            field=edu["field"]
        )
        for edu in candidate_data.get("education", [])
    ]

    return CandidateProfile(
        candidate_id=str(candidate_data["candidate_id"]),
        experience_years=candidate_data.get("experience_years"),
        job_titles=candidate_data.get("job_titles", []),
        education=education,
        languages=candidate_data.get("languages", []),
        skills=candidate_data.get("skills", []),
        certifications=candidate_data.get("certifications", [])
    )