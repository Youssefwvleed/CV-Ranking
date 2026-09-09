def build_job_text(job):
    """
    Build a searchable text representation of the job.
    """

    skills = " ".join(job.get("skills", []))
    related_titles = " ".join(job.get("related_titles", []))

    return " ".join([
        job.get("title", ""),
        related_titles,
        skills,
        job.get("description", "")
    ]).strip()


def build_candidate_text(candidate):
    """
    Build a searchable text representation of a candidate.
    """

    job_titles = " ".join(candidate.get("job_titles", []))
    skills = " ".join(candidate.get("skills", []))

    return " ".join([
        job_titles,
        skills
    ]).strip()