def normalize(text):

    if text is None:
        return ""

    return str(text).strip().lower()


def calculate_skill_coverage(
    job_skills,
    candidate_skills
):

    job_skills = {
        normalize(skill)
        for skill in (job_skills or [])
        if normalize(skill)
    }

    candidate_skills = {
        normalize(skill)
        for skill in (candidate_skills or [])
        if normalize(skill)
    }

    if not job_skills:
        return 0.0

    matched = job_skills.intersection(
        candidate_skills
    )

    return (
        len(matched)
        / len(job_skills)
    )


def calculate_title_match(
    job_titles,
    candidate_titles
):
    """
    Match both exact titles and contained titles.

    Examples:

    Sales Executive
    Senior Sales Executive

    should be considered related.
    """

    job_titles = [
        normalize(title)
        for title in (job_titles or [])
        if normalize(title)
    ]

    candidate_titles = [
        normalize(title)
        for title in (candidate_titles or [])
        if normalize(title)
    ]

    for job_title in job_titles:

        for candidate_title in candidate_titles:

            # Exact
            if job_title == candidate_title:
                return True

            # Related variation
            if (
                job_title in candidate_title
                or candidate_title in job_title
            ):
                return True

    return False


def hard_filter(
    job,
    candidate
):

    # ========================================================
    # SKILLS
    # ========================================================

    skill_coverage = calculate_skill_coverage(

        job.get(
            "skills",
            []
        ),

        candidate.get(
            "skills",
            []
        )
    )

    # ========================================================
    # JOB TITLE
    # ========================================================

    title_match = calculate_title_match(

        job.get(
            "related_titles",
            []
        ),

        candidate.get(
            "job_titles",
            []
        )
    )

    # ========================================================
    # EXPERIENCE
    # ========================================================

    candidate_experience = candidate.get(
        "experience_years"
    )

    required_experience = job.get(
        "experience_years",
        0
    )

    if candidate_experience is None:

        # Unknown is NOT automatically treated as zero.
        experience_ok = None

    else:

        experience_ok = (
            candidate_experience
            >= required_experience
        )

    # ========================================================
    # DECISION
    # ========================================================

    # Candidate has almost no skill relevance
    # AND no relevant job title.

    if (
        skill_coverage < 0.30
        and not title_match
    ):

        return {
            "candidate_id": candidate.get(
                "candidate_id"
            ),

            "qualified": False,

            "skill_coverage": skill_coverage,

            "title_match": title_match,

            "experience_ok": experience_ok,

            "reason": (
                "Very low skill relevance "
                "and no related job title"
            )
        }

    # Important:
    #
    # Missing experience data alone should NOT reject
    # a candidate.
    #
    # Only explicitly insufficient experience can
    # contribute to rejection.

    if (
        skill_coverage < 0.30
        and experience_ok is False
    ):

        return {
            "candidate_id": candidate.get(
                "candidate_id"
            ),

            "qualified": False,

            "skill_coverage": skill_coverage,

            "title_match": title_match,

            "experience_ok": experience_ok,

            "reason": (
                "Very low skill relevance "
                "and insufficient experience"
            )
        }

    return {
        "candidate_id": candidate.get(
            "candidate_id"
        ),

        "qualified": True,

        "skill_coverage": skill_coverage,

        "title_match": title_match,

        "experience_ok": experience_ok,

        "reason": "Passed hard filter"
    }