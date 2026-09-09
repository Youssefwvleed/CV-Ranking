from .schemas import Job, JobSpec, Criterion


def build_job_spec(job: Job) -> JobSpec:
    criteria = []

    # -------------------------
    # Experience
    # -------------------------
    #
    # We currently don't have
    # minimum years directly from
    # the API schema.
    #
    # So we don't create an
    # experience hard gate yet.

    # -------------------------
    # Skills
    # -------------------------

    for skill_id in job.skill_ids:
        criteria.append(
            Criterion(
                id=f"skill_{skill_id}",
                type="skill",
                required=True,
                skill_id=skill_id
            )
        )

    return JobSpec(
        job_id=job.id,
        title=job.title,
        description=job.description,
        experience_level=job.experience_level,
        criteria=criteria
    )