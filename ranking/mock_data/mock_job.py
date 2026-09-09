from ranking.schemas import Job, JobOrganization


mock_job = Job(
    id=1,
    organization_id=10,

    organization=JobOrganization(
        id=10,
        name="Tech Company",
        slug="tech-company",
        type="company",
        active=True,
        verified=True,
        trusted=True
    ),

    organization_verified=True,
    organization_trusted=True,

    job_title_id=5,
    title="Junior Data Scientist",
    slug="junior-data-scientist",

    description="""
    We are looking for a Junior Data Scientist
    with experience in Python, SQL and data analysis.
    """,

    employment_type="full_time",
    workplace_type="onsite",
    experience_level="junior",

    country_id=1,
    city="Cairo",

    vacancy_count=2,

    salary_min=None,
    salary_max=None,

    currency_id=None,
    salary_option_id=None,

    status="published",

    release_at=None,
    application_deadline=None,
    published_at=None,

    skill_ids=[101, 102, 103],

    created_at=None,
    updated_at=None
)