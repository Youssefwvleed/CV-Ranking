from pydantic import BaseModel, Field
from typing import Optional, Literal
from datetime import datetime


# =========================
# Job API Models
# =========================

class JobOrganization(BaseModel):
    id: int
    name: str
    slug: str
    type: str
    active: bool
    verified: bool
    trusted: bool


class Job(BaseModel):
    id: int
    organization_id: int
    organization: JobOrganization

    organization_verified: bool
    organization_trusted: bool

    job_title_id: int
    title: str
    slug: str
    description: str

    employment_type: Literal[
        "full_time",
        "part_time",
        "contract",
        "temporary",
        "internship",
        "freelance"
    ]

    workplace_type: str
    experience_level: str

    country_id: int
    city: Optional[str] = None

    vacancy_count: int

    salary_min: Optional[str] = None
    salary_max: Optional[str] = None

    currency_id: Optional[int] = None
    salary_option_id: Optional[int] = None

    status: str

    release_at: Optional[datetime] = None
    application_deadline: Optional[datetime] = None
    published_at: Optional[datetime] = None

    skill_ids: list[int] = Field(min_length=1, max_length=15)

    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


# =========================
# Ranking Criteria
# =========================

class Criterion(BaseModel):
    id: str

    type: Literal[
        "experience",
        "skill",
        "education",
        "language",
        "certification"
    ]

    required: bool = False

    # Experience
    min_years: Optional[float] = None

    # Skill
    skill_id: Optional[int] = None
    name: Optional[str] = None
    accepted_terms: list[str] = []


# =========================
# Internal Ranking Job
# =========================

class JobSpec(BaseModel):
    job_id: int
    title: str
    description: str

    experience_level: str
    criteria: list[Criterion]


# =========================
# Candidate
# =========================

class Education(BaseModel):
    degree: str
    field: str


class CandidateProfile(BaseModel):
    candidate_id: str

    experience_years: Optional[float] = None

    job_titles: list[str] = []

    education: list[Education] = []

    languages: list[str] = []

    skills: list[str] = []

    certifications: list[str] = []
    
class LLMExplanation(BaseModel):
    candidate_id: str

    pros: list[str]
    cons: list[str]

    matching_skills: list[str]
    missing_skills: list[str]

    experience_summary: str
    recommendation: str    
    
class LLMExplanation(BaseModel):
    candidate_id: str

    matching_skills: list[str]
    missing_skills: list[str]

    pros: list[str]
    cons: list[str]

    experience_summary: str
    recommendation: str    
class CandidateExplanation(BaseModel):
    candidate_id: str

    pros: list[str]
    cons: list[str]

    matching_skills: list[str]
    missing_skills: list[str]

    experience_summary: str


class LLMRankingVerification(BaseModel):
    ranking_valid: bool
    corrected_order: list[str]

    candidates: list[CandidateExplanation]    