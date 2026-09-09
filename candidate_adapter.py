import re
from datetime import datetime


MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,

    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "sept": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}


PRESENT_WORDS = {
    "present",
    "current",
    "now",
    "ongoing",
    "today",
}


def parse_date_to_month(date_value):
    """
    Convert different date formats into a month index.

    Examples:
        2020              -> 2020 * 12
        05/2020           -> 2020 * 12 + 5
        May 2020          -> 2020 * 12 + 5
        Present/Current   -> current month
    """

    if date_value is None:
        return None

    value = str(date_value).strip().lower()

    if not value:
        return None

    # Current / Present
    if value in PRESENT_WORDS:
        now = datetime.now()
        return now.year * 12 + now.month

    # MM/YYYY or MM-YYYY
    match = re.search(r"\b(0?[1-9]|1[0-2])\s*[/\-]\s*(\d{4})\b", value)

    if match:
        month = int(match.group(1))
        year = int(match.group(2))
        return year * 12 + month

    # Month YYYY
    match = re.search(
        r"\b("
        r"january|february|march|april|may|june|july|august|"
        r"september|october|november|december|"
        r"jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec"
        r")\s+(\d{4})\b",
        value,
    )

    if match:
        month_name = match.group(1)
        year = int(match.group(2))
        month = MONTHS[month_name]

        return year * 12 + month

    # YYYY
    match = re.search(r"\b(19\d{2}|20\d{2})\b", value)

    if match:
        year = int(match.group(1))
        return year * 12

    return None


def calculate_experience_years(experience):
    """
    Calculate total non-overlapping experience in years.

    Overlapping jobs are merged so the same period
    is not counted twice.
    """

    if not experience:
        return None

    intervals = []

    for item in experience:

        if not isinstance(item, dict):
            continue

        start = parse_date_to_month(item.get("start"))
        end = parse_date_to_month(item.get("end"))

        if start is None:
            continue

        if end is None:
            continue

        if end < start:
            continue

        intervals.append((start, end))

    if not intervals:
        return None

    intervals.sort()

    merged = []

    current_start, current_end = intervals[0]

    for start, end in intervals[1:]:

        if start <= current_end + 1:
            current_end = max(current_end, end)

        else:
            merged.append((current_start, current_end))
            current_start = start
            current_end = end

    merged.append((current_start, current_end))

    total_months = sum(
        end - start
        for start, end in merged
    )

    return round(total_months / 12, 1)


def extract_job_titles(experience):
    """
    Extract job titles from parsed experience entries.
    """

    titles = []

    if not experience:
        return titles

    for item in experience:

        if not isinstance(item, dict):
            continue

        title = item.get("title")

        if not title:
            continue

        title = str(title).strip()

        if not title:
            continue

        if title not in titles:
            titles.append(title)

    return titles


def normalize_education(education):
    """
    Keep education in a safe list/dict structure.
    """

    if education is None:
        return []

    if isinstance(education, list):
        return education

    return [education]


def adapt_candidate(parsed_resume: dict, candidate_id: str) -> dict:
    """
    Convert resume_parser.py output into the structure
    expected by the ranking pipeline.
    """

    if not isinstance(parsed_resume, dict):
        parsed_resume = {}

    experience = parsed_resume.get("experience") or []

    job_titles = extract_job_titles(experience)

    experience_years = calculate_experience_years(experience)

    candidate = {
        "candidate_id": str(candidate_id),

        "experience_years": experience_years,

        "job_titles": job_titles,

        "skills": parsed_resume.get("skills") or [],

        "education": normalize_education(
            parsed_resume.get("education")
        ),

        "languages": parsed_resume.get("languages") or [],

        "certifications": parsed_resume.get("certifications") or [],

        # Keep original parser information
        # for the final LLM explanation.
        "name": parsed_resume.get("name"),

        "email": parsed_resume.get("email"),

        "phone": parsed_resume.get("phone"),

        "summary": parsed_resume.get("summary"),

        "experience": experience,
    }

    return candidate