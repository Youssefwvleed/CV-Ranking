import json

from .llm_client import chat_completion


def compact_experience(experience, max_items=5):
    """
    Keep only the important experience information for the LLM.
    Avoid sending full achievements and large nested structures.
    Cap to the most recent jobs to control prompt size.
    """

    if not experience:
        return []

    compact = []

    for item in experience[:max_items]:
        if not isinstance(item, dict):
            continue

        compact.append({
            "title": item.get("title"),
            "employer": item.get("employer"),
            "start": item.get("start"),
            "end": item.get("end"),
        })

    return compact


def truncate_text(text, max_chars=280):
    """Cut long free-text fields (like resume summaries) to a safe length."""

    if not text:
        return ""

    text = str(text).strip()

    if len(text) <= max_chars:
        return text

    return text[:max_chars].rsplit(" ", 1)[0] + "..."


def cap_list(items, max_items=15):
    """Limit list length so one candidate can't blow up the prompt size."""

    if not items:
        return []

    return items[:max_items]


def compact_candidate(result):
    """
    Create a small HR-focused representation of a ranked candidate.
    """

    candidate = result["candidate"]

    return {
        "candidate_id": candidate.get("candidate_id"),

        "current_hybrid_rank": result.get("rank"),

        "hybrid_score": round(
            float(result.get("hybrid_score", 0)),
            4
        ),

        "job_titles": cap_list(candidate.get("job_titles", []), 5),

        "experience_years": candidate.get(
            "experience_years"
        ),

        "skills": cap_list(candidate.get("skills", []), 20),

        "summary": truncate_text(candidate.get("summary", "")),

        "experience": compact_experience(
            candidate.get("experience", [])
        ),

        "education": cap_list(candidate.get("education", []), 5),

        "certifications": cap_list(candidate.get("certifications", []), 8),

        "languages": cap_list(candidate.get("languages", []), 5),
    }

def verify_and_explain_ranking(
    job: dict,
    ranked_candidates: list[dict]
) -> dict:

    candidates = []

    for position, result in enumerate(
        ranked_candidates,
        start=1
    ):
        result_copy = dict(result)

        # Store rank explicitly because some ranking
        # results do not contain it.
        result_copy["rank"] = position

        candidates.append(
            compact_candidate(result_copy)
        )

    candidates_json = json.dumps(
        candidates,
        ensure_ascii=False,
        separators=(",", ":")
    )

    prompt = f"""
You are a recruitment ranking verification assistant.

The candidates below have already been ranked by a Hybrid
retrieval system.

Your job has TWO steps.

STEP 1 — VERIFY THE EXISTING RANKING

Check whether the current ranking is reasonable based only on:

- Job requirements
- Candidate skills
- Candidate experience
- Candidate job titles
- Candidate responsibilities
- Relevant qualifications

Do NOT create a new ranking unless the existing ranking has
a clear and significant ordering mistake.

If the ranking is reasonable:

ranking_valid = true
corrected_order = []

If the ranking needs correction:

ranking_valid = false

corrected_order MUST contain ALL candidate IDs
in the corrected order.

Do NOT provide a reason for ranking changes.

STEP 2 — EXPLAIN EACH CANDIDATE

For every candidate return:

- pros
- cons
- matching_skills
- missing_skills
- experience_summary
- recommendation

IMPORTANT RULES:

- Only use information explicitly present in the candidate data.
- Never invent skills, experience, education, or qualifications.
- Do not infer personal characteristics.
- Do not create a numerical ranking score.
- Do not change the ranking because of writing style.
- Do not use name, age, gender, photo, nationality,
  or other personal characteristics.
- Keep explanations concise.
- Missing information is not proof that the candidate lacks it.
- Return exactly one candidate object for every input candidate.
- Use the candidate_id exactly as provided.
- If a field contains no reliable information, use an empty
  array or empty string.

JOB:

Title:
{job.get("title", "")}

Description:
{job.get("description", "")}

Required Skills:
{job.get("skill_ids", job.get("skills", []))}

Experience Level:
{job.get("experience_level", "")}

Employment Type:
{job.get("employment_type", "")}

CANDIDATES:

{candidates_json}

Return ONLY valid JSON.

Use exactly this structure:

{{
    "ranking_valid": true,
    "corrected_order": [],
    "candidates": [
        {{
            "candidate_id": "CV_001",
            "pros": [],
            "cons": [],
            "matching_skills": [],
            "missing_skills": [],
            "experience_summary": "",
            "recommendation": ""
        }}
    ]
}}

There must be exactly one candidate object
for every input candidate.
"""

    messages = [
        {
            "role": "system",
            "content": (
                "You verify recruitment rankings and provide "
                "evidence-based candidate explanations. "
                "Return only valid JSON."
            )
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    print(
        f"LLM prompt size: "
        f"{len(prompt):,} characters"
    )

    content, provider = chat_completion(
        messages,
        temperature=0
    )

    print(
        f"LLM provider used: {provider}"
    )

    data = json.loads(content)

    return data