import json
import os

from ollama_parser import client, OLLAMA_MODEL

OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "8192"))

def verify_and_explain_ranking(
    job: dict,
    ranked_candidates: list[dict]
) -> dict:

    candidates_text = ""

    for position, result in enumerate(
        ranked_candidates,
        start=1
    ):

        candidate = result["candidate"]

        candidates_text += f"""
===== CANDIDATE {position} =====

Candidate ID:
{candidate["candidate_id"]}

Current Hybrid Rank:
{position}

Hybrid Score:
{result["hybrid_score"]}

Skills:
{candidate.get("skills", [])}

Summary:
{candidate.get("summary", "")}

Experience:
{candidate.get("experience", [])}

Education:
{candidate.get("education", [])}

Certifications:
{candidate.get("certifications", [])}

Languages:
{candidate.get("languages", [])}

===== END CANDIDATE =====
"""

    prompt = f"""
You are a recruitment ranking verification assistant.

The candidates below have already been ranked by a Hybrid
retrieval system.

Your job has TWO steps:

STEP 1 — VERIFY THE EXISTING RANKING

Check whether the current ranking is reasonable based only on:

- Job requirements
- Candidate skills
- Candidate experience
- Candidate responsibilities
- Relevant qualifications

Do NOT create a new ranking unless the existing ranking has
a clear and significant ordering mistake.

If the ranking is reasonable:
- ranking_valid = true
- corrected_order = []

If the ranking needs correction:
- ranking_valid = false
- corrected_order must contain ALL candidate IDs
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
- Do not use name, age, gender, photo, nationality, or other
  personal characteristics.
- Keep explanations concise.
- Missing information is not proof that the candidate lacks it.

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

Candidates:

{candidates_text}

Return ONLY valid JSON using exactly this structure:

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

There must be exactly one candidate object for every input candidate.
"""

    response = client.chat(
        model=OLLAMA_MODEL,
        messages=[
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
        ],
              format="json",
        options={
            "temperature": 0,
            "num_ctx": OLLAMA_NUM_CTX
        }
    )

    content = response["message"]["content"]

    data = json.loads(content)

    return data