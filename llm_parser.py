import os
import json

from dotenv import load_dotenv
from groq import Groq


load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is not set in .env")


client = Groq(api_key=GROQ_API_KEY)


MODEL_NAME = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")


def parse_resume_with_llm(resume_text: str) -> dict:

    prompt = f"""
You are a professional resume parser.

Extract structured information from the resume text below.

IMPORTANT RULES:
- Only extract information that actually exists in the resume.
- Never invent or guess information.
- If a field is missing, use null.
- Return ONLY valid JSON.
- Do not add explanations.
- Keep the original wording where appropriate.

Required JSON structure:

{{
    "name": null,
    "email": null,
    "phone": null,
    "location": null,
    "summary": null,
    "skills": [],
    "experience": [],
    "education": [],
    "certifications": [],
    "languages": []
}}

For experience, use objects with:
- job_title
- company
- start_date
- end_date
- description

For education, use objects with:
- degree
- institution
- start_date
- end_date

Resume text:

{resume_text}
"""





    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": "You extract resume information into structured JSON."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0,
        response_format={
            "type": "json_object"
        }
    )

    content = response.choices[0].message.content

    return json.loads(content)


def parse_resumes_with_llm_batch(cvs: list[dict]) -> list[dict]:

    combined_text = ""

    for index, cv in enumerate(cvs, start=1):

        combined_text += f"""
===== CV {index} =====
Filename: {cv["filename"]}

{cv["text"]}

===== END CV {index} =====

"""

    prompt = f"""
You are a professional resume parser.

You will receive multiple resumes.

Parse EACH resume independently.

IMPORTANT RULES:
- Never mix information between resumes.
- Never invent or guess information.
- Only extract information that actually exists.
- If a field is missing, use null.
- Return ONLY valid JSON.
- Do not add explanations.
- Keep the original wording where appropriate.
- The output MUST contain exactly one result for each CV.
- Keep the SAME ORDER as the input CVs.

Required JSON structure for EACH CV:

{{
    "name": null,
    "email": null,
    "phone": null,
    "location": null,
    "summary": null,
    "skills": [],
    "experience": [],
    "education": [],
    "certifications": [],
    "languages": []
}}

For experience, use objects with:
- job_title
- company
- start_date
- end_date
- description

For education, use objects with:
- degree
- institution
- start_date
- end_date

Return this exact overall structure:

{{
    "results": [
        {{
            "name": null,
            "email": null,
            "phone": null,
            "location": null,
            "summary": null,
            "skills": [],
            "experience": [],
            "education": [],
            "certifications": [],
            "languages": []
        }}
    ]
}}

There must be exactly {len(cvs)} objects inside "results".

Resumes:

{combined_text}
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "You extract multiple resumes into structured JSON. "
                    "Keep every resume completely independent."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0,
        response_format={
            "type": "json_object"
        }
    )

    content = response.choices[0].message.content

    data = json.loads(content)

    results = data.get("results")

    if not isinstance(results, list):
        raise ValueError("LLM batch response does not contain a results array")

    if len(results) != len(cvs):
        raise ValueError(
            f"Expected {len(cvs)} results, but LLM returned {len(results)}"
        )

    return results
