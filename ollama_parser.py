import os
import json

from dotenv import load_dotenv
import ollama


load_dotenv()

# Ollama runs locally, so no API key is needed — only the host and the
# model name (must be a model you already pulled with `ollama pull <name>`).
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:latest")

client = ollama.Client(host=OLLAMA_HOST)


def parse_resume_with_ollama(resume_text: str) -> dict:
    """Same contract as llm_parser.parse_resume_with_llm, but runs fully
    locally through Ollama instead of calling the Groq API."""

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

    response = client.chat(
        model=OLLAMA_MODEL,
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
        format="json",
        options={"temperature": 0}
    )

    content = response["message"]["content"]

    return json.loads(content)


def parse_resumes_with_ollama_batch(cvs: list[dict]) -> list[dict]:
    """Same contract as llm_parser.parse_resumes_with_llm_batch. Kept for
    interface parity, but a local model handles fewer CVs per request
    reliably than Groq — the caller may want to send smaller batches
    when this fallback is the one being used."""

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

    response = client.chat(
        model=OLLAMA_MODEL,
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
        format="json",
        options={"temperature": 0}
    )

    content = response["message"]["content"]

    data = json.loads(content)

    results = data.get("results")

    if not isinstance(results, list):
        raise ValueError("Ollama batch response does not contain a results array")

    if len(results) != len(cvs):
        raise ValueError(
            f"Expected {len(cvs)} results, but Ollama returned {len(results)}"
        )

    return results
