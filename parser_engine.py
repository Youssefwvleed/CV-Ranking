"""
Local deterministic resume parsing entry point.

IMPORTANT:
- No Groq is used here.
- No Ollama is used here.
- All CV parsing is handled by resume_parser.py.
- LLMs are reserved only for the final ranking explanation stage.
"""

from resume_parser import parse_resume


def parse_resume_with_fallback(
    resume_text: str
) -> tuple[dict, str]:
    """
    Parse one CV using the deterministic local resume parser.
    """

    result = parse_resume(resume_text)

    return result, "local_parser"


def parse_resumes_with_fallback_batch(
    cvs: list[dict]
) -> tuple[list[dict], str]:
    """
    Parse multiple CVs locally.

    Expected input:
    [
        {
            "text": "...",
            ...
        }
    ]
    """

    results = []

    for cv in cvs:

        text = cv.get("text", "")

        parsed = parse_resume(text)

        results.append(parsed)

    return results, "local_parser"