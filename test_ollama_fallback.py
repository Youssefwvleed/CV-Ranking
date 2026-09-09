from ollama_parser import parse_resume_with_ollama
from ranking.mock_data.mock_candidates import mock_candidates


print("=" * 70)
print("OLLAMA LOCAL PARSER TEST")
print("=" * 70)

cv = mock_candidates[0]

print(f"\nTesting: {cv['candidate_id']}")

# Find the resume text field
resume_text = cv.get("text")

if resume_text is None:
    resume_text = cv.get("resume_text")

if resume_text is None:
    resume_text = cv.get("raw_text")

if resume_text is None:
    print("\nAvailable fields:")
    print(list(cv.keys()))

    raise KeyError(
        "Could not find resume text in mock candidate. "
        "Check the field name printed above."
    )

try:

    result = parse_resume_with_ollama(resume_text)

    print("\nOLLAMA RESULT")
    print("-" * 70)

    print("Name:", result.get("name"))
    print("Skills:", result.get("skills"))
    print("Experience:", result.get("experience"))
    print("Education:", result.get("education"))
    print("Certifications:", result.get("certifications"))
    print("Languages:", result.get("languages"))

    print("\n" + "=" * 70)
    print("OLLAMA TEST: SUCCESS")
    print("=" * 70)

except Exception as e:

    print("\n" + "=" * 70)
    print("OLLAMA TEST: FAILED")
    print("=" * 70)

    print(type(e).__name__)
    print(e)