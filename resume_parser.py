"""High-coverage offline resume/CV text parser.

The parser makes no API calls and uses only Python's standard library. Input may be
plain UTF-8 text or JSON containing a top-level ``text`` string.

It is a deterministic rule engine, not a language model. The rules cover common
resume headings, contact formats, date styles, one/two-line employment layouts,
education ordering, skill groups, certifications, and language proficiency formats.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


SCHEMA_KEYS = (
    "name",
    "email",
    "phone",
    "summary",
    "skills",
    "experience",
    "education",
    "certifications",
    "languages",
)

EMAIL_RE = re.compile(r"(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)+(?![\w.-])", re.I)
PHONE_RE = re.compile(r"(?<!\w)\+?\(?\d(?:[ \t()./-]*\d){6,14}(?!\w)")

MONTH_NAME = (
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|"
    r"Nov(?:ember)?|Dec(?:ember)?)\.?"
)
DATE_TOKEN = rf"(?:{MONTH_NAME}\s+\d{{4}}|(?:0?[1-9]|1[0-2])/\d{{4}}|\d{{4}})"
END_DATE_TOKEN = rf"(?:{DATE_TOKEN}|Present|Current|Now|Ongoing|Today)"
DATE_RANGE_RE = re.compile(
    rf"(?P<start>{DATE_TOKEN})\s*(?:[-\u2013\u2014]|\bto\b|\bthrough\b)\s*"
    rf"(?P<end>{END_DATE_TOKEN})",
    re.I,
)
SINCE_RE = re.compile(rf"\bSince\s+(?P<start>{DATE_TOKEN})\b", re.I)
EXPECTED_RE = re.compile(
    rf"\b(?:Expected|Graduating|Graduation(?:\s+date)?\s*:?)\s*(?P<end>{DATE_TOKEN})\b",
    re.I,
)
TRAILING_DATE_RE = re.compile(
    rf"(?:[,|]\s*|\s+)(?:Issued\s+)?(?P<date>{DATE_TOKEN})\s*$", re.I
)

BULLET_RE = re.compile(r"^\s*(?:[\u2022\u25cf\u25aa\u25e6\u2023]|[-*])\s+")
ROLE_RE = re.compile(
    r"\b(?:intern(?:ship)?|trainee|volunteer|engineer|developer|programmer|"
    r"analyst|scientist|manager|specialist|consultant|assistant|associate|"
    r"coordinator|designer|member|lead|director|officer|administrator|"
    r"researcher|teacher|professor|accountant|photographer|writer|editor|"
    r"evaluator|architect|technician|supervisor|representative|recruiter|"
    r"founder|owner|president|head|executive|tester|qa|sales|marketing|"
    r"product|operations|customer service|data entry)\b",
    re.I,
)
ORG_RE = re.compile(
    r"\b(?:inc\.?|llc|ltd\.?|limited|corp(?:oration)?\.?|company|co\.?|"
    r"group|club|foundation|organization|association|agency|bank|hospital|"
    r"university|college|academy|school|institute|studio|laborator(?:y|ies)|"
    r"labs?|systems?|solutions?|technolog(?:y|ies)|services?|consulting|"
    r"enterprises?|industries|department|ministry|authority|committee|team)\b",
    re.I,
)
DEGREE_RE = re.compile(
    r"\b(?:b\.?\s*sc\.?|m\.?\s*sc\.?|b\.?\s*a\.?|m\.?\s*a\.?|"
    r"b\.?\s*eng\.?|m\.?\s*eng\.?|b\.?\s*tech\.?|m\.?\s*tech\.?|"
    r"bba|mba|llb|jd|md|bachelor(?:'s)?|master(?:'s)?|doctorate|"
    r"ph\.?\s*d\.?|diploma|associate(?:'s)?|high school|secondary school|"
    r"puc|sslc|hsc|ssc|gcse|a[ -]?levels?|matriculation|"
    r"computer science|engineering degree)\b",
    re.I,
)
INSTITUTION_RE = re.compile(
    r"\b(?:university|college|academy|school|institute|polytechnic|faculty|"
    r"conservatory|seminary)\b",
    re.I,
)
IGNORE_EDUCATION_RE = re.compile(
    r"^(?:(?:relevant )?coursework|courses|gpa|grade|percentage|marks?|score|"
    r"honors?|activities)\s*(?::|-)",
    re.I,
)

LANGUAGE_NAMES = {
    "afrikaans", "albanian", "amharic", "arabic", "armenian", "azerbaijani",
    "basque", "bengali", "bosnian", "bulgarian", "burmese", "cantonese",
    "catalan", "chinese", "croatian", "czech", "danish", "dari", "dutch",
    "english", "estonian", "farsi", "filipino", "finnish", "french", "georgian",
    "german", "greek", "gujarati", "hebrew", "hindi", "hungarian", "icelandic",
    "indonesian", "irish", "italian", "japanese", "kannada", "kazakh", "khmer",
    "korean", "kurdish", "lao", "latvian", "lithuanian", "malay", "malayalam",
    "mandarin", "marathi", "nepali", "norwegian", "pashto", "persian", "polish",
    "portuguese", "punjabi", "romanian", "russian", "serbian", "sinhala",
    "slovak", "slovenian", "somali", "spanish", "swahili", "swedish", "tamil",
    "telugu", "thai", "turkish", "ukrainian", "urdu", "uzbek", "vietnamese",
    "welsh", "yoruba",
}
HOBBY_WORDS = {
    "art", "blogging", "chess", "cooking", "cycling", "dance", "dancing",
    "drawing", "fitness", "football", "gaming", "gardening", "hiking", "music",
    "painting", "photography", "reading", "running", "singing", "sports", "swimming",
    "travel", "traveling", "travelling", "volleyball", "writing", "yoga",
}
ACTION_START_RE = re.compile(
    r"^(?:achieved|analyzed|answered|assisted|built|collaborated|coordinated|created|"
    r"delivered|demonstrated|designed|developed|directed|established|evaluated|"
    r"handled|implemented|improved|increased|led|maintained|managed|monitored|"
    r"optimized|organized|performed|prepared|processed|provided|reduced|resolved|"
    r"supported|supervised|trained|used|worked)\b",
    re.I,
)


HEADING_ALIASES = {
    # Summary
    "summary": "summary",
    "professional summary": "summary",
    "executive summary": "summary",
    "profile": "summary",
    "professional profile": "summary",
    "personal profile": "summary",
    "career objective": "summary",
    "objective": "summary",
    "about me": "summary",
    "overview": "summary",
    # Contact/sidebar
    "contact": "contact",
    "contact details": "contact",
    "contact information": "contact",
    "personal details": "contact",
    # Skills
    "skills": "skills",
    "technical skills": "skills",
    "professional skills": "skills",
    "key skills": "skills",
    "core skills": "skills",
    "core competencies": "skills",
    "competencies": "skills",
    "areas of expertise": "skills",
    "expertise": "skills",
    "technologies": "skills",
    "tools and technologies": "skills",
    "tech stack": "skills",
    # Experience-like sections
    "experience": "experience",
    "work experience": "experience",
    "professional experience": "experience",
    "relevant experience": "experience",
    "employment": "experience",
    "employment history": "experience",
    "work history": "experience",
    "career history": "experience",
    "internship experience": "experience",
    "internships": "experience",
    "volunteer experience": "experience",
    "volunteering": "experience",
    "leadership experience": "experience",
    "extracurricular activities": "experience",
    "activities": "experience",
    # Education
    "education": "education",
    "education and training": "education",
    "academic background": "education",
    "academic experience": "education",
    "academic qualifications": "education",
    "qualifications": "education",
    # Certifications
    "certifications": "certifications",
    "certification": "certifications",
    "certificates": "certifications",
    "licenses": "certifications",
    "licenses and certifications": "certifications",
    "certifications and licenses": "certifications",
    "professional certifications": "certifications",
    # Languages
    "languages": "languages",
    "language": "languages",
    "language skills": "languages",
    "language proficiency": "languages",
    # Boundaries not represented in the requested schema
    "projects": "ignored",
    "selected projects": "ignored",
    "personal projects": "ignored",
    "academic projects": "ignored",
    "portfolio": "ignored",
    "awards": "ignored",
    "awards and honors": "ignored",
    "honors": "ignored",
    "publications": "ignored",
    "interests": "ignored",
    "hobbies": "hobbies",
    "hobbies and interests": "hobbies",
    "personal interests": "hobbies",
    "references": "ignored",
    "courses": "ignored",
    "training": "ignored",
    "conferences": "ignored",
    "personal information": "ignored",
}

SECTION_LANES = {
    "summary": "main",
    "experience": "main",
    "education": "main",
    "contact": "sidebar",
    "skills": "sidebar",
    "languages": "sidebar",
    "hobbies": "sidebar",
}


def clean_space(value: str) -> str:
    return " ".join(value.replace("\x00", "").replace("\u00a0", " ").split())


def strip_bullet(value: str) -> str:
    return BULLET_RE.sub("", value).strip()


def unique_strings(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = clean_space(value).strip(" ,;|\u00b7\u2022")
        key = cleaned.casefold()
        if cleaned and key not in seen:
            seen.add(key)
            result.append(cleaned)
    return result


def normalized_heading(line: str) -> str:
    value = strip_bullet(line).strip().strip("#=_~").strip().rstrip(":")
    value = re.sub(r"[^\w& ]+", " ", value, flags=re.UNICODE)
    return clean_space(value).casefold()


def detect_heading(line: str) -> tuple[str | None, str]:
    """Return a canonical section and any content after an inline heading."""
    stripped = line.strip().strip("#=_~").strip()
    exact = HEADING_ALIASES.get(normalized_heading(stripped))
    if exact:
        return exact, ""

    # Supports forms such as "SKILLS: Python, SQL" and "PROFILE: ...".
    if ":" in stripped:
        prefix, remainder = stripped.split(":", 1)
        inline = HEADING_ALIASES.get(normalized_heading(prefix))
        if inline:
            return inline, remainder.strip()
    return None, ""


def is_interleaved_layout(text: str) -> bool:
    """Detect the stacked headings produced by many two-column PDF extractors."""
    previous: str | None = None
    cross_lane_pairs = 0
    for raw_line in text.splitlines():
        if not raw_line.strip():
            continue
        section, remainder = detect_heading(raw_line)
        if section and not remainder:
            if (
                previous
                and SECTION_LANES.get(previous)
                and SECTION_LANES.get(section)
                and SECTION_LANES[previous] != SECTION_LANES[section]
            ):
                cross_lane_pairs += 1
            previous = section
        else:
            previous = None
    return cross_lane_pairs >= 2


def is_contact_line(value: str) -> bool:
    if EMAIL_RE.search(value) or PHONE_RE.search(value):
        return True
    if re.search(r"(?:https?://|www\.|linkedin\.com|github\.com)", value, re.I):
        return True
    if re.match(r"^(?:e-?mail|phone|mobile|cell|tel|address|location|linkedin)\s*:", value, re.I):
        return True
    words = value.split()
    return len(words) <= 7 and value.count(",") >= 2 and not value.endswith(".")


def is_language_line(value: str) -> bool:
    cleaned = clean_space(value).strip(" ,;:()[]-").casefold()
    if not cleaned:
        return False
    first = re.split(r"\s*(?::|[-\u2013\u2014(])\s*", cleaned, maxsplit=1)[0]
    if first in LANGUAGE_NAMES:
        return True
    match = re.match(
        r"^(?:native|fluent|advanced|intermediate|basic|beginner)\s+(?:in\s+)?(.+)$",
        cleaned,
    )
    return bool(match and match.group(1).strip() in LANGUAGE_NAMES)


def is_hobby_line(value: str) -> bool:
    cleaned = clean_space(value).strip(" ,;:()[]-").casefold()
    return cleaned in HOBBY_WORDS


def is_education_line(value: str) -> bool:
    return bool(
        DEGREE_RE.search(value)
        or INSTITUTION_RE.search(value)
        or DATE_RANGE_RE.search(value)
        or SINCE_RE.search(value)
        or EXPECTED_RE.search(value)
        or IGNORE_EDUCATION_RE.search(value)
    )


def has_unclosed_group(value: str) -> bool:
    return value.count("(") > value.count(")") or value.count("[") > value.count("]")


def section_line_score(
    section: str,
    value: str,
    *,
    state: dict[str, Any],
    candidates: list[str],
) -> int:
    """Score a line for one active column using only observable text features."""
    words = value.split()
    word_count = len(words)
    ends_sentence = value.endswith((".", "!", "?", ";"))
    has_date = bool(DATE_RANGE_RE.search(value) or SINCE_RE.search(value) or EXPECTED_RE.search(value))
    previous = state["last_non_date"].get(section, "")
    score = 0

    if section == "contact":
        score += 150 if is_contact_line(value) else -30
    elif section == "summary":
        score += 45 if word_count >= 7 and not is_contact_line(value) else -20
        if ACTION_START_RE.search(value):
            score -= 15
    elif section == "languages":
        score += 180 if is_language_line(value) else -35
    elif section == "hobbies":
        if is_hobby_line(value):
            score += 160
        elif word_count <= 3 and not is_education_line(value):
            score += 15
        else:
            score -= 30
    elif section == "education":
        score += 140 if is_education_line(value) else -15
    elif section == "experience":
        if has_date:
            score += 130
        if looks_like_role(value):
            score += 85
        if looks_like_organization(value):
            score += 75
        if ACTION_START_RE.search(value):
            score += 80
        if word_count >= 7:
            score += 45
        if ends_sentence:
            score += 20
    elif section == "skills":
        if not has_date and not is_contact_line(value) and not is_language_line(value):
            if word_count <= 7 and not ends_sentence:
                score += 45
            if re.search(r"\b(?:skill|management|communication|problem|focus|attitude|"
                         r"leadership|teamwork|organization|planning|development)\b", value, re.I):
                score += 25
        if looks_like_organization(value) or ACTION_START_RE.search(value):
            score -= 35

    # A closing fragment belongs to the column with an unmatched opening group,
    # even when a date from that column appeared between the two fragments.
    if state["open_groups"].get(section, 0) > 0 and re.search(r"[)\]]", value):
        score += 220

    # Strong grammatical continuations should stay with their previous column.
    if (
        previous
        and value[:1].islower()
        and not previous.endswith((".", "!", "?", ";", ":"))
    ):
        if re.search(
            r"(?:\b(?:and|or|to|of|for|with|within|on|at|in|from|according to|wherever)|"
            r"[,/-])\s*$",
            previous,
            re.I,
        ):
            score += 90
        elif value[:1].islower() and word_count <= 5:
            score += 20

    # Interleaved extractors commonly alternate columns for short wrapped fragments.
    if (
        len(candidates) == 2
        and state.get("last_section") != section
        and word_count <= 4
        and value[:1].islower()
    ):
        score += 35
    return score


def split_interleaved_sections(text: str) -> dict[str, list[str]]:
    """Recover logical columns from an interleaved two-column text stream."""
    sections: dict[str, list[str]] = defaultdict(list)
    lanes: dict[str, str | None] = {"main": None, "sidebar": None}
    state: dict[str, Any] = {
        "last_section": None,
        "last_non_date": {},
        "open_groups": defaultdict(int),
    }

    for raw_line in text.splitlines():
        value = strip_bullet(raw_line)
        if not value:
            continue
        section, remainder = detect_heading(raw_line)
        if section:
            lane = SECTION_LANES.get(section)
            if lane:
                lanes[lane] = section
            if remainder:
                value = remainder
            else:
                continue

        candidates = [candidate for candidate in lanes.values() if candidate]
        candidates = list(dict.fromkeys(candidates))
        if not candidates:
            chosen = "preamble"
        elif len(candidates) == 1:
            chosen = candidates[0]
        else:
            ranked = [
                (
                    section_line_score(candidate, value, state=state, candidates=candidates),
                    candidate == state.get("last_section"),
                    candidate,
                )
                for candidate in candidates
            ]
            chosen = max(ranked)[2]

        sections[chosen].append(raw_line.rstrip())
        if not (DATE_RANGE_RE.search(value) or SINCE_RE.search(value) or EXPECTED_RE.search(value)):
            state["last_non_date"][chosen] = clean_space(value)
        state["open_groups"][chosen] += value.count("(") - value.count(")")
        state["open_groups"][chosen] += value.count("[") - value.count("]")
        state["open_groups"][chosen] = max(0, state["open_groups"][chosen])
        state["last_section"] = chosen

    if "skills" in sections:
        sections["skills"] = repair_interleaved_skill_lines(sections["skills"])
    if "experience" in sections:
        sections["experience"] = repair_interleaved_experience_lines(sections["experience"])
    return dict(sections)


def repair_interleaved_skill_lines(lines: list[str]) -> list[str]:
    """Join lowercase skill fragments after the other column has been removed."""
    repaired: list[str] = []
    for raw_line in lines:
        value = strip_bullet(raw_line)
        if (
            repaired
            and value[:1].islower()
            and len(value.split()) <= 4
            and not strip_bullet(repaired[-1]).endswith((".", "!", "?", ";", ":"))
        ):
            repaired[-1] = clean_space(strip_bullet(repaired[-1]) + " " + value)
        else:
            repaired.append(raw_line)
    return repaired


def repair_interleaved_experience_lines(lines: list[str]) -> list[str]:
    """Repair title brackets and corporate suffixes split around sidebar text."""
    repaired = list(lines)
    removed: set[int] = set()

    for index, raw_line in enumerate(repaired):
        if index in removed or not has_unclosed_group(strip_bullet(raw_line)):
            continue
        for candidate_index in range(index + 1, min(len(repaired), index + 4)):
            candidate = strip_bullet(repaired[candidate_index])
            if DATE_RANGE_RE.search(candidate) or SINCE_RE.search(candidate):
                continue
            if re.search(r"[)\]]", candidate):
                repaired[index] = clean_space(strip_bullet(raw_line) + " " + candidate)
                removed.add(candidate_index)
            break

    compact = [line for index, line in enumerate(repaired) if index not in removed]
    result: list[str] = []
    suffix_re = re.compile(
        r"^(?:pvt\.?\s+)?(?:ltd\.?|limited|llc|inc\.?|corp\.?|corporation|co\.?)$",
        re.I,
    )
    for raw_line in compact:
        value = strip_bullet(raw_line)
        if result and suffix_re.fullmatch(value):
            previous = strip_bullet(result[-1])
            if looks_like_organization(previous) or re.search(r"\b(?:pvt|private|co)\.?$", previous, re.I):
                result[-1] = clean_space(previous + " " + value)
                continue
        result.append(raw_line)
    return result


def split_sections(text: str) -> dict[str, list[str]]:
    if is_interleaved_layout(text):
        return split_interleaved_sections(text)

    sections: dict[str, list[str]] = defaultdict(list)
    current = "preamble"
    for raw_line in text.splitlines():
        section, remainder = detect_heading(raw_line)
        if section:
            current = section
            if remainder:
                sections[current].append(remainder)
            continue
        sections[current].append(raw_line.rstrip())
    return dict(sections)


def load_resume_text(path: Path) -> str:
    raw = path.read_text(encoding="utf-8-sig")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        text = raw
    else:
        if isinstance(value, dict):
            text = next(
                (
                    value[key]
                    for key in ("text", "extracted_text", "resume_text", "content")
                    if isinstance(value.get(key), str)
                ),
                None,
            )
            if text is None:
                raise ValueError(
                    "JSON input must contain a top-level string field named 'text', "
                    "'extracted_text', 'resume_text', or 'content'."
                )
        elif isinstance(value, str):
            text = value
        else:
            raise ValueError("JSON input must be an object with 'text', or a JSON string.")

    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    if not text.strip():
        raise ValueError("The input contains no resume text.")
    return text.strip()


def find_email(text: str) -> str | None:
    labeled = re.search(r"\b(?:e-?mail)\s*:\s*(%s)" % EMAIL_RE.pattern, text, re.I)
    if labeled:
        return labeled.group(1)
    match = EMAIL_RE.search(text)
    return match.group(0) if match else None


def find_phone(text: str) -> str | None:
    candidates: list[tuple[int, int, str]] = []
    for match in PHONE_RE.finditer(text):
        value = clean_space(match.group(0)).strip(" ./-")
        digits = re.sub(r"\D", "", value)
        if not 7 <= len(digits) <= 15:
            continue
        if re.fullmatch(r"\d{4}\s*[-/]\s*\d{4}", value):
            continue
        context = text[max(0, match.start() - 24) : match.start()].casefold()
        score = 0
        if re.search(r"(?:phone|mobile|cell|tel)\s*:?[ \t]*$", context):
            score += 4
        if value.startswith("+"):
            score += 2
        if match.start() < min(len(text), 800):
            score += 1
        candidates.append((-score, match.start(), value))
    return min(candidates)[2] if candidates else None


def natural_name_case(value: str) -> str:
    return value.title() if value.isupper() else value


def find_name(preamble: list[str]) -> str | None:
    lines = [clean_space(strip_bullet(line)) for line in preamble if line.strip()]
    for value in lines[:12]:
        explicit = re.match(r"^(?:full\s+name|candidate|name)\s*:\s*(.+)$", value, re.I)
        if explicit:
            return natural_name_case(clean_space(explicit.group(1)))

    excluded = {"resume", "curriculum vitae", "cv", "professional resume"}
    for value in lines[:12]:
        if not value or len(value) > 70 or normalized_heading(value) in excluded:
            continue
        if "@" in value or re.search(r"https?://|www\.|linkedin|github", value, re.I):
            continue
        if re.search(r"\d", value) or any(mark in value for mark in ("\u00b7", "|", ";")):
            continue
        if ROLE_RE.fullmatch(value) or INSTITUTION_RE.search(value):
            continue
        words = value.split()
        letters = sum(char.isalpha() for char in value)
        if 2 <= len(words) <= 6 and letters >= max(4, int(len(value) * 0.6)):
            return natural_name_case(value)
    return None


def split_top_level(text: str, separators: str = ",;|\u00b7\u2022\n\t") -> list[str]:
    """Split a list without splitting punctuation inside parentheses or brackets."""
    text = re.sub(r"\s+/\s+", "|", text)
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    for char in text:
        if char in "([":
            depth += 1
        elif char in ")]" and depth:
            depth -= 1
        if char in separators and depth == 0:
            value = "".join(current).strip()
            if value:
                parts.append(value)
            current = []
        else:
            current.append(char)
    value = "".join(current).strip()
    if value:
        parts.append(value)
    return parts


def join_wrapped_lines(lines: list[str]) -> str:
    parts: list[str] = []
    for line in lines:
        value = strip_bullet(line)
        if not value:
            continue
        if parts and parts[-1].endswith("-") and value[:1].islower():
            parts[-1] = parts[-1][:-1] + value
        else:
            parts.append(value)
    return clean_space(" ".join(parts))


def parse_summary(lines: list[str]) -> str | None:
    return join_wrapped_lines(lines) or None


def parse_skills(lines: list[str]) -> list[str]:
    result: list[str] = []
    for raw_line in lines:
        value = strip_bullet(raw_line)
        if not value:
            continue
        if ":" in value:
            prefix, rest = value.split(":", 1)
            if 1 <= len(prefix.split()) <= 6 and rest.strip():
                value = rest.strip()
        # A comma without following whitespace is often punctuation inside a single
        # sidebar skill (for example "Confident,Positive Attitude"), not a list.
        if "," in value and not re.search(r",\s+", value):
            result.append(value)
        else:
            result.extend(split_top_level(value))
    return unique_strings(result)


def extract_date_range(value: str) -> tuple[str | None, str | None, str]:
    match = DATE_RANGE_RE.search(value)
    if match:
        remainder = clean_space(value[: match.start()] + " " + value[match.end() :])
        return clean_space(match.group("start")), clean_space(match.group("end")), remainder

    match = SINCE_RE.search(value)
    if match:
        remainder = clean_space(value[: match.start()] + " " + value[match.end() :])
        return clean_space(match.group("start")), "Present", remainder

    match = EXPECTED_RE.search(value)
    if match:
        remainder = clean_space(value[: match.start()] + " " + value[match.end() :])
        return None, clean_space(match.group("end")), remainder

    return None, None, clean_space(value)


def looks_like_role(value: str | None) -> bool:
    return bool(value and ROLE_RE.search(value))


def looks_like_organization(value: str | None) -> bool:
    return bool(value and ORG_RE.search(value))


HEADER_SEPARATOR_RE = re.compile(
    r"\s+(?:[\u2014\u2013]|\||@|\bat\b)\s+|\s+-\s+", re.I
)


def header_parts(value: str) -> list[str]:
    parts = [clean_space(part).strip(" ,|-") for part in HEADER_SEPARATOR_RE.split(value)]
    parts = [part for part in parts if part]
    if len(parts) == 1 and "," in parts[0]:
        comma_parts = [clean_space(part) for part in parts[0].split(",") if part.strip()]
        role_index = next(
            (index for index, part in enumerate(comma_parts) if looks_like_role(part)), None
        )
        if role_index is not None:
            role = comma_parts[role_index]
            organization = next(
                (part for index, part in enumerate(comma_parts) if index != role_index), None
            )
            parts = [organization, role] if organization else [role]
    return parts


def choose_title_employer(parts: list[str]) -> tuple[str | None, str | None]:
    if not parts:
        return None, None
    if len(parts) == 1:
        return parts[0], None

    role_indexes = [index for index, part in enumerate(parts) if looks_like_role(part)]
    if role_indexes:
        role_index = role_indexes[0]
        title = parts[role_index]
        employer = next((part for index, part in enumerate(parts) if index != role_index), None)
        return title, employer
    return parts[0], parts[1]


def new_experience(
    title: str | None = None,
    employer: str | None = None,
    start: str | None = None,
    end: str | None = None,
) -> dict[str, Any]:
    return {
        "title": title,
        "employer": employer,
        "start": start,
        "end": end,
        "achievements": [],
    }


def parse_experience_header(line: str) -> tuple[dict[str, Any], str]:
    start, end, without_dates = extract_date_range(strip_bullet(line))
    value = without_dates.strip(" ,-\u2013\u2014|")
    parts = header_parts(value)
    title, employer = choose_title_employer(parts)
    return new_experience(title, employer, start, end), value


def line_is_date_only(value: str) -> bool:
    start, end, remainder = extract_date_range(value)
    return bool((start or end) and not remainder.strip(" ()[],:|-"))


def add_achievement(item: dict[str, Any], value: str, continuation: bool = False) -> None:
    cleaned = clean_space(value)
    if not cleaned:
        return
    achievements: list[str] = item["achievements"]
    if continuation and achievements:
        achievements[-1] = clean_space(achievements[-1] + " " + cleaned)
    else:
        achievements.append(cleaned)


def parse_experience(lines: list[str]) -> list[dict[str, Any]]:
    nonempty = [line for line in lines if line.strip()]
    result: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    previous_was_bullet = False

    def flush() -> None:
        nonlocal current
        if current and (current["title"] or current["employer"] or current["achievements"]):
            current["achievements"] = unique_strings(current["achievements"])
            result.append(current)
        current = None

    for index, raw_line in enumerate(nonempty):
        value = strip_bullet(raw_line)
        if not value:
            continue
        is_bullet = bool(BULLET_RE.match(raw_line))
        next_value = strip_bullet(nonempty[index + 1]) if index + 1 < len(nonempty) else ""

        if is_bullet:
            if current is not None:
                add_achievement(current, value)
            previous_was_bullet = True
            continue

        parsed, remainder = parse_experience_header(value)
        has_dates = bool(parsed["start"] or parsed["end"])
        parts = header_parts(remainder)
        paired_header = len(parts) >= 2
        date_only = line_is_date_only(value)
        action_line = bool(ACTION_START_RE.search(value))
        role_line = looks_like_role(remainder) and not action_line
        organization_line = looks_like_organization(remainder) and not action_line
        short_line = len(value.split()) <= 16 and not value.endswith((".", ";"))
        next_is_bullet = bool(index + 1 < len(nonempty) and BULLET_RE.match(nonempty[index + 1]))
        next_has_dates = bool(DATE_RANGE_RE.search(next_value) or SINCE_RE.search(next_value))
        next_is_label = looks_like_role(next_value) or looks_like_organization(next_value)
        header_like = bool(
            (has_dates and not action_line)
            or (paired_header and not action_line)
            or ((role_line or organization_line) and short_line)
            or (
                not action_line
                and short_line
                and (next_is_bullet or next_has_dates or next_is_label)
            )
        )

        if date_only:
            if current is not None:
                current["start"] = current["start"] or parsed["start"]
                current["end"] = current["end"] or parsed["end"]
            previous_was_bullet = False
            continue

        if header_like:
            label = parsed["title"]
            if paired_header:
                flush()
                current = parsed
            elif current is None:
                if organization_line and not role_line:
                    current = new_experience(
                        employer=label, start=parsed["start"], end=parsed["end"]
                    )
                else:
                    current = parsed
            elif not current["achievements"]:
                if current["title"] is None and role_line:
                    current["title"] = label
                elif current["employer"] is None and (organization_line or not role_line):
                    current["employer"] = label
                elif current["title"] is None:
                    current["title"] = label
                else:
                    flush()
                    current = parsed
                if current is not None:
                    current["start"] = current["start"] or parsed["start"]
                    current["end"] = current["end"] or parsed["end"]
            else:
                flush()
                if organization_line and not role_line:
                    current = new_experience(
                        employer=label, start=parsed["start"], end=parsed["end"]
                    )
                else:
                    current = parsed
            previous_was_bullet = False
            continue

        if current is not None:
            continuation = previous_was_bullet or (
                bool(current["achievements"])
                and not current["achievements"][-1].endswith((".", "!", "?", ";"))
            )
            add_achievement(current, value, continuation=continuation)
        previous_was_bullet = False

    flush()
    return result


def split_education_labels(value: str) -> tuple[str | None, str | None]:
    parts = header_parts(value)
    degree = next((part for part in parts if DEGREE_RE.search(part)), None)
    institution = next(
        (part for part in parts if part != degree and INSTITUTION_RE.search(part)), None
    )
    if len(parts) == 1:
        if degree:
            return degree, None
        return None, parts[0]
    if degree and institution is None:
        institution = next((part for part in parts if part != degree), None)
    if institution and degree is None:
        degree = next((part for part in parts if part != institution and DEGREE_RE.search(part)), None)
    return degree, institution


def parse_education(lines: list[str]) -> list[dict[str, str | None]]:
    result: list[dict[str, str | None]] = []
    current: dict[str, str | None] | None = None

    def new_item() -> dict[str, str | None]:
        return {"degree": None, "institution": None, "start": None, "end": None}

    def complete(item: dict[str, str | None] | None) -> bool:
        return bool(item and item["degree"] and item["institution"])

    def flush() -> None:
        nonlocal current
        if current and (current["degree"] or current["institution"]):
            result.append(current)
        current = None

    for raw_line in lines:
        value = clean_space(strip_bullet(raw_line))
        if not value or IGNORE_EDUCATION_RE.match(value):
            continue
        start, end, label = extract_date_range(value)
        label = label.strip(" ,-\u2013\u2014|")

        if not label and (start or end):
            if current is not None:
                current["start"] = current["start"] or start
                current["end"] = current["end"] or end
            continue

        degree, institution = split_education_labels(label)
        if degree and institution:
            if complete(current):
                flush()
            if current is None:
                current = new_item()
            current["degree"] = current["degree"] or degree
            current["institution"] = current["institution"] or institution
        elif degree:
            if current and current["degree"] and complete(current):
                flush()
            if current is None:
                current = new_item()
            current["degree"] = current["degree"] or degree
        elif institution:
            if current and current["institution"] and complete(current):
                flush()
            if current is None:
                current = new_item()
            current["institution"] = current["institution"] or institution

        if current is not None:
            current["start"] = current["start"] or start
            current["end"] = current["end"] or end

    flush()
    return result


def parse_certifications(lines: list[str]) -> list[str]:
    values: list[str] = []
    for line in lines:
        value = strip_bullet(line)
        if not value:
            continue
        _, _, value = extract_date_range(value)
        value = TRAILING_DATE_RE.sub("", value).strip(" ,|-")
        for item in split_top_level(value, separators=";\u2022\n"):
            # In "Certificate | Issuer", the requested schema stores the name only.
            name = item.split("|", 1)[0].strip()
            if name:
                values.append(name)
    return unique_strings(values)


def parse_languages(lines: list[str]) -> list[str]:
    result: list[str] = []
    for line in lines:
        for item in split_top_level(strip_bullet(line), separators=",;|\u2022\n\t"):
            value = clean_space(item)
            if not value:
                continue
            if re.match(r"^.+\([^()]+\)$", value):
                result.append(value)
                continue
            match = re.match(r"^(.+?)\s*(?::|[\u2014\u2013-])\s*(.+)$", value)
            if match:
                value = f"{clean_space(match.group(1))} ({clean_space(match.group(2))})"
            else:
                match = re.match(
                    r"^(Native|Fluent|Advanced|Intermediate|Basic|Beginner)\s+(?:in\s+)?(.+)$",
                    value,
                    re.I,
                )
                if match:
                    value = f"{clean_space(match.group(2))} ({clean_space(match.group(1))})"
            result.append(value)
    return unique_strings(result)


def ensure_schema(result: dict[str, Any]) -> dict[str, Any]:
    """Enforce the exact public contract even if internal rules change later."""
    normalized = {key: result.get(key) for key in SCHEMA_KEYS}

    def nullable_string(value: Any) -> str | None:
        return clean_space(value) if isinstance(value, str) and value.strip() else None

    for key in ("name", "email", "phone", "summary"):
        normalized[key] = nullable_string(normalized[key])

    for key in ("skills", "certifications", "languages"):
        values = normalized[key] if isinstance(normalized[key], list) else []
        normalized[key] = unique_strings(value for value in values if isinstance(value, str))

    experience: list[dict[str, Any]] = []
    raw_experience = normalized["experience"] if isinstance(normalized["experience"], list) else []
    for item in raw_experience:
        if not isinstance(item, dict):
            continue
        achievements = item.get("achievements")
        experience.append(
            {
                "title": nullable_string(item.get("title")),
                "employer": nullable_string(item.get("employer")),
                "start": nullable_string(item.get("start")),
                "end": nullable_string(item.get("end")),
                "achievements": unique_strings(
                    value
                    for value in (achievements if isinstance(achievements, list) else [])
                    if isinstance(value, str)
                ),
            }
        )
    normalized["experience"] = experience

    education: list[dict[str, str | None]] = []
    raw_education = normalized["education"] if isinstance(normalized["education"], list) else []
    for item in raw_education:
        if not isinstance(item, dict):
            continue
        education.append(
            {
                "degree": nullable_string(item.get("degree")),
                "institution": nullable_string(item.get("institution")),
                "start": nullable_string(item.get("start")),
                "end": nullable_string(item.get("end")),
            }
        )
    normalized["education"] = education
    return normalized


def parse_resume(text: str) -> dict[str, Any]:
    sections = split_sections(text)
    preamble = sections.get("preamble", [])
    result = {
        "name": find_name(preamble),
        "email": find_email(text),
        "phone": find_phone(text),
        "summary": parse_summary(sections.get("summary", [])),
        "skills": parse_skills(sections.get("skills", [])),
        "experience": parse_experience(sections.get("experience", [])),
        "education": parse_education(sections.get("education", [])),
        "certifications": parse_certifications(sections.get("certifications", [])),
        "languages": parse_languages(sections.get("languages", [])),
    }
    return ensure_schema(result)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a raw-text or JSON-wrapped resume into JSON offline."
    )
    parser.add_argument("input", type=Path, help="Path to a .txt or .json input file")
    parser.add_argument(
        "-o", "--output", type=Path, help="Output .json path (default: print to stdout)"
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=200_000,
        help="Reject inputs longer than this many characters (default: 200000)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        text = load_resume_text(args.input)
        if len(text) > args.max_chars:
            raise ValueError(
                f"Resume contains {len(text):,} characters; limit is {args.max_chars:,}."
            )
        output_json = json.dumps(parse_resume(text), ensure_ascii=False, indent=2)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(output_json + "\n", encoding="utf-8")
        else:
            print(output_json)
        return 0
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())