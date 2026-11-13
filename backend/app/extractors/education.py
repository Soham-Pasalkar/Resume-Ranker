import re
from typing import Dict, Any
from backend.app.extractors.sections import segment_resume_sections

DEGREE_LEVELS = [
    ("Doctorate", re.compile(r"\b(?:ph\.?d\.?|doctor\s+of\s+philosophy|doctorate)\b", re.IGNORECASE)),
    ("Master's", re.compile(r"\b(?:master(?:'?s)?|m\.?s\.?|m\.?a\.?|m\.?tech|m\.?sc\.?|mba)\b", re.IGNORECASE)),
    ("Bachelor's", re.compile(r"\b(?:bachelor(?:'?s)?|b\.?s\.?|b\.?a\.?|b\.?tech|b\.?e\.?|b\.?sc\.?)\b", re.IGNORECASE)),
    ("Associate", re.compile(r"\b(?:associate(?:'?s)?|a\.?s\.?|a\.?a\.?)\b", re.IGNORECASE)),
]

INSTITUTION_PATTERNS = [
    # University of California, Berkeley / University of Oxford
    re.compile(r"\b(University\s+of\s+[A-Z][a-zA-Z\s,]+)\b"),
    # Stanford University / MIT / Georgia Institute of Technology
    re.compile(r"\b([A-Z][a-zA-Z\s]+(?:University|College|Institute(?:\s+of\s+Technology)?|Polytechnic|Academy))\b"),
]

GRAD_YEAR_PATTERN = re.compile(r"\b(19[89]\d|20[0-2]\d)\b")


def extract_education_info(text: str, education_section: str = "") -> Dict[str, Any]:
    """
    Extract highest degree attained, institution name, and graduation year.
    Prioritizes the extracted Education section if available.
    """
    if not education_section.strip():
        sections = segment_resume_sections(text)
        education_section = sections.get("education", "")

    search_target = education_section if education_section.strip() else text

    highest_degree = None
    for deg_name, deg_regex in DEGREE_LEVELS:
        if deg_regex.search(search_target):
            highest_degree = deg_name
            break

    # If no degree found in education section, check full text
    if not highest_degree and search_target != text:
        for deg_name, deg_regex in DEGREE_LEVELS:
            if deg_regex.search(text):
                highest_degree = deg_name
                break

    # Institution
    institution = None
    for pat in INSTITUTION_PATTERNS:
        match = pat.search(search_target)
        if match:
            raw_inst = match.group(1).strip()
            institution = re.sub(r"[\s,\|]+$", "", raw_inst)
            break

    # Graduation year within education section
    year_matches = [int(y) for y in GRAD_YEAR_PATTERN.findall(search_target)]
    valid_years = [y for y in year_matches if 1985 <= y <= 2030]
    grad_year = max(valid_years) if valid_years else None

    return {
        "highest_degree": highest_degree,
        "institution": institution,
        "graduation_year": grad_year,
    }
