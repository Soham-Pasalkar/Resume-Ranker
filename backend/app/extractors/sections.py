import re
from typing import Dict

SECTION_HEADERS = {
    "summary": re.compile(r"^(?:professional\s+summary|summary|profile|about\s+me|objective)", re.IGNORECASE),
    "experience": re.compile(r"^(?:work\s+experience|professional\s+experience|experience|employment\s+history|career\s+history)", re.IGNORECASE),
    "education": re.compile(r"^(?:education|academic\s+background|academics|qualifications)", re.IGNORECASE),
    "skills": re.compile(r"^(?:technical\s+skills|core\s+competencies|skills\s+&\s+technologies|skills|expertise|technologies)", re.IGNORECASE),
    "projects": re.compile(r"^(?:projects|key\s+projects|notable\s+projects|personal\s+projects)", re.IGNORECASE),
    "certifications": re.compile(r"^(?:certifications|licenses|certifications\s+and\s+licenses|awards)", re.IGNORECASE),
}


def segment_resume_sections(text: str) -> Dict[str, str]:
    """
    Split resume text into functional sections based on standard headings.
    Returns a dictionary mapping section name -> section text.
    """
    lines = text.splitlines()
    sections: Dict[str, list] = {"header": []}
    current_section = "header"

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Check if line looks like a section header (short line, matches header regex)
        if len(stripped) < 40:
            clean_heading = re.sub(r"[:\-\#\*]+", "", stripped).strip()
            matched_header = None
            for sec_name, sec_regex in SECTION_HEADERS.items():
                if sec_regex.match(clean_heading):
                    matched_header = sec_name
                    break

            if matched_header:
                current_section = matched_header
                if current_section not in sections:
                    sections[current_section] = []
                continue

        sections[current_section].append(line)

    return {k: "\n".join(v).strip() for k, v in sections.items() if v}
