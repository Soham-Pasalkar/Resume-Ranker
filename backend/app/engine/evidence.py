import re
from typing import List, Set, Dict


def extract_evidence_citations(
    resume_text: str,
    matched_skills: Set[str],
    sections: Dict[str, str] = None,
    max_citations: int = 4
) -> List[str]:
    """
    Find sentences in the resume text that contain matched skills,
    preferring evidence originating from the Experience and Projects sections.
    """
    if not matched_skills:
        return []

    # If sections exist, search experience and projects first
    search_texts = []
    if sections:
        if "experience" in sections:
            search_texts.append(sections["experience"])
        if "projects" in sections:
            search_texts.append(sections["projects"])
    if not search_texts:
        search_texts.append(resume_text)

    citations = []
    seen = set()

    for chunk in search_texts:
        sentences = re.split(r"(?<=[.!?\n])\s+", chunk)
        for s in sentences:
            clean = s.strip()
            if 25 <= len(clean) <= 220 and clean not in seen:
                clean_lower = clean.lower()
                contained = [sk for sk in matched_skills if re.search(r"\b" + re.escape(sk) + r"\b", clean_lower)]
                if contained:
                    seen.add(clean)
                    citations.append(clean)
                    if len(citations) >= max_citations:
                        return citations

    return citations


def generate_executive_rationale(
    candidate_name: str,
    overall_score: float,
    matched_skills: List[str],
    missing_skills: List[str],
    candidate_yoe: float,
    jd_yoe: float,
    highest_degree: str = None
) -> str:
    """
    Generate an executive decision rationale summarizing why the candidate fits
    and highlighting key strengths or hiring reservations.
    """
    deg_str = f" with a {highest_degree}" if highest_degree else ""
    yoe_str = f"{candidate_yoe:.0f} years of relevant experience" if candidate_yoe > 0 else "unspecified experience duration"

    if overall_score >= 65.0:
        top_skills = ", ".join(matched_skills[:4]).title() if matched_skills else "core competencies"
        rationale = (
            f"{candidate_name} is a high-priority match ({overall_score:.0f}/100){deg_str}. "
            f"The candidate brings {yoe_str} and demonstrates hands-on proficiency in {top_skills}. "
        )
        if missing_skills:
            rationale += f"Minor gaps identified in {', '.join(missing_skills[:2]).title()}, which can be addressed during screening."
        else:
            rationale += "Complete coverage across all core technical requirements."
    elif overall_score >= 40.0:
        rationale = (
            f"{candidate_name} demonstrates moderate alignment ({overall_score:.0f}/100) with {yoe_str}. "
            f"Matches key areas like {', '.join(matched_skills[:3]).title() if matched_skills else 'foundational skills'}, "
            f"but has notable gaps in {', '.join(missing_skills[:3]).title() if missing_skills else 'specialized requirements'}."
        )
    else:
        rationale = (
            f"{candidate_name} exhibits low alignment ({overall_score:.0f}/100). "
            f"Candidate background lacks substantial coverage of primary requirements ({', '.join(missing_skills[:4]).title() if missing_skills else 'required stack'})."
        )

    return rationale
