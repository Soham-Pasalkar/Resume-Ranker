import re
from typing import List, Set


def extract_evidence_citations(resume_text: str, matched_skills: Set[str], max_citations: int = 4) -> List[str]:
    """
    Find sentences in the resume text that contain matched skills,
    providing direct citations of candidate qualifications.
    """
    if not matched_skills or not resume_text:
        return []

    # Split resume into sentences
    raw_sentences = re.split(r"(?<=[.!?\n])\s+", resume_text)
    citations = []

    for sentence in raw_sentences:
        clean = sentence.strip()
        # Look for sentences of reasonable length that mention matched skills
        if 20 <= len(clean) <= 220:
            clean_lower = clean.lower()
            contained = [s for s in matched_skills if re.search(r"\b" + re.escape(s) + r"\b", clean_lower)]
            if contained:
                # Format snippet with bold skill tag
                citations.append(clean)
                if len(citations) >= max_citations:
                    break

    return citations
