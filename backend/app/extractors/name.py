import re
from typing import Optional

NON_NAME_WORDS = {
    "resume", "curriculum", "vitae", "cv", "summary", "profile", "contact",
    "phone", "email", "address", "experience", "education", "skills",
    "developer", "engineer", "scientist", "manager", "lead", "analyst",
    "page", "github", "linkedin", "portfolio", "http", "https", "www"
}


def extract_candidate_name(text: str, filename: str = "") -> str:
    """
    Extract candidate name from the top header of the resume.
    Falls back gracefully to the sanitized filename if no valid name line is found.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    
    # Check the first 8 lines
    for line in lines[:8]:
        # Strip bullets or non-alpha characters
        clean_line = re.sub(r"[^a-zA-Z\s\.]", "", line).strip()
        tokens = clean_line.split()
        
        # Valid name typically 2 to 4 words, each capitalized, not numbers, not emails
        if 2 <= len(tokens) <= 4:
            # Check length of each token and if any token matches non-name words
            if any(t.lower() in NON_NAME_WORDS for t in tokens):
                continue
            
            # Check if all tokens start with an uppercase letter
            if all(t[0].isupper() for t in tokens if t):
                return clean_line

    # Fallback to sanitized filename
    if filename:
        stem = filename.rsplit(".", 1)[0]
        # Replace hyphens/underscores with spaces
        clean_stem = re.sub(r"[_\-]+", " ", stem).strip()
        # Remove words like "resume", "cv", "final", "2024", etc.
        filtered = [
            w.capitalize() for w in clean_stem.split()
            if w.lower() not in NON_NAME_WORDS and not w.isdigit()
        ]
        if filtered:
            return " ".join(filtered)
        return stem

    return "Candidate"
