import re
from backend.app.schemas.candidate import ContactInfo


EMAIL_REGEX = re.compile(
    r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
)

PHONE_REGEX = re.compile(
    r"(?:(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}|\b\d{10}\b)"
)

LINKEDIN_REGEX = re.compile(
    r"(?:https?://)?(?:www\.)?linkedin\.com/in/[a-zA-Z0-9_-]+",
    re.IGNORECASE
)

GITHUB_REGEX = re.compile(
    r"(?:https?://)?(?:www\.)?github\.com/[a-zA-Z0-9_-]+",
    re.IGNORECASE
)

LOCATION_REGEX = re.compile(
    r"\b([A-Z][a-zA-Z\s]+,\s*(?:[A-Z]{2}|USA|Canada|UK|India|Germany|Australia))\b"
)


def extract_contact_info(text: str) -> ContactInfo:
    """Extract contact information from text using high-precision regular expressions."""
    # Email
    email_match = EMAIL_REGEX.search(text)
    email = email_match.group(0) if email_match else None

    # Phone
    phone_match = PHONE_REGEX.search(text)
    phone = phone_match.group(0) if phone_match else None

    # LinkedIn
    linkedin_match = LINKEDIN_REGEX.search(text)
    linkedin = linkedin_match.group(0) if linkedin_match else None

    # GitHub
    github_match = GITHUB_REGEX.search(text)
    github = github_match.group(0) if github_match else None

    # Location heuristic (looks in the first 1000 characters)
    header_chunk = text[:1000]
    location_match = LOCATION_REGEX.search(header_chunk)
    location = location_match.group(0) if location_match else None

    return ContactInfo(
        email=email,
        phone=phone,
        linkedin=linkedin,
        github=github,
        location=location,
    )
