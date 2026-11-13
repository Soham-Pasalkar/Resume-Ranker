import re
from backend.app.extractors.contact import EMAIL_REGEX, PHONE_REGEX, LINKEDIN_REGEX, GITHUB_REGEX


def anonymize_resume(
    text: str,
    candidate_name: str = "",
    institution: str = "",
    graduation_year: int = None
) -> str:
    """
    Sanitize personal identifying information (PII) from resume text
    to support unbiased / blind hiring evaluations.
    """
    anonymized = text

    # Redact Email FIRST before name token substitution splits email username
    anonymized = EMAIL_REGEX.sub("[REDACTED_EMAIL]", anonymized)

    # Redact Phone
    anonymized = PHONE_REGEX.sub("[REDACTED_PHONE]", anonymized)

    # Redact Social URLs
    anonymized = LINKEDIN_REGEX.sub("[REDACTED_LINKEDIN]", anonymized)
    anonymized = GITHUB_REGEX.sub("[REDACTED_GITHUB]", anonymized)

    # Redact generic URLs
    anonymized = re.sub(r"https?://\S+", "[REDACTED_URL]", anonymized)

    # Redact explicit candidate name tokens
    if candidate_name and len(candidate_name) > 2:
        name_tokens = candidate_name.split()
        for token in name_tokens:
            if len(token) > 2:
                anonymized = re.sub(rf"\b{re.escape(token)}\b", "[REDACTED_NAME]", anonymized, flags=re.IGNORECASE)

    # Redact institution if provided
    if institution and len(institution) > 3:
        anonymized = re.sub(rf"\b{re.escape(institution)}\b", "[REDACTED_INSTITUTION]", anonymized, flags=re.IGNORECASE)

    # Redact graduation year if provided
    if graduation_year:
        anonymized = re.sub(rf"\b{graduation_year}\b", "[REDACTED_YEAR]", anonymized)

    # Redact potential gender pronouns
    anonymized = re.sub(r"\b(he|she|him|her|his|hers)\b", "[THEY/THEIR]", anonymized, flags=re.IGNORECASE)

    return anonymized
