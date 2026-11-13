from backend.app.extractors.contact import extract_contact_info
from backend.app.extractors.name import extract_candidate_name
from backend.app.extractors.experience import extract_years_of_experience
from backend.app.extractors.sections import segment_resume_sections
from backend.app.extractors.anonymizer import anonymize_resume
from backend.app.extractors.education import extract_education_info


def test_contact_extraction(sample_senior_resume_text):
    info = extract_contact_info(sample_senior_resume_text)
    assert info.email == "alex.rivera@example.com"
    assert "555" in info.phone
    assert "linkedin.com/in/alexriveradev" in info.linkedin
    assert "github.com/alexrivera" in info.github


def test_name_extraction(sample_senior_resume_text):
    name = extract_candidate_name(sample_senior_resume_text, "alex_rivera_resume.pdf")
    assert name == "Alex Rivera"


def test_name_extraction_fallback():
    raw_headerless = "Software developer with skills in python."
    name = extract_candidate_name(raw_headerless, "Sarah_Connor_CV_2024.pdf")
    assert name == "Sarah Connor"


def test_experience_extraction(sample_senior_resume_text):
    yoe = extract_years_of_experience(sample_senior_resume_text)
    assert yoe >= 5.0


def test_sections_segmentation(sample_senior_resume_text):
    sections = segment_resume_sections(sample_senior_resume_text)
    assert "summary" in sections or "experience" in sections
    assert "skills" in sections


def test_anonymizer(sample_senior_resume_text):
    anonymized = anonymize_resume(
        sample_senior_resume_text,
        candidate_name="Alex Rivera",
        institution="University of California, Berkeley",
        graduation_year=2018
    )
    assert "alex.rivera@example.com" not in anonymized
    assert "[REDACTED_EMAIL]" in anonymized
    assert "[REDACTED_PHONE]" in anonymized
    assert "Alex Rivera" not in anonymized
    assert "[REDACTED_INSTITUTION]" in anonymized
    assert "2018" not in anonymized or "[REDACTED_YEAR]" in anonymized


def test_education_extraction(sample_senior_resume_text):
    edu = extract_education_info(sample_senior_resume_text)
    assert edu["highest_degree"] == "Bachelor's"
    assert "Berkeley" in edu["institution"] or "California" in edu["institution"]
    assert edu["graduation_year"] == 2018
