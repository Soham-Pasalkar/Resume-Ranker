from fastapi import APIRouter, HTTPException
from backend.app.schemas.job import JobDescriptionRequest, JobDescriptionAnalysis
from backend.app.nlp.taxonomy import extract_skills_from_text, extract_domains_from_text
from backend.app.nlp.tokenizer import get_top_keywords
from backend.app.extractors.experience import extract_years_of_experience
from backend.app.core.store import store

router = APIRouter(prefix="/jobs", tags=["Jobs"])


@router.post("/analyze", response_model=JobDescriptionAnalysis)
async def analyze_job_description(payload: JobDescriptionRequest):
    """
    Analyze a Job Description, extracting required skills, target YOE,
    industry domains, and top keywords.
    """
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Job description text cannot be empty.")

    store.active_jd = text

    skills = extract_skills_from_text(text)
    # Add any explicit mandatory skills
    if payload.mandatory_skills:
        for ms in payload.mandatory_skills:
            if ms.strip():
                skills.add(ms.strip().lower())

    yoe = payload.target_yoe if payload.target_yoe is not None else extract_years_of_experience(text)
    domains = extract_domains_from_text(text)
    keywords = get_top_keywords(text, top_n=15)
    word_count = len(text.split())

    return JobDescriptionAnalysis(
        title=payload.title,
        skills_detected=sorted(list(skills)),
        required_yoe=yoe,
        domains_detected=domains,
        top_keywords=keywords,
        word_count=word_count,
    )
