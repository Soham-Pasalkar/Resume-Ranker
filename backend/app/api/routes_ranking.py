from typing import Dict
from fastapi import APIRouter, HTTPException
from backend.app.schemas.ranking import RankingRequest, RankingResponse, WeightProfile
from backend.app.engine.scoring import RankingEngine
from backend.app.engine.constraints import PRESET_PROFILES
from backend.app.nlp.taxonomy import extract_skills_from_text, extract_domains_from_text
from backend.app.extractors.experience import extract_years_of_experience
from backend.app.core.store import store

router = APIRouter(prefix="/ranking", tags=["Ranking"])


@router.get("/profiles", response_model=Dict[str, WeightProfile])
async def get_weight_profiles():
    """Return all available pre-configured recruitment weight profiles."""
    return PRESET_PROFILES


@router.post("/rank", response_model=RankingResponse)
async def rank_resumes(payload: RankingRequest):
    """
    Rank all loaded candidates against the Job Description.
    Applies hybrid semantic matching, Reciprocal Rank Fusion, YOE verification,
    hard knockout checks, and generates evidence citations and interview prep.
    """
    jd_text = payload.jd_text.strip()
    if not jd_text:
        raise HTTPException(status_code=400, detail="Job description text cannot be empty.")

    all_candidates = store.get_all_candidates()
    if payload.candidate_ids:
        candidates = [c for c in all_candidates if c.id in payload.candidate_ids]
    else:
        candidates = all_candidates

    if not candidates:
        raise HTTPException(
            status_code=400,
            detail="No candidates available for ranking. Please upload resumes first."
        )

    # Extract JD metadata
    jd_skills = extract_skills_from_text(jd_text)
    jd_yoe = extract_years_of_experience(jd_text)
    jd_domains = extract_domains_from_text(jd_text)

    # Execute ranking
    ranking_result = RankingEngine.rank_candidates(
        candidates=candidates,
        jd_text=jd_text,
        jd_skills=jd_skills,
        jd_yoe=jd_yoe,
        jd_domains=jd_domains,
        weights=payload.weights,
        preset_profile=payload.preset_profile,
        knockout_rules=payload.knockout_rules,
        min_score=payload.min_score_filter,
    )

    return ranking_result
