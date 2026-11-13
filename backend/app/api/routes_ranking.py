import io
import csv
from typing import Dict
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
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


@router.post("/export/csv")
async def export_ranking_csv(payload: RankingRequest):
    """Export ranking results directly as a downloadable CSV report."""
    ranking_result = await rank_resumes(payload)

    output = io.StringIO()
    writer = csv.writer(output)

    # Header
    writer.writerow([
        "Rank",
        "Candidate Name",
        "Overall Score",
        "Match Level",
        "Semantic Score",
        "Skill Score",
        "Experience Score",
        "Domain Score",
        "Detected YOE",
        "Highest Degree",
        "Institution",
        "Matched Skills",
        "Missing Skills",
        "Knocked Out",
        "Knockout Reasons",
        "Executive Rationale"
    ])

    for r in ranking_result.results:
        writer.writerow([
            r.rank,
            r.candidate.name,
            r.score.overall_score,
            r.score.match_level,
            r.score.semantic_score,
            r.score.skill_score,
            r.score.experience_score,
            r.score.domain_score,
            r.candidate.detected_yoe,
            r.candidate.highest_degree or "N/A",
            r.candidate.institution or "N/A",
            "; ".join(r.evidence.matched_skills),
            "; ".join(r.evidence.missing_skills),
            "YES" if r.knockout.is_knocked_out else "NO",
            "; ".join(r.knockout.reasons),
            r.evidence.rationale
        ])

    csv_data = output.getvalue()
    return Response(
        content=csv_data,
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="resume_rankings_report.csv"'}
    )
