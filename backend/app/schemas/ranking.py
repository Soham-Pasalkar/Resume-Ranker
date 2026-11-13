from pydantic import BaseModel, Field
from typing import List, Optional
from backend.app.schemas.candidate import CandidateProfile


class WeightProfile(BaseModel):
    semantic: float = Field(default=0.40, ge=0.0, le=1.0)
    skills: float = Field(default=0.35, ge=0.0, le=1.0)
    experience: float = Field(default=0.15, ge=0.0, le=1.0)
    domain: float = Field(default=0.10, ge=0.0, le=1.0)

    def normalized(self) -> "WeightProfile":
        total = self.semantic + self.skills + self.experience + self.domain
        if total == 0:
            return WeightProfile(semantic=0.25, skills=0.25, experience=0.25, domain=0.25)
        return WeightProfile(
            semantic=self.semantic / total,
            skills=self.skills / total,
            experience=self.experience / total,
            domain=self.domain / total,
        )


class ScoreBreakdown(BaseModel):
    overall_score: float = Field(..., ge=0.0, le=100.0)
    semantic_score: float = Field(..., ge=0.0, le=100.0)
    skill_score: float = Field(..., ge=0.0, le=100.0)
    experience_score: float = Field(..., ge=0.0, le=100.0)
    domain_score: float = Field(..., ge=0.0, le=100.0)
    match_level: str = Field(..., description="'Strong Match', 'Moderate Match', or 'Weak Match'")


class CandidateEvidence(BaseModel):
    matched_skills: List[str]
    missing_skills: List[str]
    experience_note: str
    domain_note: str
    key_citations: List[str] = Field(default_factory=list, description="Direct text excerpts supporting match")
    interview_questions: List[str] = Field(default_factory=list, description="Targeted technical screening questions")


class RankedCandidate(BaseModel):
    rank: int
    candidate: CandidateProfile
    score: ScoreBreakdown
    evidence: CandidateEvidence


class RankingRequest(BaseModel):
    jd_text: str = Field(..., min_length=20)
    candidate_ids: Optional[List[str]] = None
    weights: Optional[WeightProfile] = None
    blind_hiring_mode: bool = Field(default=False, description="Redact candidate names & PII during evaluation")
    min_score_filter: float = Field(default=0.0, ge=0.0, le=100.0)


class RankingSummary(BaseModel):
    total_candidates: int
    strong_matches: int
    average_score: float
    top_score: float


class RankingResponse(BaseModel):
    summary: RankingSummary
    results: List[RankedCandidate]
