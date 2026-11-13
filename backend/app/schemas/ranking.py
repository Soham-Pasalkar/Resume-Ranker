from pydantic import BaseModel, Field
from typing import List, Optional, Dict
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


class KnockoutRules(BaseModel):
    mandatory_skills: List[str] = Field(default_factory=list, description="Must-have skills")
    min_yoe: Optional[float] = Field(default=None, description="Strict minimum years of experience required")
    min_degree: Optional[str] = Field(default=None, description="Minimum degree, e.g. Bachelor's or Master's")
    apply_penalty_instead: bool = Field(default=False, description="Apply heavy penalty instead of hard knockout")


class KnockoutStatus(BaseModel):
    is_knocked_out: bool = False
    reasons: List[str] = Field(default_factory=list)


class ScoreBreakdown(BaseModel):
    overall_score: float = Field(..., ge=0.0, le=100.0)
    semantic_score: float = Field(..., ge=0.0, le=100.0)
    skill_score: float = Field(..., ge=0.0, le=100.0)
    experience_score: float = Field(..., ge=0.0, le=100.0)
    domain_score: float = Field(..., ge=0.0, le=100.0)
    rrf_score: Optional[float] = Field(default=None, description="Reciprocal Rank Fusion score")
    match_level: str = Field(..., description="'Strong Match', 'Moderate Match', or 'Weak Match'")


class CandidateEvidence(BaseModel):
    matched_skills: List[str]
    missing_skills: List[str]
    experience_note: str
    domain_note: str
    rationale: str = Field(default="", description="Executive hiring rationale")
    key_citations: List[str] = Field(default_factory=list, description="Direct text excerpts supporting match")
    interview_questions: List[str] = Field(default_factory=list, description="Targeted technical screening questions")


class RankedCandidate(BaseModel):
    rank: int
    candidate: CandidateProfile
    score: ScoreBreakdown
    evidence: CandidateEvidence
    knockout: KnockoutStatus = Field(default_factory=KnockoutStatus)
    recruitment_status: str = Field(
        default="Applied",
        description="Recruitment pipeline stage: Applied, Shortlisted, Interview, Selected, Rejected"
    )


class RankingRequest(BaseModel):
    jd_text: str = Field(..., min_length=20)
    candidate_ids: Optional[List[str]] = None
    preset_profile: Optional[str] = Field(default="balanced", description="balanced, specialist, leadership, or generalist")
    weights: Optional[WeightProfile] = None
    knockout_rules: Optional[KnockoutRules] = None
    blind_hiring_mode: bool = Field(default=False, description="Redact candidate names & PII during evaluation")
    min_score_filter: float = Field(default=0.0, ge=0.0, le=100.0)


class RankingSummary(BaseModel):
    total_candidates: int
    strong_matches: int
    average_score: float
    top_score: float
    knockouts_count: int = 0
    stage_counts: Dict[str, int] = Field(
        default_factory=lambda: {
            "Applied": 0,
            "Shortlisted": 0,
            "Interview": 0,
            "Selected": 0,
            "Rejected": 0,
        }
    )


class RankingResponse(BaseModel):
    summary: RankingSummary
    results: List[RankedCandidate]
