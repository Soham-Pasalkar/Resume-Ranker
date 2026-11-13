from typing import List, Optional, Dict
from backend.app.schemas.ranking import WeightProfile, KnockoutRules, KnockoutStatus
from backend.app.schemas.candidate import CandidateProfile

DEGREE_RANK = {
    "Doctorate": 4,
    "Master's": 3,
    "Bachelor's": 2,
    "Associate": 1,
}

PRESET_PROFILES: Dict[str, WeightProfile] = {
    "balanced": WeightProfile(semantic=0.40, skills=0.35, experience=0.15, domain=0.10),
    "specialist": WeightProfile(semantic=0.25, skills=0.50, experience=0.15, domain=0.10),
    "leadership": WeightProfile(semantic=0.30, skills=0.20, experience=0.40, domain=0.10),
    "generalist": WeightProfile(semantic=0.35, skills=0.25, experience=0.20, domain=0.20),
}


def evaluate_knockouts(
    candidate: CandidateProfile,
    rules: Optional[KnockoutRules]
) -> KnockoutStatus:
    """
    Evaluate candidate against hard knockout constraints (must-have skills,
    minimum required experience, minimum education credentials).
    """
    if not rules:
        return KnockoutStatus(is_knocked_out=False, reasons=[])

    reasons: List[str] = []
    candidate_skills_set = set(candidate.detected_skills)

    # 1. Mandatory skills check
    if rules.mandatory_skills:
        missing_mandatory = [
            s for s in rules.mandatory_skills
            if s.lower() not in candidate_skills_set
        ]
        if missing_mandatory:
            reasons.append(f"Missing mandatory skill(s): {', '.join(missing_mandatory).title()}")

    # 2. Strict minimum YOE check
    if rules.min_yoe is not None and rules.min_yoe > 0:
        if candidate.detected_yoe < rules.min_yoe:
            reasons.append(
                f"Does not meet minimum experience: has {candidate.detected_yoe:.0f} yrs, required {rules.min_yoe:.0f} yrs"
            )

    # 3. Minimum education degree check
    if rules.min_degree:
        min_rank = DEGREE_RANK.get(rules.min_degree, 0)
        cand_deg = candidate.highest_degree
        cand_rank = DEGREE_RANK.get(cand_deg, 0) if cand_deg else 0

        if cand_rank < min_rank:
            reasons.append(
                f"Degree requirement not met: required at least {rules.min_degree}, detected {cand_deg or 'None'}"
            )

    return KnockoutStatus(
        is_knocked_out=len(reasons) > 0,
        reasons=reasons
    )
