from typing import List, Set, Dict
from backend.app.schemas.candidate import CandidateProfile
from backend.app.schemas.ranking import (
    WeightProfile,
    ScoreBreakdown,
    CandidateEvidence,
    RankedCandidate,
    RankingSummary,
    RankingResponse,
)
from backend.app.nlp.embeddings import semantic_engine
from backend.app.nlp.bm25 import bm25_matcher
from backend.app.nlp.tokenizer import clean_and_tokenize
from backend.app.engine.evidence import extract_evidence_citations
from backend.app.engine.interview import generate_interview_questions


def calculate_match_label(score: float) -> str:
    if score >= 65.0:
        return "Strong Match"
    elif score >= 40.0:
        return "Moderate Match"
    return "Weak Match"


def compute_skill_overlap(jd_skills: Set[str], candidate_skills: Set[str]) -> float:
    if not jd_skills:
        return 0.8  # Neutral score if JD specifies no skills
    matched = jd_skills & candidate_skills
    return len(matched) / len(jd_skills)


def compute_experience_fit(jd_yoe: float, candidate_yoe: float) -> float:
    if jd_yoe <= 0:
        return 0.8  # Neutral score if JD specifies no YOE requirement
    if candidate_yoe >= jd_yoe:
        # Slight overqualification damping if > 2.5x
        if candidate_yoe > jd_yoe * 2.5:
            penalty = (candidate_yoe - jd_yoe * 2.5) * 0.02
            return max(0.7, 1.0 - penalty)
        return 1.0
    return max(0.1, candidate_yoe / jd_yoe)


def compute_domain_fit(jd_domains: List[str], candidate_domains: List[str]) -> float:
    if not jd_domains:
        return 0.75  # Neutral if JD doesn't mandate specific industry domain
    overlap = len(set(jd_domains) & set(candidate_domains))
    return min(1.0, overlap / max(1, len(jd_domains)))


class RankingEngine:
    """Core production ranking engine that orchestrates multi-factor evaluation."""

    @classmethod
    def evaluate_candidate(
        cls,
        candidate: CandidateProfile,
        jd_text: str,
        jd_skills: Set[str],
        jd_yoe: float,
        jd_domains: List[str],
        weights: WeightProfile,
    ) -> RankedCandidate:
        candidate_text = candidate.raw_text
        candidate_skills = set(candidate.detected_skills)
        candidate_domains = candidate.detected_domains
        candidate_yoe = candidate.detected_yoe

        # 1. Semantic + Lexical Match
        cosine_sim = semantic_engine.compute_similarity(jd_text, candidate_text)
        query_tokens = clean_and_tokenize(jd_text)
        doc_tokens = clean_and_tokenize(candidate_text)
        bm25_sim = bm25_matcher.compute_lexical_score(query_tokens, doc_tokens)
        
        # Combined semantic score (80% cosine, 20% BM25 lexical)
        semantic_score = (0.80 * cosine_sim + 0.20 * bm25_sim) * 100.0

        # 2. Skill overlap score
        skill_score = compute_skill_overlap(jd_skills, candidate_skills) * 100.0

        # 3. Experience fit score
        exp_score = compute_experience_fit(jd_yoe, candidate_yoe) * 100.0

        # 4. Domain fit score
        domain_score = compute_domain_fit(jd_domains, candidate_domains) * 100.0

        # Normalized weights
        w = weights.normalized()
        overall = (
            w.semantic * semantic_score +
            w.skills * skill_score +
            w.experience * exp_score +
            w.domain * domain_score
        )

        matched_skills = sorted(list(jd_skills & candidate_skills))
        missing_skills = sorted(list(jd_skills - candidate_skills))

        # Experience note
        if jd_yoe > 0:
            if candidate_yoe >= jd_yoe:
                exp_note = f"Meets requirement: has {candidate_yoe:.0f} yrs (Target: {jd_yoe:.0f} yrs)"
            else:
                exp_note = f"Short by {jd_yoe - candidate_yoe:.0f} yr(s): has {candidate_yoe:.0f} yrs (Target: {jd_yoe:.0f} yrs)"
        else:
            exp_note = f"Detected {candidate_yoe:.0f} yrs experience"

        # Domain note
        if candidate_domains:
            domain_note = f"Verified alignment in: {', '.join(candidate_domains).title()}"
        else:
            domain_note = "No direct industry domain keyword overlap detected"

        citations = extract_evidence_citations(candidate_text, set(matched_skills))
        interview_qs = generate_interview_questions(
            set(missing_skills),
            set(matched_skills),
            jd_yoe,
            candidate_yoe
        )

        breakdown = ScoreBreakdown(
            overall_score=round(overall, 1),
            semantic_score=round(semantic_score, 1),
            skill_score=round(skill_score, 1),
            experience_score=round(exp_score, 1),
            domain_score=round(domain_score, 1),
            match_level=calculate_match_label(overall),
        )

        evidence = CandidateEvidence(
            matched_skills=matched_skills,
            missing_skills=missing_skills,
            experience_note=exp_note,
            domain_note=domain_note,
            key_citations=citations,
            interview_questions=interview_qs,
        )

        return RankedCandidate(
            rank=0,  # Populated after sorting
            candidate=candidate,
            score=breakdown,
            evidence=evidence,
        )

    @classmethod
    def rank_candidates(
        cls,
        candidates: List[CandidateProfile],
        jd_text: str,
        jd_skills: Set[str],
        jd_yoe: float,
        jd_domains: List[str],
        weights: WeightProfile = WeightProfile(),
        min_score: float = 0.0,
    ) -> RankingResponse:
        evaluated: List[RankedCandidate] = []
        for cand in candidates:
            res = cls.evaluate_candidate(cand, jd_text, jd_skills, jd_yoe, jd_domains, weights)
            if res.score.overall_score >= min_score:
                evaluated.append(res)

        # Sort descending by overall score
        evaluated.sort(key=lambda x: x.score.overall_score, reverse=True)

        # Assign ranks
        for i, item in enumerate(evaluated):
            item.rank = i + 1

        total = len(evaluated)
        strong = sum(1 for r in evaluated if r.score.overall_score >= 65.0)
        avg = float(sum(r.score.overall_score for r in evaluated) / total) if total > 0 else 0.0
        top = evaluated[0].score.overall_score if total > 0 else 0.0

        summary = RankingSummary(
            total_candidates=total,
            strong_matches=strong,
            average_score=round(avg, 1),
            top_score=round(top, 1),
        )

        return RankingResponse(summary=summary, results=evaluated)
