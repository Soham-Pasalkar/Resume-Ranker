from typing import List, Set, Dict, Optional
from backend.app.schemas.candidate import CandidateProfile
from backend.app.schemas.ranking import (
    WeightProfile,
    ScoreBreakdown,
    CandidateEvidence,
    RankedCandidate,
    RankingSummary,
    RankingResponse,
    KnockoutRules,
    KnockoutStatus,
)
from backend.app.nlp.embeddings import semantic_engine
from backend.app.nlp.bm25 import bm25_matcher
from backend.app.nlp.tokenizer import clean_and_tokenize
from backend.app.nlp.fusion import build_section_weighted_text, compute_reciprocal_rank_fusion
from backend.app.engine.evidence import extract_evidence_citations, generate_executive_rationale
from backend.app.engine.interview import generate_interview_questions
from backend.app.engine.constraints import evaluate_knockouts, PRESET_PROFILES


def calculate_match_label(score: float, is_knocked_out: bool = False) -> str:
    if is_knocked_out:
        return "Knocked Out"
    if score >= 65.0:
        return "Strong Match"
    elif score >= 40.0:
        return "Moderate Match"
    return "Weak Match"


def compute_skill_overlap(jd_skills: Set[str], candidate_skills: Set[str]) -> float:
    if not jd_skills:
        return 0.8
    matched = jd_skills & candidate_skills
    return len(matched) / len(jd_skills)


def compute_experience_fit(jd_yoe: float, candidate_yoe: float) -> float:
    if jd_yoe <= 0:
        return 0.8
    if candidate_yoe >= jd_yoe:
        if candidate_yoe > jd_yoe * 2.5:
            penalty = (candidate_yoe - jd_yoe * 2.5) * 0.02
            return max(0.7, 1.0 - penalty)
        return 1.0
    return max(0.1, candidate_yoe / jd_yoe)


def compute_domain_fit(jd_domains: List[str], candidate_domains: List[str]) -> float:
    if not jd_domains:
        return 0.75
    overlap = len(set(jd_domains) & set(candidate_domains))
    return min(1.0, overlap / max(1, len(jd_domains)))


class RankingEngine:
    """Production ranking engine orchestrating hybrid scoring, fusion, and decision rationales."""

    @classmethod
    def rank_candidates(
        cls,
        candidates: List[CandidateProfile],
        jd_text: str,
        jd_skills: Set[str],
        jd_yoe: float,
        jd_domains: List[str],
        weights: Optional[WeightProfile] = None,
        preset_profile: Optional[str] = "balanced",
        knockout_rules: Optional[KnockoutRules] = None,
        min_score: float = 0.0,
    ) -> RankingResponse:
        # Resolve weights (preset profile or custom weights)
        if weights is not None:
            active_weights = weights.normalized()
        elif preset_profile and preset_profile.lower() in PRESET_PROFILES:
            active_weights = PRESET_PROFILES[preset_profile.lower()].normalized()
        else:
            active_weights = PRESET_PROFILES["balanced"].normalized()

        query_tokens = clean_and_tokenize(jd_text)

        # First pass: collect dense & sparse scores for Reciprocal Rank Fusion
        dense_scores = []
        sparse_scores = []
        parsed_cache = {}

        for cand in candidates:
            # Build section-weighted text prioritizing professional experience
            weighted_text = build_section_weighted_text(cand.sections, cand.raw_text)
            cosine_sim = semantic_engine.compute_similarity(jd_text, weighted_text)
            doc_tokens = clean_and_tokenize(weighted_text)
            bm25_sim = bm25_matcher.compute_lexical_score(query_tokens, doc_tokens)

            dense_scores.append((cand.id, cosine_sim))
            sparse_scores.append((cand.id, bm25_sim))
            parsed_cache[cand.id] = {
                "cosine": cosine_sim,
                "bm25": bm25_sim,
                "weighted_text": weighted_text,
            }

        # Calculate RRF scores
        rrf_map = compute_reciprocal_rank_fusion(dense_scores, sparse_scores)

        evaluated: List[RankedCandidate] = []
        knockouts_count = 0

        for cand in candidates:
            cache = parsed_cache[cand.id]
            cosine_sim = cache["cosine"]
            bm25_sim = cache["bm25"]
            rrf_score = rrf_map.get(cand.id, 0.0)

            # Combined semantic score (75% cosine, 25% BM25)
            semantic_score = (0.75 * cosine_sim + 0.25 * bm25_sim) * 100.0

            # Skill overlap
            candidate_skills = set(cand.detected_skills)
            skill_score = compute_skill_overlap(jd_skills, candidate_skills) * 100.0

            # Experience & Domain
            exp_score = compute_experience_fit(jd_yoe, cand.detected_yoe) * 100.0
            domain_score = compute_domain_fit(jd_domains, cand.detected_domains) * 100.0

            # Base overall composite
            overall = (
                active_weights.semantic * semantic_score +
                active_weights.skills * skill_score +
                active_weights.experience * exp_score +
                active_weights.domain * domain_score
            )

            # Evaluate hard knockout constraints
            knockout_status = evaluate_knockouts(cand, knockout_rules)
            if knockout_status.is_knocked_out:
                knockouts_count += 1
                if knockout_rules and knockout_rules.apply_penalty_instead:
                    overall = max(0.0, overall - 35.0)
                else:
                    overall = min(overall, 20.0)  # Demote to bottom of leaderboard

            matched_skills = sorted(list(jd_skills & candidate_skills))
            missing_skills = sorted(list(jd_skills - candidate_skills))

            # Experience note
            if jd_yoe > 0:
                if cand.detected_yoe >= jd_yoe:
                    exp_note = f"Meets requirement: has {cand.detected_yoe:.0f} yrs (Target: {jd_yoe:.0f} yrs)"
                else:
                    exp_note = f"Short by {jd_yoe - cand.detected_yoe:.0f} yr(s): has {cand.detected_yoe:.0f} yrs (Target: {jd_yoe:.0f} yrs)"
            else:
                exp_note = f"Detected {cand.detected_yoe:.0f} yrs experience"

            # Domain note
            if cand.detected_domains:
                domain_note = f"Verified alignment in: {', '.join(cand.detected_domains).title()}"
            else:
                domain_note = "No direct industry domain keyword overlap detected"

            citations = extract_evidence_citations(cand.raw_text, set(matched_skills), cand.sections)
            interview_qs = generate_interview_questions(
                set(missing_skills),
                set(matched_skills),
                jd_yoe,
                cand.detected_yoe
            )
            rationale = generate_executive_rationale(
                candidate_name=cand.name,
                overall_score=overall,
                matched_skills=matched_skills,
                missing_skills=missing_skills,
                candidate_yoe=cand.detected_yoe,
                jd_yoe=jd_yoe,
                highest_degree=cand.highest_degree
            )

            breakdown = ScoreBreakdown(
                overall_score=round(overall, 1),
                semantic_score=round(semantic_score, 1),
                skill_score=round(skill_score, 1),
                experience_score=round(exp_score, 1),
                domain_score=round(domain_score, 1),
                rrf_score=rrf_score,
                match_level=calculate_match_label(overall, knockout_status.is_knocked_out),
            )

            evidence = CandidateEvidence(
                matched_skills=matched_skills,
                missing_skills=missing_skills,
                experience_note=exp_note,
                domain_note=domain_note,
                rationale=rationale,
                key_citations=citations,
                interview_questions=interview_qs,
            )

            if overall >= min_score:
                evaluated.append(RankedCandidate(
                    rank=0,
                    candidate=cand,
                    score=breakdown,
                    evidence=evidence,
                    knockout=knockout_status,
                ))

        # Sort: Non-knockouts first by score descending, then knockouts
        evaluated.sort(
            key=lambda x: (not x.knockout.is_knocked_out, x.score.overall_score),
            reverse=True
        )

        for i, item in enumerate(evaluated):
            item.rank = i + 1

        total = len(evaluated)
        strong = sum(1 for r in evaluated if r.score.overall_score >= 65.0 and not r.knockout.is_knocked_out)
        avg = float(sum(r.score.overall_score for r in evaluated) / total) if total > 0 else 0.0
        top = evaluated[0].score.overall_score if total > 0 else 0.0

        summary = RankingSummary(
            total_candidates=total,
            strong_matches=strong,
            average_score=round(avg, 1),
            top_score=round(top, 1),
            knockouts_count=knockouts_count,
        )

        return RankingResponse(summary=summary, results=evaluated)
