from backend.app.schemas.candidate import CandidateProfile
from backend.app.schemas.ranking import WeightProfile, KnockoutRules
from backend.app.engine.scoring import RankingEngine
from backend.app.nlp.taxonomy import extract_skills_from_text, extract_domains_from_text
from backend.app.extractors.experience import extract_years_of_experience


def test_ranking_senior_vs_junior(sample_jd_text, sample_senior_resume_text, sample_junior_resume_text):
    jd_skills = extract_skills_from_text(sample_jd_text)
    jd_yoe = extract_years_of_experience(sample_jd_text)
    jd_domains = extract_domains_from_text(sample_jd_text)

    senior_cand = CandidateProfile(
        id="cand-1",
        filename="alex_rivera.pdf",
        name="Alex Rivera",
        detected_yoe=extract_years_of_experience(sample_senior_resume_text),
        detected_skills=list(extract_skills_from_text(sample_senior_resume_text)),
        detected_domains=extract_domains_from_text(sample_senior_resume_text),
        highest_degree="Bachelor's",
        raw_text=sample_senior_resume_text,
    )

    junior_cand = CandidateProfile(
        id="cand-2",
        filename="jordan_lee.pdf",
        name="Jordan Lee",
        detected_yoe=extract_years_of_experience(sample_junior_resume_text),
        detected_skills=list(extract_skills_from_text(sample_junior_resume_text)),
        detected_domains=extract_domains_from_text(sample_junior_resume_text),
        highest_degree=None,
        raw_text=sample_junior_resume_text,
    )

    response = RankingEngine.rank_candidates(
        candidates=[junior_cand, senior_cand],
        jd_text=sample_jd_text,
        jd_skills=jd_skills,
        jd_yoe=jd_yoe,
        jd_domains=jd_domains,
        preset_profile="balanced",
    )

    assert len(response.results) == 2
    # Senior should rank #1
    assert response.results[0].candidate.id == "cand-1"
    assert response.results[0].score.overall_score > response.results[1].score.overall_score
    assert response.results[0].score.overall_score >= 65.0
    assert response.results[0].score.rrf_score is not None
    assert response.results[0].evidence.rationale != ""
    assert len(response.results[0].evidence.matched_skills) > 4
    assert len(response.results[0].evidence.interview_questions) > 0


def test_knockout_constraints(sample_jd_text, sample_senior_resume_text, sample_junior_resume_text):
    jd_skills = extract_skills_from_text(sample_jd_text)
    jd_yoe = extract_years_of_experience(sample_jd_text)
    jd_domains = extract_domains_from_text(sample_jd_text)

    cand = CandidateProfile(
        id="cand-2",
        filename="jordan_lee.pdf",
        name="Jordan Lee",
        detected_yoe=1.0,
        detected_skills=["html", "css", "javascript"],
        highest_degree=None,
        raw_text=sample_junior_resume_text,
    )

    # Mandatory skill rule: requires python and kubernetes
    rules = KnockoutRules(
        mandatory_skills=["python", "kubernetes"],
        min_yoe=3.0,
    )

    response = RankingEngine.rank_candidates(
        candidates=[cand],
        jd_text=sample_jd_text,
        jd_skills=jd_skills,
        jd_yoe=jd_yoe,
        jd_domains=jd_domains,
        knockout_rules=rules,
    )

    assert len(response.results) == 1
    assert response.results[0].knockout.is_knocked_out is True
    assert len(response.results[0].knockout.reasons) >= 1
    assert response.results[0].score.match_level == "Knocked Out"
    assert response.summary.knockouts_count == 1
