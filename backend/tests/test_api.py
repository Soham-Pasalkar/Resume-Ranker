from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data


def test_weight_profiles_endpoint():
    response = client.get("/api/v1/ranking/profiles")
    assert response.status_code == 200
    data = response.json()
    assert "balanced" in data
    assert "specialist" in data
    assert "leadership" in data
    assert "generalist" in data


def test_job_analysis(sample_jd_text):
    response = client.post("/api/v1/jobs/analyze", json={"text": sample_jd_text})
    assert response.status_code == 200
    data = response.json()
    assert "python" in data["skills_detected"]
    assert data["required_yoe"] >= 5.0
    assert "fintech" in data["domains_detected"] or "finance" in data["domains_detected"]


def test_resume_upload_and_ranking_with_profiles(sample_jd_text, sample_senior_resume_text):
    client.delete("/api/v1/resumes")

    files = [
        ("files", ("alex_rivera.txt", sample_senior_resume_text.encode("utf-8"), "text/plain"))
    ]
    upload_res = client.post("/api/v1/resumes/upload", files=files)
    assert upload_res.status_code == 200
    candidates = upload_res.json()
    assert len(candidates) == 1
    assert candidates[0]["name"] == "Alex Rivera"
    assert candidates[0]["highest_degree"] == "Bachelor's"
    assert candidates[0]["recruitment_status"] == "Applied"

    cand_id = candidates[0]["id"]

    # Test status update
    status_res = client.patch(f"/api/v1/resumes/{cand_id}/status", json={"status": "Interview"})
    assert status_res.status_code == 200
    assert status_res.json()["recruitment_status"] == "Interview"

    # Rank with specialist preset
    rank_res = client.post(
        "/api/v1/ranking/rank",
        json={
            "jd_text": sample_jd_text,
            "preset_profile": "specialist",
            "knockout_rules": {
                "mandatory_skills": ["python"],
                "min_yoe": 4.0
            }
        }
    )
    assert rank_res.status_code == 200
    result_data = rank_res.json()
    assert result_data["summary"]["total_candidates"] == 1
    assert result_data["results"][0]["score"]["overall_score"] > 60.0
    assert result_data["results"][0]["knockout"]["is_knocked_out"] is False
    assert result_data["results"][0]["recruitment_status"] == "Interview"
    assert result_data["summary"]["stage_counts"]["Interview"] == 1


def test_export_csv_endpoint(sample_jd_text, sample_senior_resume_text):
    client.delete("/api/v1/resumes")

    files = [
        ("files", ("alex_rivera.txt", sample_senior_resume_text.encode("utf-8"), "text/plain"))
    ]
    client.post("/api/v1/resumes/upload", files=files)

    res = client.post(
        "/api/v1/ranking/export/csv",
        json={"jd_text": sample_jd_text}
    )
    assert res.status_code == 200
    assert res.headers["content-type"] == "text/csv; charset=utf-8"
    csv_text = res.text
    assert "Rank,Candidate Name,Recruitment Status,Overall Score" in csv_text
    assert "Alex Rivera" in csv_text
