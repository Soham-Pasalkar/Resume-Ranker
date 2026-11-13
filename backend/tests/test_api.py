from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data


def test_job_analysis(sample_jd_text):
    response = client.post("/api/v1/jobs/analyze", json={"text": sample_jd_text})
    assert response.status_code == 200
    data = response.json()
    assert "python" in data["skills_detected"]
    assert data["required_yoe"] >= 5.0
    assert "fintech" in data["domains_detected"] or "finance" in data["domains_detected"]


def test_resume_upload_and_ranking(sample_jd_text, sample_senior_resume_text):
    # Clear store
    client.delete("/api/v1/resumes")

    # Upload resume as text file
    files = [
        ("files", ("alex_rivera.txt", sample_senior_resume_text.encode("utf-8"), "text/plain"))
    ]
    upload_res = client.post("/api/v1/resumes/upload", files=files)
    assert upload_res.status_code == 200
    candidates = upload_res.json()
    assert len(candidates) == 1
    assert candidates[0]["name"] == "Alex Rivera"

    # Rank against JD
    rank_res = client.post("/api/v1/ranking/rank", json={"jd_text": sample_jd_text})
    assert rank_res.status_code == 200
    result_data = rank_res.json()
    assert result_data["summary"]["total_candidates"] == 1
    assert result_data["results"][0]["score"]["overall_score"] > 60.0
