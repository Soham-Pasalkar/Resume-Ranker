import uuid
from typing import List
from pydantic import BaseModel
from fastapi import APIRouter, UploadFile, File, Query, HTTPException
from backend.app.schemas.candidate import CandidateProfile
from backend.app.parsers.factory import DocumentParserFactory
from backend.app.extractors.contact import extract_contact_info
from backend.app.extractors.name import extract_candidate_name
from backend.app.extractors.experience import extract_years_of_experience
from backend.app.extractors.sections import segment_resume_sections
from backend.app.extractors.anonymizer import anonymize_resume
from backend.app.extractors.education import extract_education_info
from backend.app.nlp.taxonomy import extract_skills_from_text, extract_domains_from_text
from backend.app.core.store import store

router = APIRouter(prefix="/resumes", tags=["Resumes"])


class StatusUpdateRequest(BaseModel):
    status: str


@router.post("/upload", response_model=List[CandidateProfile])
async def upload_resumes(
    files: List[UploadFile] = File(...),
    anonymize: bool = Query(default=False, description="Enable blind hiring PII redaction")
):
    """
    Batch upload candidate resumes (PDF, DOCX, TXT).
    Parses document structures, extracts candidate metadata, education credentials,
    and indexes skills.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    processed_candidates: List[CandidateProfile] = []

    for file in files:
        filename = file.filename or "resume.txt"
        content_bytes = await file.read()
        if not content_bytes:
            continue

        raw_text = DocumentParserFactory.extract_text(content_bytes, filename)
        if not raw_text.strip():
            continue

        candidate_name = extract_candidate_name(raw_text, filename)
        contact_info = extract_contact_info(raw_text)
        yoe = extract_years_of_experience(raw_text)
        skills = extract_skills_from_text(raw_text)
        domains = extract_domains_from_text(raw_text)
        sections = segment_resume_sections(raw_text)
        edu_info = extract_education_info(raw_text, sections.get("education", ""))

        display_name = candidate_name
        text_to_save = raw_text

        if anonymize:
            display_name = f"Candidate-{str(uuid.uuid4())[:8].upper()}"
            text_to_save = anonymize_resume(
                raw_text,
                candidate_name=candidate_name,
                institution=edu_info["institution"] or "",
                graduation_year=edu_info["graduation_year"]
            )

        profile = CandidateProfile(
            id=str(uuid.uuid4()),
            filename=filename,
            name=display_name,
            contact=contact_info,
            detected_yoe=yoe,
            detected_skills=sorted(list(skills)),
            detected_domains=domains,
            highest_degree=edu_info["highest_degree"],
            institution=edu_info["institution"],
            graduation_year=edu_info["graduation_year"],
            sections=sections,
            raw_text=text_to_save,
            is_anonymized=anonymize,
            recruitment_status="Applied",
        )

        store.save_candidate(profile)
        processed_candidates.append(profile)

    return processed_candidates


@router.get("", response_model=List[CandidateProfile])
async def list_candidates():
    """Retrieve all currently queued candidates in memory."""
    return store.get_all_candidates()


@router.patch("/{candidate_id}/status", response_model=CandidateProfile)
async def update_candidate_status(candidate_id: str, payload: StatusUpdateRequest):
    """
    Update a candidate's recruitment pipeline status:
    Applied, Shortlisted, Interview, Selected, Rejected.
    """
    valid_statuses = {"Applied", "Shortlisted", "Interview", "Selected", "Rejected"}
    if payload.status not in valid_statuses:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid status '{payload.status}'. Must be one of: {sorted(list(valid_statuses))}"
        )
    candidate = store.update_candidate_status(candidate_id, payload.status)
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found.")
    return candidate


@router.delete("")
async def clear_candidates():
    """Clear candidate queue."""
    store.clear()
    return {"message": "All candidates cleared successfully."}
