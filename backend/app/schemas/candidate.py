from pydantic import BaseModel, Field
from typing import Dict, List, Optional


class ContactInfo(BaseModel):
    email: Optional[str] = None
    phone: Optional[str] = None
    linkedin: Optional[str] = None
    github: Optional[str] = None
    location: Optional[str] = None


class CandidateProfile(BaseModel):
    id: str = Field(..., description="Unique ID for candidate")
    filename: str
    name: str = Field(..., description="Extracted candidate name (or sanitized code)")
    contact: ContactInfo = Field(default_factory=ContactInfo)
    detected_yoe: float = 0.0
    detected_skills: List[str] = Field(default_factory=list)
    detected_domains: List[str] = Field(default_factory=list)
    sections: Dict[str, str] = Field(default_factory=dict, description="Identified sections, e.g. experience, education")
    raw_text: str = ""
    is_anonymized: bool = False
