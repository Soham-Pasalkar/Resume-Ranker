from pydantic import BaseModel, Field
from typing import List, Set, Optional


class JobDescriptionRequest(BaseModel):
    title: Optional[str] = Field(default=None, description="Job title, e.g. Senior Software Engineer")
    text: str = Field(..., min_length=20, description="Full raw text of the Job Description")
    target_yoe: Optional[float] = Field(default=None, description="Optional override for required years of experience")
    mandatory_skills: Optional[List[str]] = Field(default_factory=list, description="Explicit must-have skills")


class JobDescriptionAnalysis(BaseModel):
    title: Optional[str] = None
    skills_detected: List[str]
    required_yoe: float
    domains_detected: List[str]
    top_keywords: List[str]
    word_count: int
