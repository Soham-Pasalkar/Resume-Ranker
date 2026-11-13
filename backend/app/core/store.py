from typing import Dict, List, Optional
from backend.app.schemas.candidate import CandidateProfile


class InMemoryStore:
    """Simple in-memory candidate store for active sessions."""

    def __init__(self):
        self._candidates: Dict[str, CandidateProfile] = {}
        self._active_jd_text: str = ""

    def save_candidate(self, candidate: CandidateProfile) -> None:
        self._candidates[candidate.id] = candidate

    def get_candidate(self, candidate_id: str) -> Optional[CandidateProfile]:
        return self._candidates.get(candidate_id)

    def get_all_candidates(self) -> List[CandidateProfile]:
        return list(self._candidates.values())

    def clear(self) -> None:
        self._candidates.clear()
        self._active_jd_text = ""

    @property
    def active_jd(self) -> str:
        return self._active_jd_text

    @active_jd.setter
    def active_jd(self, text: str) -> None:
        self._active_jd_text = text


store = InMemoryStore()
