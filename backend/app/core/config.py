from pydantic import BaseModel
from typing import List


class Settings(BaseModel):
    PROJECT_NAME: str = "People Lens API"
    VERSION: str = "1.2.0"
    API_V1_STR: str = "/api/v1"
    
    # CORS Origins (Allow local Next.js frontend)
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
    ]
    
    # Default Ranking Weights (Must sum to 1.0)
    DEFAULT_WEIGHT_SEMANTIC: float = 0.40
    DEFAULT_WEIGHT_SKILLS: float = 0.35
    DEFAULT_WEIGHT_EXPERIENCE: float = 0.15
    DEFAULT_WEIGHT_DOMAIN: float = 0.10


settings = Settings()
