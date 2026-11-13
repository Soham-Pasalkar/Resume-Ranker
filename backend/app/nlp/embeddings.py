import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class SemanticSimilarityEngine:
    """Computes semantic textual similarity between JD and Candidate Resume."""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            max_features=10000,
            sublinear_tf=True
        )

    def compute_similarity(self, jd_text: str, resume_text: str) -> float:
        """
        Calculates cosine similarity between Job Description and Resume.
        Returns a score in range [0.0, 1.0].
        """
        if not jd_text.strip() or not resume_text.strip():
            return 0.0

        try:
            matrix = self.vectorizer.fit_transform([jd_text, resume_text])
            sim = cosine_similarity(matrix[0:1], matrix[1:2])[0][0]
            return float(np.clip(sim, 0.0, 1.0))
        except Exception:
            return 0.0


# Shared singleton instance
semantic_engine = SemanticSimilarityEngine()
