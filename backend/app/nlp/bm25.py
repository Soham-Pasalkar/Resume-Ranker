import math
from typing import List, Set
from collections import Counter
from backend.app.nlp.tokenizer import clean_and_tokenize


class BM25LexicalMatcher:
    """
    In-memory BM25 lexical relevance calculator.
    Measures how strongly key JD terminology appears across a resume.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b

    def compute_lexical_score(self, query_tokens: List[str], doc_tokens: List[str]) -> float:
        if not query_tokens or not doc_tokens:
            return 0.0

        doc_len = len(doc_tokens)
        doc_freq = Counter(doc_tokens)
        query_set = set(query_tokens)

        score = 0.0
        for q in query_set:
            tf = doc_freq.get(q, 0)
            if tf > 0:
                # BM25 term weighting
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / 350.0))
                score += numerator / max(1e-6, denominator)

        # Normalize relative to query length
        max_possible = len(query_set) * (self.k1 + 1)
        normalized = score / max(1.0, max_possible)
        return min(1.0, normalized)


bm25_matcher = BM25LexicalMatcher()
