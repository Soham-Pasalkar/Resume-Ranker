from typing import Dict, List, Tuple
from collections import defaultdict


def build_section_weighted_text(sections: Dict[str, str], raw_text: str) -> str:
    """
    Construct an enriched text representation weighting hands-on professional
    experience higher than static keyword lists.
    """
    if not sections:
        return raw_text

    parts: List[str] = []

    # Experience section receives 2x representation weight
    if "experience" in sections:
        parts.append(sections["experience"])
        parts.append(sections["experience"])

    # Projects section receives 1.2x
    if "projects" in sections:
        parts.append(sections["projects"])

    # Skills section
    if "skills" in sections:
        parts.append(sections["skills"])

    # Summary section
    if "summary" in sections:
        parts.append(sections["summary"])

    # Education section
    if "education" in sections:
        parts.append(sections["education"])

    if not parts:
        return raw_text

    return "\n\n".join(parts)


def compute_reciprocal_rank_fusion(
    dense_scores: List[Tuple[str, float]],
    sparse_scores: List[Tuple[str, float]],
    k: int = 60
) -> Dict[str, float]:
    """
    Compute Reciprocal Rank Fusion (RRF) across Dense Semantic and Sparse BM25 ranks.
    RRF(d) = sum(1 / (k + rank_i(d)))
    Returns a dictionary mapping candidate_id -> normalized RRF score [0, 100].
    """
    # Sort descending to compute ranks (1-based)
    dense_sorted = sorted(dense_scores, key=lambda x: x[1], reverse=True)
    sparse_sorted = sorted(sparse_scores, key=lambda x: x[1], reverse=True)

    dense_ranks = {cid: rank + 1 for rank, (cid, _) in enumerate(dense_sorted)}
    sparse_ranks = {cid: rank + 1 for rank, (cid, _) in enumerate(sparse_sorted)}

    all_ids = set(dense_ranks.keys()) | set(sparse_ranks.keys())
    rrf_raw: Dict[str, float] = {}

    for cid in all_ids:
        r_dense = dense_ranks.get(cid, len(dense_sorted) + 1)
        r_sparse = sparse_ranks.get(cid, len(sparse_sorted) + 1)
        score = (1.0 / (k + r_dense)) + (1.0 / (k + r_sparse))
        rrf_raw[cid] = score

    if not rrf_raw:
        return {}

    # Normalize RRF scores between min and max to a 0-100 scale
    max_val = max(rrf_raw.values())
    min_val = min(rrf_raw.values())
    spread = max_val - min_val if max_val > min_val else 1.0

    return {
        cid: round(((val - min_val) / spread) * 100.0, 1)
        for cid, val in rrf_raw.items()
    }
