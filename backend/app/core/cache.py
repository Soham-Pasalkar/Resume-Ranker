import hashlib
from typing import Dict, Any, Optional


class PerformanceCache:
    """
    In-memory SHA-256 caching layer to prevent redundant re-tokenization,
    vector embedding computations, and section segmentation.
    """

    def __init__(self, max_entries: int = 1000):
        self._max_entries = max_entries
        self._cache: Dict[str, Any] = {}

    @staticmethod
    def _compute_hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def get(self, key_text: str, namespace: str = "general") -> Optional[Any]:
        cache_key = f"{namespace}:{self._compute_hash(key_text)}"
        return self._cache.get(cache_key)

    def set(self, key_text: str, value: Any, namespace: str = "general") -> None:
        if len(self._cache) >= self._max_entries:
            # Evict oldest 20%
            keys_to_remove = list(self._cache.keys())[: int(self._max_entries * 0.2)]
            for k in keys_to_remove:
                self._cache.pop(k, None)

        cache_key = f"{namespace}:{self._compute_hash(key_text)}"
        self._cache[cache_key] = value

    def clear(self) -> None:
        self._cache.clear()


cache = PerformanceCache()
