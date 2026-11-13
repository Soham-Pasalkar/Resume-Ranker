from abc import ABC, abstractmethod


class BaseParser(ABC):
    """Abstract interface for all document parsers."""

    @abstractmethod
    def parse(self, file_bytes: bytes, filename: str = "") -> str:
        """Extract clean plain text from raw file bytes."""
        pass
