from backend.app.parsers.base import BaseParser


class TextParser(BaseParser):
    """Plain text parser supporting UTF-8 and latin-1 fallback."""

    def parse(self, file_bytes: bytes, filename: str = "") -> str:
        for encoding in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
            try:
                return file_bytes.decode(encoding).strip()
            except UnicodeDecodeError:
                continue
        return file_bytes.decode("utf-8", errors="ignore").strip()
