from backend.app.parsers.base import BaseParser
from backend.app.parsers.pdf_parser import PDFParser
from backend.app.parsers.docx_parser import DOCXParser
from backend.app.parsers.text_parser import TextParser


class DocumentParserFactory:
    """Factory to retrieve the appropriate parser based on file extension."""

    _pdf_parser = PDFParser()
    _docx_parser = DOCXParser()
    _text_parser = TextParser()

    @classmethod
    def get_parser(cls, filename: str) -> BaseParser:
        lower = filename.lower()
        if lower.endswith(".pdf"):
            return cls._pdf_parser
        elif lower.endswith(".docx") or lower.endswith(".doc"):
            return cls._docx_parser
        else:
            return cls._text_parser

    @classmethod
    def extract_text(cls, file_bytes: bytes, filename: str) -> str:
        parser = cls.get_parser(filename)
        return parser.parse(file_bytes, filename)
