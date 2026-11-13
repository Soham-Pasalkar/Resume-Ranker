from backend.app.parsers.base import BaseParser
from backend.app.parsers.pdf_parser import PDFParser
from backend.app.parsers.docx_parser import DOCXParser
from backend.app.parsers.text_parser import TextParser
from backend.app.parsers.factory import DocumentParserFactory

__all__ = [
    "BaseParser",
    "PDFParser",
    "DOCXParser",
    "TextParser",
    "DocumentParserFactory",
]
