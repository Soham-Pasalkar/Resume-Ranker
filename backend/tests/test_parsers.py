from backend.app.parsers.text_parser import TextParser
from backend.app.parsers.factory import DocumentParserFactory
from backend.app.parsers.pdf_parser import PDFParser
from backend.app.parsers.docx_parser import DOCXParser


def test_text_parser():
    parser = TextParser()
    raw = b"Hello world, this is a plain text resume."
    parsed = parser.parse(raw, "resume.txt")
    assert "Hello world" in parsed


def test_parser_factory_resolution():
    assert isinstance(DocumentParserFactory.get_parser("resume.pdf"), PDFParser)
    assert isinstance(DocumentParserFactory.get_parser("cv.docx"), DOCXParser)
    assert isinstance(DocumentParserFactory.get_parser("notes.txt"), TextParser)
