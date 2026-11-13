import io
import logging
from backend.app.parsers.base import BaseParser

logger = logging.getLogger(__name__)

class DOCXParser(BaseParser):
    """DOCX text extractor utilizing python-docx."""

    def parse(self, file_bytes: bytes, filename: str = "") -> str:
        paragraphs = []
        try:
            import docx
            doc = docx.Document(io.BytesIO(file_bytes))
            
            # Extract header paragraphs
            for p in doc.paragraphs:
                txt = p.text.strip()
                if txt:
                    paragraphs.append(txt)
                    
            # Extract text from tables (often used in resumes for skill matrices)
            for table in doc.tables:
                for row in table.rows:
                    row_cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                    if row_cells:
                        paragraphs.append(" | ".join(row_cells))
        except Exception as e:
            logger.error("Failed to parse DOCX %s: %s", filename, e)

        return "\n".join(paragraphs).strip()
