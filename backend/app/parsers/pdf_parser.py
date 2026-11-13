import io
import logging
from backend.app.parsers.base import BaseParser

logger = logging.getLogger(__name__)

class PDFParser(BaseParser):
    """Layout-aware PDF text extractor utilizing pdfplumber."""

    def parse(self, file_bytes: bytes, filename: str = "") -> str:
        text_parts = []
        try:
            import pdfplumber
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                for page_idx, page in enumerate(pdf.pages):
                    try:
                        # Extract layout-aware words & text
                        extracted = page.extract_text(layout=True)
                        if not extracted:
                            # Fallback to standard flow extraction
                            extracted = page.extract_text()
                        
                        if extracted and extracted.strip():
                            text_parts.append(extracted.strip())
                    except Exception as page_err:
                        logger.warning("Error parsing page %d of %s: %s", page_idx + 1, filename, page_err)
        except Exception as e:
            logger.error("Failed to parse PDF %s: %s", filename, e)

        # Fallback if pdfplumber returned empty (e.g. pypdfium2 if available)
        if not text_parts:
            try:
                import pypdfium2
                pdf = pypdfium2.PdfDocument(io.BytesIO(file_bytes))
                for page in pdf:
                    textpage = page.get_textpage()
                    raw = textpage.get_text_range()
                    if raw and raw.strip():
                        text_parts.append(raw.strip())
            except Exception as e2:
                logger.debug("pypdfium2 fallback also failed for %s: %s", filename, e2)

        return "\n\n".join(text_parts).strip()
