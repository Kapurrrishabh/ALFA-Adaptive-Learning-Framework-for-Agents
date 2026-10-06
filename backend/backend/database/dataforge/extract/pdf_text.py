"""PDF to text. Used for the OpenStax textbooks."""

from pypdf import PdfReader

# Textbook pages carry a running header and footer. Two lines from each end drops most of them
# without eating body text.
_TRIM_LINES = 2


def extract(pdf_path):
    """Returns the whole book as text, or raises if the PDF cannot be read at all."""
    reader = PdfReader(str(pdf_path))
    pages = []
    for page in reader.pages:
        text = (page.extract_text() or "").strip()
        if not text:
            continue
        lines = text.splitlines()
        if len(lines) > 2 * _TRIM_LINES:
            lines = lines[_TRIM_LINES:-_TRIM_LINES]
        pages.append("\n".join(lines))
    return "\n\n".join(pages)
