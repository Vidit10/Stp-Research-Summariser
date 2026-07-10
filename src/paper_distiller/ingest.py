from __future__ import annotations

import logging
import re
from pathlib import Path

from paper_distiller.document import FigureRef, Page, PaperDocument

LOGGER = logging.getLogger(__name__)

VISUAL_REF_PATTERN = re.compile(
    r"\b(?P<kind>Figure|Fig\.|Table|Equation|Eq\.)\s*(?P<label>[A-Za-z]?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)


class IngestionError(RuntimeError):
    pass


def load_document(path: Path) -> PaperDocument:
    path = path.expanduser().resolve()
    if not path.exists():
        raise IngestionError(f"Input file does not exist: {path}")
    if path.suffix.lower() == ".pdf":
        pages = _load_pdf(path)
    elif path.suffix.lower() in {".txt", ".md"}:
        pages = [Page(number=1, text=path.read_text(encoding="utf-8", errors="replace"))]
    else:
        raise IngestionError(
            f"Unsupported input format '{path.suffix}'. Use PDF, TXT, or Markdown."
        )

    visual_refs = detect_visual_references(pages)
    return PaperDocument(source_path=path, pages=pages, visual_refs=visual_refs)


def _load_pdf(path: Path) -> list[Page]:
    errors: list[str] = []

    try:
        import pdfplumber  # type: ignore

        pages: list[Page] = []
        with pdfplumber.open(path) as pdf:
            for index, page in enumerate(pdf.pages, start=1):
                text = page.extract_text(x_tolerance=1, y_tolerance=3) or ""
                pages.append(Page(number=index, text=text))
        if any(page.text.strip() for page in pages):
            return pages
    except ImportError:
        errors.append("pdfplumber is not installed")
    except Exception as exc:  # pragma: no cover - extractor-specific failure
        LOGGER.warning("pdfplumber extraction failed: %s", exc)
        errors.append(f"pdfplumber failed: {exc}")

    try:
        from pypdf import PdfReader  # type: ignore

        reader = PdfReader(str(path))
        pages = []
        for index, page in enumerate(reader.pages, start=1):
            pages.append(Page(number=index, text=page.extract_text() or ""))
        if any(page.text.strip() for page in pages):
            return pages
    except ImportError:
        errors.append("pypdf is not installed")
    except Exception as exc:  # pragma: no cover - extractor-specific failure
        LOGGER.warning("pypdf extraction failed: %s", exc)
        errors.append(f"pypdf failed: {exc}")

    raise IngestionError(
        "Could not extract text from PDF. Install optional PDF dependencies with "
        "`python -m pip install -e .[pdf]`. Details: " + "; ".join(errors)
    )


def detect_visual_references(pages: list[Page]) -> list[FigureRef]:
    refs: list[FigureRef] = []
    seen: set[tuple[str, str, int | None]] = set()
    for page in pages:
        for match in VISUAL_REF_PATTERN.finditer(page.text):
            kind = match.group("kind").replace(".", "")
            label = match.group("label")
            key = (kind.lower(), label.lower(), page.number)
            if key in seen:
                continue
            seen.add(key)
            start = max(0, match.start() - 160)
            end = min(len(page.text), match.end() + 260)
            context = " ".join(page.text[start:end].split())
            refs.append(FigureRef(kind=kind.title(), label=label, page=page.number, context=context))
    return refs

