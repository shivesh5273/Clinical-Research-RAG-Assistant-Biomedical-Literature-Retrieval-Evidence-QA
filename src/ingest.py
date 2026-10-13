"""PDF ingestion: turn a folder of biomedical PDFs into page-level records.

Design notes
------------
* One record per *page*, never per document. Page boundaries are preserved so
  that every downstream chunk can carry an honest `page` citation.
* Extraction is deliberately conservative: we normalize obvious whitespace
  artefacts (line-wrapped hyphenation, repeated blank lines) and nothing else.
  We do not strip references, headers or footers, because guessing wrong would
  silently delete evidence.
* OCR is out of scope for the MVP. Image-only (scanned) PDFs are detected and
  reported rather than silently producing an empty index.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

# PyMuPDF >= 1.24.3 exposes the module as `pymupdf`; older releases only as
# `fitz` (which newer releases still provide but deprecate). Try the modern
# name first so we do not emit a deprecation warning on every import.
try:
    import pymupdf as fitz
except ImportError:  # pragma: no cover - older PyMuPDF
    try:
        import fitz  # type: ignore[no-redef]
    except ImportError as exc:
        raise ImportError(
            "PyMuPDF is required for ingestion. Install it with:\n"
            "    pip install PyMuPDF"
        ) from exc


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass
class DocumentStatus:
    """What happened to one PDF during ingestion."""

    document: str
    status: str  # "ok" | "no_text" | "empty" | "error"
    pages_total: int = 0
    pages_kept: int = 0
    characters: int = 0
    title: Optional[str] = None
    message: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "ok"


@dataclass
class IngestReport:
    """Summary of an ingestion run over a folder of PDFs."""

    documents: List[DocumentStatus] = field(default_factory=list)

    @property
    def n_found(self) -> int:
        return len(self.documents)

    @property
    def n_loaded(self) -> int:
        return sum(1 for d in self.documents if d.ok)

    @property
    def n_failed(self) -> int:
        return sum(1 for d in self.documents if not d.ok)

    @property
    def total_pages_kept(self) -> int:
        return sum(d.pages_kept for d in self.documents)

    def problems(self) -> List[DocumentStatus]:
        """Documents that did not ingest cleanly."""
        return [d for d in self.documents if not d.ok]

    def as_text(self) -> str:
        """Human-readable summary, used by the CLI smoke test and the app."""
        lines = [
            f"PDFs found:  {self.n_found}",
            f"PDFs loaded: {self.n_loaded}",
            f"Pages kept:  {self.total_pages_kept}",
        ]
        if self.n_failed:
            lines.append(f"PDFs with problems: {self.n_failed}")
            for doc in self.problems():
                lines.append(f"  - {doc.document}: [{doc.status}] {doc.message}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------
# Matches a word broken across a line by a hyphen, e.g. "random-\nized".
_HYPHEN_LINEBREAK = re.compile(r"(\w)-\n(\w)")
# Three or more newlines collapse to a paragraph break.
_MANY_NEWLINES = re.compile(r"\n{3,}")
# Runs of spaces/tabs.
_MANY_SPACES = re.compile(r"[ \t ]{2,}")
# Trailing whitespace on each line.
_TRAILING_WS = re.compile(r"[ \t]+\n")


def normalize_text(raw: str) -> str:
    """Apply conservative whitespace normalization to raw PDF text.

    Steps (in order):
      1. Unicode NFKC normalization (fixes ligatures such as "fi" -> "fi").
      2. Rejoin words hyphenated across a line break.
      3. Convert single newlines inside a paragraph into spaces, while keeping
         blank-line paragraph breaks intact.
      4. Collapse repeated spaces and blank lines.

    No content is removed - only whitespace is changed.
    """
    if not raw:
        return ""

    text = unicodedata.normalize("NFKC", raw)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _HYPHEN_LINEBREAK.sub(r"\1\2", text)

    # Protect paragraph breaks, unwrap single line breaks, then restore.
    text = text.replace("\n\n", "\x00PARA\x00")
    text = text.replace("\n", " ")
    text = text.replace("\x00PARA\x00", "\n\n")

    text = _MANY_SPACES.sub(" ", text)
    text = _TRAILING_WS.sub("\n", text)
    text = _MANY_NEWLINES.sub("\n\n", text)
    return text.strip()


def _count_alphanumeric(text: str) -> int:
    """Count letters/digits - a better 'is this real text?' signal than len()."""
    return sum(1 for ch in text if ch.isalnum())


# ---------------------------------------------------------------------------
# Symbol-font minus-sign repair (narrowly scoped)
# ---------------------------------------------------------------------------
# Some publisher PDFs typeset mathematical symbols in a separate subsetted
# font whose ToUnicode mapping is wrong or absent. MuPDF then decodes the
# glyph to whatever the font's encoding claims, which can be an unrelated
# Latin character.
#
# Confirmed instance (Diabetes Care 2012, PMC3308269): the font `AdvPS586B`
# renders the minus sign, and it decodes to the DIGIT "2". So
#     "-0.40% [95% CI -0.61 to -0.18]"
# extracts as
#     "20.40% [95% CI 20.61 to 20.18]"
# turning every negative result into a positive number an order of magnitude
# too large. That corrupts both retrieval and any answer quoted from it.
#
# The repair below is deliberately narrow. It rewrites a character to "-"
# only when ALL of these hold:
#   1. the decoded character is one confirmed to stand in for a minus ("2");
#   2. it came from a font whose document-wide decoded inventory is tiny
#      (<= SYMBOL_FONT_MAX_DISTINCT_CHARS) - the signature of a symbol font,
#      not a text font;
#   3. the next character is a digit rendered in a DIFFERENT font, i.e. the
#      suspect character is not part of that number's own font run;
#   4. the previous character is not a digit, so a number split across a font
#      boundary can never be corrupted. (A plain number like "120.40" is
#      already excluded by rule 3, since all of its digits share one font.)
# If the page's character stream cannot be aligned to the extracted text
# exactly, the page is left completely untouched.
#
# This fixes the minus sign only. The same font in that paper also mis-decodes
# "<" as ",", ">" as "." and the degree sign as "8"; those are left alone
# because each needs its own evidence and carries its own risk.

SYMBOL_FONT_MAX_DISTINCT_CHARS = 8
MINUS_MISREAD_AS = frozenset({"2"})

CharFontStream = List[Tuple[str, str]]


def extract_char_font_stream(page: "fitz.Page") -> CharFontStream:
    """Return `[(character, font_name), ...]` for a page, in reading order."""
    stream: CharFontStream = []
    for block in page.get_text("rawdict").get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                font = span.get("font", "")
                for char in span.get("chars", []):
                    stream.append((char["c"], font))
    return stream


def font_char_inventory(doc: "fitz.Document") -> Dict[str, Set[str]]:
    """Map each font in the document to the set of characters it decodes to.

    A font with only a handful of distinct characters across an entire paper
    is almost certainly a symbol font rather than a text font.
    """
    inventory: Dict[str, Set[str]] = {}
    for page_index in range(doc.page_count):
        try:
            page = doc.load_page(page_index)
        except Exception:
            continue
        for char, font in extract_char_font_stream(page):
            inventory.setdefault(font, set()).add(char)
    return inventory


def repair_minus_glyphs(
    stream: CharFontStream,
    text: str,
    font_inventory: Dict[str, Set[str]],
) -> Tuple[str, int]:
    """Repair symbol-font minus signs in `text`. Returns `(text, n_repaired)`.

    `stream` is the page's character/font stream and `text` the same page as
    extracted by `get_text("text")`. The two agree character-for-character once
    whitespace is ignored; that exact alignment is what makes the repair
    deterministic rather than a guess. If the alignment does not hold, the text
    is returned unchanged.
    """
    non_ws_indices = [i for i, (char, _f) in enumerate(stream) if not char.isspace()]
    non_ws = [stream[i] for i in non_ws_indices]
    text_positions = [i for i, char in enumerate(text) if not char.isspace()]

    # Safety gate: without an exact alignment we cannot know which character
    # in `text` a stream entry refers to, so we change nothing.
    if len(non_ws) != len(text_positions):
        return text, 0
    for position, (char, _font) in zip(text_positions, non_ws):
        if text[position] != char:
            return text, 0

    characters = list(text)
    repaired = 0
    for index, (char, font) in enumerate(non_ws):
        if char not in MINUS_MISREAD_AS:
            continue
        if len(font_inventory.get(font, ())) > SYMBOL_FONT_MAX_DISTINCT_CHARS:
            continue
        if index + 1 >= len(non_ws):
            continue
        next_char, next_font = non_ws[index + 1]
        if not next_char.isdigit() or next_font == font:
            continue
        # Look at the literal previous character in the stream (whitespace
        # included). Only a directly adjacent digit is dangerous: that would
        # mean splitting a number that straddles a font boundary. A preceding
        # letter is fine and must be allowed, because this same class of PDF
        # also drops inter-word spaces ("changewas-0.40%").
        stream_index = non_ws_indices[index]
        if stream_index > 0 and stream[stream_index - 1][0].isdigit():
            continue
        characters[text_positions[index]] = "-"
        repaired += 1

    return "".join(characters), repaired


# ---------------------------------------------------------------------------
# Title extraction (best effort, never invented)
# ---------------------------------------------------------------------------
def extract_title(doc: "fitz.Document", first_page_text: str) -> Optional[str]:
    """Best-effort paper title.

    Order of preference:
      1. The PDF's embedded `title` metadata, if it looks like a real title.
      2. The longest line among the first few lines of page 1.
      3. None - the caller then falls back to the filename.

    We never fabricate a title; returning None is an acceptable outcome.
    """
    meta_title = (doc.metadata or {}).get("title") or ""
    meta_title = meta_title.strip()
    # Reject junk metadata such as "untitled", "Microsoft Word - paper.doc".
    if len(meta_title) >= 12 and not meta_title.lower().startswith(
        ("untitled", "microsoft word", "doi:", "pii:")
    ):
        return meta_title

    candidates = [ln.strip() for ln in first_page_text.split("\n") if ln.strip()]
    for line in candidates[:6]:
        # A plausible title: reasonably long, not an all-caps journal banner.
        if 20 <= len(line) <= 250 and _count_alphanumeric(line) > 15:
            return line
    return None


# ---------------------------------------------------------------------------
# Single-PDF loading
# ---------------------------------------------------------------------------
def load_pdf(
    pdf_path: Path,
    min_page_chars: int = 50,
    repair_glyphs: bool = True,
) -> Tuple[List[Dict[str, object]], DocumentStatus]:
    """Extract page records from one PDF.

    Parameters
    ----------
    pdf_path:
        Path to the PDF file.
    min_page_chars:
        Pages whose normalized text has fewer alphanumeric characters than this
        are skipped (cover images, blank pages, figure-only pages).

    Returns
    -------
    (records, status)
        `records` is a list of dicts shaped like::

            {"document": "paper1.pdf", "page": 5, "text": "...", "title": "..."}

        Page numbers are 1-based, matching what a human sees in a PDF viewer.
        `status` describes how the document fared, including failures.
    """
    pdf_path = Path(pdf_path)
    name = pdf_path.name

    if not pdf_path.exists():
        return [], DocumentStatus(name, "error", message="File does not exist.")
    if pdf_path.stat().st_size == 0:
        return [], DocumentStatus(name, "empty", message="File is 0 bytes.")

    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:  # corrupted / password protected / not a PDF
        return [], DocumentStatus(
            name, "error", message=f"Could not open PDF ({type(exc).__name__}: {exc})"
        )

    try:
        if doc.needs_pass:
            return [], DocumentStatus(
                name, "error", message="PDF is password protected."
            )

        pages_total = doc.page_count
        if pages_total == 0:
            return [], DocumentStatus(name, "empty", message="PDF has 0 pages.")

        records: List[Dict[str, object]] = []
        first_page_raw = ""
        total_chars = 0
        glyphs_repaired = 0

        # Built once per document: a font's character inventory is only
        # meaningful across the whole paper, not a single page.
        inventory: Dict[str, Set[str]] = {}
        if repair_glyphs:
            try:
                inventory = font_char_inventory(doc)
            except Exception as exc:  # never let the repair break ingestion
                print(
                    f"[ingest] WARNING: {name} font inventory unavailable "
                    f"({type(exc).__name__}: {exc}) - glyph repair skipped."
                )

        for page_index in range(pages_total):
            page_number = page_index + 1
            try:
                page = doc.load_page(page_index)
                raw = page.get_text("text")
                if inventory:
                    raw, n_fixed = repair_minus_glyphs(
                        extract_char_font_stream(page), raw, inventory
                    )
                    glyphs_repaired += n_fixed
            except Exception as exc:
                # One unreadable page must not kill the whole document.
                print(
                    f"[ingest] WARNING: {name} page {page_number} unreadable "
                    f"({type(exc).__name__}: {exc}) - skipping this page."
                )
                continue

            if page_index == 0:
                first_page_raw = raw

            text = normalize_text(raw)
            if _count_alphanumeric(text) < min_page_chars:
                continue  # image-only / near-empty page

            total_chars += len(text)
            records.append(
                {
                    "document": name,
                    "page": page_number,
                    "text": text,
                }
            )

        if not records:
            return [], DocumentStatus(
                name,
                "no_text",
                pages_total=pages_total,
                message=(
                    "No extractable text. This is usually a scanned / image-only "
                    "PDF. OCR is not part of the MVP - replace it with a "
                    "text-based (born-digital) PDF."
                ),
            )

        title = extract_title(doc, normalize_text(first_page_raw))
        for rec in records:
            rec["title"] = title or name

        if glyphs_repaired:
            print(
                f"[ingest] {name}: repaired {glyphs_repaired} symbol-font "
                f"minus sign(s) mis-decoded as the digit '2'."
            )

        return records, DocumentStatus(
            document=name,
            status="ok",
            pages_total=pages_total,
            pages_kept=len(records),
            characters=total_chars,
            title=title or name,
            message="",
        )
    finally:
        doc.close()


# ---------------------------------------------------------------------------
# Folder-level loading
# ---------------------------------------------------------------------------
def load_pdf_directory(
    directory: Path,
    min_page_chars: int = 50,
    recursive: bool = False,
) -> Tuple[List[Dict[str, object]], IngestReport]:
    """Load every PDF in `directory` into page records.

    Returns `(records, report)`. `records` is a flat list across all documents;
    `report` records per-document success/failure so the UI can show honest
    feedback instead of silently indexing nothing.
    """
    directory = Path(directory)
    report = IngestReport()

    if not directory.exists():
        print(f"[ingest] Directory does not exist: {directory}")
        return [], report

    pattern = "**/*.pdf" if recursive else "*.pdf"
    pdf_paths = sorted(
        p for p in directory.glob(pattern) if p.is_file() and not p.name.startswith(".")
    )

    if not pdf_paths:
        print(f"[ingest] No PDF files found in {directory}")
        return [], report

    all_records: List[Dict[str, object]] = []
    for pdf_path in pdf_paths:
        records, status = load_pdf(pdf_path, min_page_chars=min_page_chars)
        report.documents.append(status)
        all_records.extend(records)

    return all_records, report


def load_pdf_paths(
    paths: Iterable[Path],
    min_page_chars: int = 50,
) -> Tuple[List[Dict[str, object]], IngestReport]:
    """Load an explicit list of PDF paths (used for Streamlit uploads)."""
    report = IngestReport()
    all_records: List[Dict[str, object]] = []
    for pdf_path in paths:
        records, status = load_pdf(Path(pdf_path), min_page_chars=min_page_chars)
        report.documents.append(status)
        all_records.extend(records)
    return all_records, report


# ---------------------------------------------------------------------------
# Manual check:  python -m src.ingest
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from src.config import SAMPLE_PAPERS_DIR, Settings

    settings = Settings.from_env()
    recs, rep = load_pdf_directory(
        SAMPLE_PAPERS_DIR, min_page_chars=settings.min_page_chars
    )
    print(rep.as_text())
    if recs:
        first = recs[0]
        print("\n--- first page record ---")
        print(f"document: {first['document']}")
        print(f"title:    {first['title']}")
        print(f"page:     {first['page']}")
        print(f"text[:400]:\n{str(first['text'])[:400]}")
