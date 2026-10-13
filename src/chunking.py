"""Chunking: page records -> overlapping, citable text chunks.

Why a hand-written chunker
--------------------------
Chunking is the single biggest lever on retrieval quality, so for an MVP it is
worth keeping it transparent and debuggable rather than hiding it behind a
framework. The algorithm here is small enough to explain in a minute:

1. Split each page into *units* - paragraphs first, then sentences if a
   paragraph is too long, then hard word splits as a last resort.
2. Greedily pack units into a chunk until adding the next one would exceed
   `chunk_size`.
3. Seed the next chunk with the tail of the previous one (`chunk_overlap`
   characters, snapped to a sentence or word boundary) so that a fact sitting
   on a chunk boundary is not lost.

Crucially, chunking never crosses a page boundary. That is what makes
"paper1.pdf - page 7" an honest citation rather than an approximation.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------
# Split after . ! ? when followed by whitespace and something that looks like
# the start of a new sentence. The negative lookbehinds suppress the most
# common false positives in biomedical prose ("et al.", "e.g.", "Fig.", ...).
_SENTENCE_BOUNDARY = re.compile(
    r"(?<=[.!?])"
    r"(?<!al\.)"       # et al.
    r"(?<!e\.g\.)"
    r"(?<!i\.e\.)"
    r"(?<!Fig\.)"
    r"(?<!fig\.)"
    r"(?<!Eq\.)"
    r"(?<!Ref\.)"
    r"(?<!No\.)"
    r"(?<!vs\.)"
    r"(?<!ca\.)"
    r"(?<!cf\.)"
    r"(?<!Dr\.)"
    r"(?<!Mr\.)"
    r"(?<!Ms\.)"
    r"\s+"
    r"(?=[A-Z(\[\"'“0-9])"
)

_WORD_BOUNDARY = re.compile(r"\s+")


def split_sentences(text: str) -> List[str]:
    """Split a paragraph into sentences (best effort, never lossy)."""
    parts = [s.strip() for s in _SENTENCE_BOUNDARY.split(text)]
    return [s for s in parts if s]


def split_paragraphs(text: str) -> List[str]:
    """Split page text into paragraphs on blank lines."""
    parts = [p.strip() for p in text.split("\n\n")]
    return [p for p in parts if p]


def _hard_split(text: str, max_chars: int) -> List[str]:
    """Split an over-long unit on word boundaries, never mid-word if avoidable."""
    words = _WORD_BOUNDARY.split(text)
    pieces: List[str] = []
    current = ""
    for word in words:
        if not word:
            continue
        # A single "word" longer than max_chars (e.g. a long URL or a table row
        # glued together) is cut bluntly - rare, but must terminate.
        if len(word) > max_chars:
            if current:
                pieces.append(current)
                current = ""
            for i in range(0, len(word), max_chars):
                pieces.append(word[i : i + max_chars])
            continue
        candidate = f"{current} {word}".strip()
        if len(candidate) > max_chars and current:
            pieces.append(current)
            current = word
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces


def build_units(text: str, max_chars: int) -> List[Tuple[str, bool]]:
    """Break page text into units no longer than `max_chars`.

    Preference order: paragraph -> sentence -> word run.

    Returns a list of `(unit_text, starts_new_paragraph)` pairs. The flag lets
    the packer rejoin sentences from the same paragraph with a space and
    separate genuine paragraphs with a blank line, so chunk text stays readable
    (which matters: a human reviewer reads these snippets in the UI).
    """
    units: List[Tuple[str, bool]] = []
    for paragraph in split_paragraphs(text):
        if len(paragraph) <= max_chars:
            units.append((paragraph, True))
            continue
        first = True
        for sentence in split_sentences(paragraph):
            pieces = (
                [sentence]
                if len(sentence) <= max_chars
                else _hard_split(sentence, max_chars)
            )
            for piece in pieces:
                units.append((piece, first))
                first = False
    return units


def _overlap_tail(text: str, overlap_chars: int) -> str:
    """Return the last ~`overlap_chars` of `text`, snapped to a clean boundary.

    Tries a sentence boundary first so the overlap reads as real prose; falls
    back to a word boundary.
    """
    if overlap_chars <= 0 or not text:
        return ""
    if len(text) <= overlap_chars:
        return text

    tail = text[-overlap_chars:]

    # Prefer starting the tail at a sentence boundary inside the window.
    match = _SENTENCE_BOUNDARY.search(tail)
    if match:
        candidate = tail[match.end() :].strip()
        if len(candidate) >= overlap_chars // 3:
            return candidate

    # Otherwise snap forward to the next whitespace so we do not cut a word.
    space = tail.find(" ")
    if space != -1:
        return tail[space + 1 :].strip()
    return tail.strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def chunk_page_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 175,
) -> List[str]:
    """Split one page's text into overlapping chunks of ~`chunk_size` chars."""
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be smaller than "
            f"chunk_size ({chunk_size})."
        )
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    units = build_units(text, max_chars=chunk_size)
    chunks: List[str] = []
    current = ""

    for unit, new_paragraph in units:
        if not current:
            current = unit
            continue
        separator = "\n\n" if new_paragraph else " "
        candidate = f"{current}{separator}{unit}"
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            chunks.append(current.strip())
            seed = _overlap_tail(current, chunk_overlap)
            # Only seed the next chunk if the overlap leaves room for new content.
            if seed and len(seed) + len(unit) + 1 <= chunk_size:
                current = f"{seed} {unit}"
            else:
                current = unit

    if current.strip():
        chunks.append(current.strip())

    return chunks


def make_chunk_id(document: str, page: int, index: int) -> str:
    """Stable, human-readable chunk identifier, e.g. `paper1_p5_c2`."""
    stem = Path(document).stem
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("_") or "doc"
    return f"{stem}_p{page}_c{index}"


def chunk_records(
    page_records: Sequence[Dict[str, object]],
    chunk_size: int = 1000,
    chunk_overlap: int = 175,
    min_chunk_chars: int = 120,
) -> List[Dict[str, object]]:
    """Convert page records into chunk records with full metadata.

    Parameters
    ----------
    page_records:
        Output of `src.ingest.load_pdf_directory` - dicts with
        `document`, `page`, `text` (and optionally `title`).
    chunk_size, chunk_overlap:
        Target chunk length and overlap, in characters.
    min_chunk_chars:
        A trailing fragment shorter than this is merged back into the previous
        chunk of the same page (when that stays within 1.3x chunk_size),
        otherwise kept. Nothing is ever discarded.

    Returns
    -------
    A list of dicts::

        {
            "document": "paper1.pdf",
            "title":    "Effect of ...",
            "page":     5,
            "chunk_id": "paper1_p5_c2",
            "text":     "...",
            "n_chars":  987,
        }
    """
    chunks: List[Dict[str, object]] = []

    for record in page_records:
        document = str(record["document"])
        page = int(record["page"])
        title = str(record.get("title") or document)
        text = str(record.get("text") or "")

        pieces = chunk_page_text(text, chunk_size, chunk_overlap)

        # Merge a too-short trailing fragment back into its predecessor.
        if (
            len(pieces) >= 2
            and len(pieces[-1]) < min_chunk_chars
            and len(pieces[-2]) + len(pieces[-1]) + 1 <= int(chunk_size * 1.3)
        ):
            pieces[-2] = f"{pieces[-2]} {pieces[-1]}".strip()
            pieces.pop()

        for i, piece in enumerate(pieces, start=1):
            chunks.append(
                {
                    "document": document,
                    "title": title,
                    "page": page,
                    "chunk_id": make_chunk_id(document, page, i),
                    "text": piece,
                    "n_chars": len(piece),
                }
            )

    return chunks


def chunk_statistics(chunks: Sequence[Dict[str, object]]) -> Dict[str, float]:
    """Basic descriptive stats, handy in the notebook and the Streamlit sidebar."""
    if not chunks:
        return {"n_chunks": 0, "n_documents": 0, "mean_chars": 0.0,
                "min_chars": 0, "max_chars": 0}
    lengths = [int(c["n_chars"]) for c in chunks]
    return {
        "n_chunks": len(chunks),
        "n_documents": len({str(c["document"]) for c in chunks}),
        "mean_chars": round(sum(lengths) / len(lengths), 1),
        "min_chars": min(lengths),
        "max_chars": max(lengths),
    }


# ---------------------------------------------------------------------------
# Manual check:  python -m src.chunking
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from src.config import SAMPLE_PAPERS_DIR, Settings
    from src.ingest import load_pdf_directory

    settings = Settings.from_env()
    records, report = load_pdf_directory(
        SAMPLE_PAPERS_DIR, min_page_chars=settings.min_page_chars
    )
    print(report.as_text())

    chunks = chunk_records(records, settings.chunk_size, settings.chunk_overlap)
    print("\n", chunk_statistics(chunks))
    for chunk in chunks[:2]:
        print("\n---", chunk["chunk_id"], f"({chunk['n_chars']} chars)")
        print(str(chunk["text"])[:300], "...")
