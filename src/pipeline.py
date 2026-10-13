"""Small orchestration layer that wires the five pipeline stages together.

This exists so that `app.py`, the notebook and the command-line scripts all
run *exactly the same* code path. Without it, the Streamlit app and the
notebook drift apart and you end up demoing something you never evaluated.

    PDFs -> ingest -> chunk -> embed -> FAISS index -> retrieve -> generate
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple


from src.chunking import chunk_records, chunk_statistics
from src.config import Settings
from src.embeddings import EmbeddingModel, embed_chunks
from src.generator import GenerationResult, generate_answer
from src.ingest import IngestReport, load_pdf_directory, load_pdf_paths
from src.retriever import FaissRetriever, RetrievedChunk, build_index

ProgressFn = Callable[[str], None]


def _noop(_message: str) -> None:
    """Default progress callback - silent."""


@dataclass
class BuildOutcome:
    """Result of building an index, including everything the UI wants to show."""

    retriever: Optional[FaissRetriever]
    chunks: List[Dict[str, object]] = field(default_factory=list)
    ingest_report: Optional[IngestReport] = None
    stats: Dict[str, float] = field(default_factory=dict)
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.retriever is not None and not self.error


def build_pipeline(
    pdf_dir: Optional[Path] = None,
    pdf_paths: Optional[Sequence[Path]] = None,
    settings: Optional[Settings] = None,
    model: Optional[EmbeddingModel] = None,
    progress: ProgressFn = _noop,
    save_to: Optional[Path] = None,
) -> BuildOutcome:
    """Run ingestion -> chunking -> embedding -> index build.

    Supply either `pdf_dir` (index a folder) or `pdf_paths` (index specific
    files, e.g. Streamlit uploads). Errors are returned inside `BuildOutcome`
    rather than raised, so callers can show a message instead of a traceback.
    """
    settings = settings or Settings.from_env()

    # ---- 1. Ingest --------------------------------------------------------
    progress("Reading PDFs ...")
    if pdf_paths is not None:
        records, report = load_pdf_paths(pdf_paths, settings.min_page_chars)
    elif pdf_dir is not None:
        records, report = load_pdf_directory(pdf_dir, settings.min_page_chars)
    else:
        return BuildOutcome(None, error="No PDF source given (pdf_dir or pdf_paths).")

    if report.n_found == 0:
        return BuildOutcome(
            None,
            ingest_report=report,
            error=(
                "No PDF files were found. Put open-access biomedical papers in "
                "data/sample_papers/ (or upload them in the sidebar) and try again."
            ),
        )

    if not records:
        return BuildOutcome(
            None,
            ingest_report=report,
            error=(
                "PDFs were found but no readable text could be extracted. "
                "They are most likely scanned / image-only documents; OCR is "
                "not part of this MVP.\n\n" + report.as_text()
            ),
        )

    # ---- 2. Chunk ---------------------------------------------------------
    progress(f"Chunking {len(records)} pages ...")
    chunks = chunk_records(records, settings.chunk_size, settings.chunk_overlap)
    if not chunks:
        return BuildOutcome(
            None, ingest_report=report, error="Chunking produced no chunks."
        )

    # ---- 3. Embed ---------------------------------------------------------
    progress(f"Embedding {len(chunks)} chunks (first run downloads the model) ...")
    encoder = model or EmbeddingModel(
        settings.embedding_model, settings.embedding_device
    )
    try:
        matrix, chunks = embed_chunks(chunks, model=encoder, show_progress=False)
    except Exception as exc:
        return BuildOutcome(
            None,
            chunks=chunks,
            ingest_report=report,
            error=f"Embedding failed: {exc}",
        )

    # ---- 4. Index ---------------------------------------------------------
    progress("Building FAISS index ...")
    try:
        retriever = build_index(chunks, matrix, settings.embedding_model)
    except Exception as exc:
        return BuildOutcome(
            None,
            chunks=chunks,
            ingest_report=report,
            error=f"Index construction failed: {exc}",
        )

    if save_to is not None:
        try:
            retriever.save(save_to)
            progress(f"Index saved to {save_to}")
        except Exception as exc:  # saving is a convenience, not a requirement
            progress(f"WARNING: could not save index ({exc})")

    return BuildOutcome(
        retriever=retriever,
        chunks=chunks,
        ingest_report=report,
        stats=chunk_statistics(chunks),
    )


def answer_question(
    question: str,
    retriever: FaissRetriever,
    model: EmbeddingModel,
    settings: Optional[Settings] = None,
    top_k: Optional[int] = None,
    min_score: Optional[float] = None,
) -> Tuple[List[RetrievedChunk], GenerationResult]:
    """Retrieve evidence for `question` and generate a grounded answer.

    Returns `(hits, generation_result)` so the caller can display the raw
    retrieved passages *separately* from the generated answer - which is the
    whole point of a citation-first design.
    """
    settings = settings or Settings.from_env()
    question = (question or "").strip()

    if not question:
        return [], GenerationResult(
            mode=settings.generation_mode(), answer="", error="Please enter a question."
        )

    k = int(top_k or settings.top_k)
    hits = retriever.retrieve(model.encode_query(question), top_k=k, min_score=min_score)
    result = generate_answer(question, hits, settings)
    return hits, result
