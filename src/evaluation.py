"""Retrieval evaluation: Recall@K and MRR@depth over hand-labelled ground truth.

What is measured
----------------
This module evaluates **retrieval only** - can the system put the right
passage in front of the model? It does not score answer quality. Answer
quality is assessed by a separate, explicitly manual grounding review
(`create_grounding_review_template`), because automatic answer scoring would
require either a second LLM judge or human-written reference answers, neither
of which is part of the MVP.

Strictness rules (these exist so a number can be trusted)
---------------------------------------------------------
A question is **measured** only if all of the following hold. Otherwise it is
**skipped**, with a recorded reason, and excluded from every aggregate:

1. `expected_document` is non-empty.
2. `expected_page` is non-empty *when* `require_page=True` (the default for a
   page-grounded evaluation). A blank page field never silently degrades into
   document-level matching.
3. `expected_page` parses cleanly. A malformed value (e.g. `"p4"`, `"4-5"`,
   `"four"`) is **always** a skip, in both page and document modes - it is a
   data-entry error, and guessing at it would corrupt the metric.
4. `expected_document` is present in the index being evaluated. A filename
   typo would otherwise be scored as a retrieval miss and silently deflate
   Recall@K.
5. Every `expected_page` is present among the pages actually indexed for that
   document (widened by `page_tolerance`). Ingestion drops near-empty pages,
   so a ground-truth page that was never indexed can never be retrieved; that
   is a ground-truth/ingestion mismatch, not a retrieval failure.

A measured question that retrieves nothing relevant is a genuine **miss**:
`first_relevant_rank` is blank and `reciprocal_rank` is 0.0. Misses count
against the metrics. Skips do not - they are reported separately.

Metric naming
-------------
Reciprocal rank is only observable within the depth actually searched, so the
summary reports `mrr_at_<search_depth>`, never a bare `mrr`. Per question the
binary outcome is a **hit** (`hit_at_k`); the aggregate mean over questions is
**recall** (`recall_at_k`). Requesting a K deeper than the search depth is an
error rather than a silently truncated number.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

import pandas as pd

from src.embeddings import EmbeddingModel
from src.retriever import FaissRetriever, RetrievedChunk

REQUIRED_COLUMNS = ["question", "expected_document", "expected_page", "notes"]
DEFAULT_K_VALUES: Tuple[int, ...] = (1, 3, 5)

# Skip reason codes (stable strings - used in CSV output and tests)
SKIP_MISSING_DOCUMENT = "missing_expected_document"
SKIP_MISSING_PAGE = "missing_expected_page"
SKIP_MALFORMED_PAGE = "malformed_expected_page"
SKIP_UNKNOWN_DOCUMENT = "expected_document_not_in_index"
SKIP_PAGE_NOT_INDEXED = "expected_page_not_in_index"


class EvaluationError(RuntimeError):
    """Raised when the evaluation set or configuration is unusable."""


# ---------------------------------------------------------------------------
# Loading the question set
# ---------------------------------------------------------------------------
def load_questions(csv_path: Union[str, Path]) -> pd.DataFrame:
    """Load the evaluation question set and validate its columns."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise EvaluationError(
            f"Evaluation question file not found: {csv_path}\n"
            f"Create it with columns: {', '.join(REQUIRED_COLUMNS)}"
        )

    frame = pd.read_csv(csv_path, dtype=str).fillna("")
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise EvaluationError(
            f"{csv_path} is missing required column(s): {', '.join(missing)}"
        )

    for column in ("question", "expected_document", "expected_page"):
        frame[column] = frame[column].astype(str).str.strip()
    frame = frame[frame["question"] != ""].reset_index(drop=True)
    return frame


def parse_expected_pages(raw: str) -> Tuple[Set[int], List[str]]:
    """Parse `expected_page` into `(pages, bad_tokens)`.

    Accepts a single page (``7``), or several separated by ``;`` ``,`` or
    whitespace (``7;8``). Any token that is not a positive integer is returned
    in `bad_tokens` rather than being dropped, so the caller can skip the row
    instead of silently evaluating against a partial page set.

    An empty input yields `(set(), [])` - "not specified", which is different
    from "specified but unparseable".
    """
    raw = (raw or "").strip()
    if not raw:
        return set(), []

    pages: Set[int] = set()
    bad: List[str] = []
    for token in re.split(r"[;,\s]+", raw):
        if not token:
            continue
        if re.fullmatch(r"\d+", token) and int(token) >= 1:
            pages.add(int(token))
        else:
            bad.append(token)
    return pages, bad


def _normalize_document(name: str) -> str:
    """Compare documents by lower-cased file name, ignoring directories."""
    return Path(str(name).strip()).name.lower()


# ---------------------------------------------------------------------------
# Relevance judgement
# ---------------------------------------------------------------------------
def is_relevant(
    hit: RetrievedChunk,
    expected_document: str,
    expected_pages: Set[int],
    page_tolerance: int = 0,
    require_page: bool = True,
) -> bool:
    """Does this retrieved chunk match the recorded ground truth?

    With `require_page=True` (the default) an empty `expected_pages` can never
    match - a page-grounded evaluation has nothing to compare against, and the
    caller is expected to have skipped the row before reaching this point.
    """
    if _normalize_document(hit.document) != _normalize_document(expected_document):
        return False
    if not expected_pages:
        return not require_page  # document-level match only when explicitly allowed
    return any(abs(hit.page - page) <= page_tolerance for page in expected_pages)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass
class SkippedQuestion:
    """A question that could not be measured, and why."""

    question: str
    expected_document: str
    expected_page: str
    reason: str
    detail: str = ""

    def as_row(self) -> Dict[str, str]:
        return {
            "question": self.question,
            "expected_document": self.expected_document,
            "expected_page": self.expected_page,
            "skip_reason": self.reason,
            "detail": self.detail,
        }


@dataclass
class EvaluationOutcome:
    """Per-question and aggregate results of a retrieval evaluation run."""

    results: pd.DataFrame
    summary: Dict[str, object]
    skipped: List[SkippedQuestion] = field(default_factory=list)

    @property
    def n_measured(self) -> int:
        return int(len(self.results))

    @property
    def n_skipped(self) -> int:
        return len(self.skipped)

    def skipped_frame(self) -> pd.DataFrame:
        """Skipped rows preserved explicitly, never folded into the metrics."""
        if not self.skipped:
            return pd.DataFrame(
                columns=["question", "expected_document", "expected_page",
                         "skip_reason", "detail"]
            )
        return pd.DataFrame([s.as_row() for s in self.skipped])

    def summary_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"metric": list(self.summary.keys()),
             "value": list(self.summary.values())}
        )

    def as_text(self) -> str:
        lines = ["Retrieval evaluation summary", "-" * 34]
        for key, value in self.summary.items():
            lines.append(f"{key:<30} {value}")
        if self.skipped:
            lines.append("")
            lines.append(f"Skipped (NOT counted as failures): {len(self.skipped)}")
            for item in self.skipped:
                lines.append(f"  [{item.reason}] {item.question[:64]}")
                if item.detail:
                    lines.append(f"      {item.detail}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Ground-truth validation
# ---------------------------------------------------------------------------
def _index_pages_by_document(retriever: FaissRetriever) -> Dict[str, Set[int]]:
    """Map normalized document name -> set of page numbers actually indexed."""
    pages: Dict[str, Set[int]] = {}
    for chunk in retriever.chunks:
        key = _normalize_document(str(chunk["document"]))
        pages.setdefault(key, set()).add(int(chunk["page"]))
    return pages


def validate_row(
    question: str,
    expected_document: str,
    expected_page_raw: str,
    indexed_pages: Dict[str, Set[int]],
    require_page: bool,
    page_tolerance: int,
) -> Tuple[Optional[Set[int]], Optional[SkippedQuestion]]:
    """Validate one ground-truth row.

    Returns `(expected_pages, None)` when the row is measurable, or
    `(None, SkippedQuestion)` explaining why it is not.
    """
    def skip(reason: str, detail: str = "") -> Tuple[None, SkippedQuestion]:
        return None, SkippedQuestion(
            question=question,
            expected_document=expected_document,
            expected_page=expected_page_raw,
            reason=reason,
            detail=detail,
        )

    if not expected_document:
        return skip(SKIP_MISSING_DOCUMENT, "No expected_document recorded.")

    pages, bad_tokens = parse_expected_pages(expected_page_raw)

    # Malformed is always fatal for the row - never partially evaluated.
    if bad_tokens:
        return skip(
            SKIP_MALFORMED_PAGE,
            f"Unparseable page token(s): {', '.join(bad_tokens)} "
            f"(expected positive integers, e.g. '4' or '4;5').",
        )

    if not pages and require_page:
        return skip(
            SKIP_MISSING_PAGE,
            "Page-grounded evaluation requires at least one expected_page; "
            "pass require_page=False only for a deliberately document-level run.",
        )

    key = _normalize_document(expected_document)
    if key not in indexed_pages:
        return skip(
            SKIP_UNKNOWN_DOCUMENT,
            f"'{expected_document}' is not among the indexed documents "
            f"({len(indexed_pages)} indexed). Check the filename spelling.",
        )

    if pages:
        available = indexed_pages[key]
        reachable = {
            page
            for page in pages
            if any(abs(page - indexed) <= page_tolerance for indexed in available)
        }
        if not reachable:
            return skip(
                SKIP_PAGE_NOT_INDEXED,
                f"Page(s) {sorted(pages)} were not indexed for "
                f"'{expected_document}' (indexed pages: {sorted(available)}). "
                f"Ingestion skips near-empty pages, so this page can never be "
                f"retrieved - fix the ground truth or check extraction.",
            )

    return pages, None


# ---------------------------------------------------------------------------
# Evaluation run
# ---------------------------------------------------------------------------
def evaluate_retrieval(
    retriever: FaissRetriever,
    model: EmbeddingModel,
    questions: pd.DataFrame,
    top_k: int = 5,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    page_tolerance: int = 0,
    require_page: bool = True,
) -> EvaluationOutcome:
    """Run every evaluation question through retrieval and score the results.

    Parameters
    ----------
    top_k:
        The application's configured retrieval depth.
    k_values:
        The K values to report Recall at. Every K must be <= the search depth.
    page_tolerance:
        Allow a match within N pages of an expected page (0 = exact).
    require_page:
        True (default) enforces page-grounded ground truth. False permits
        document-level matching for rows with a blank page field - malformed
        page values are still skipped.

    Returns
    -------
    `EvaluationOutcome` with one results row per **measured** question, a
    summary whose aggregates cover measured questions only, and the skipped
    rows preserved separately.
    """
    k_values = tuple(sorted({int(k) for k in k_values}))
    if not k_values:
        raise EvaluationError("k_values must contain at least one K.")

    # The search depth is exactly `top_k` - the depth the application actually
    # retrieves at. It is deliberately NOT widened to cover a larger requested
    # K: that would silently change which depth the reported MRR refers to.
    search_depth = int(top_k)
    if search_depth < 1:
        raise EvaluationError(f"top_k must be >= 1, got {top_k}.")

    too_deep = [k for k in k_values if k > search_depth]
    if too_deep:
        raise EvaluationError(
            f"Cannot report Recall@{too_deep} - only the top {search_depth} "
            f"results are retrieved (top_k={top_k}). Raise top_k to evaluate "
            f"that depth, or drop those K values."
        )

    indexed_pages = _index_pages_by_document(retriever)

    rows: List[Dict[str, object]] = []
    skipped: List[SkippedQuestion] = []

    for _, row in questions.iterrows():
        question = str(row["question"]).strip()
        expected_document = str(row["expected_document"]).strip()
        expected_page_raw = str(row["expected_page"]).strip()

        expected_pages, skip = validate_row(
            question=question,
            expected_document=expected_document,
            expected_page_raw=expected_page_raw,
            indexed_pages=indexed_pages,
            require_page=require_page,
            page_tolerance=page_tolerance,
        )
        if skip is not None:
            skipped.append(skip)
            continue
        assert expected_pages is not None  # for type checkers

        hits = retriever.retrieve(model.encode_query(question), top_k=search_depth)

        first_relevant_rank: Optional[int] = None
        for hit in hits:
            if is_relevant(
                hit, expected_document, expected_pages, page_tolerance, require_page
            ):
                first_relevant_rank = hit.rank
                break

        record: Dict[str, object] = {
            "question": question,
            "expected_document": expected_document,
            "expected_page": expected_page_raw,
            "match_level": "page" if expected_pages else "document",
            "search_depth": search_depth,
            "retrieved_documents": "; ".join(h.document for h in hits),
            "retrieved_pages": "; ".join(str(h.page) for h in hits),
            "retrieved_chunk_ids": "; ".join(h.chunk_id for h in hits),
            "retrieved_scores": "; ".join(f"{h.score:.4f}" for h in hits),
            "top1_score": round(hits[0].score, 4) if hits else "",
            "first_relevant_rank": first_relevant_rank if first_relevant_rank else "",
            "reciprocal_rank": (
                round(1.0 / first_relevant_rank, 4) if first_relevant_rank else 0.0
            ),
        }
        # Per question the outcome is a HIT (binary); recall is the aggregate.
        for k in k_values:
            record[f"hit_at_{k}"] = int(
                first_relevant_rank is not None and first_relevant_rank <= k
            )
        rows.append(record)

    results = pd.DataFrame(rows)
    reason_counts = Counter(item.reason for item in skipped)

    summary: Dict[str, object] = {
        "n_questions_in_file": int(len(questions)),
        "n_measured": int(len(results)),
        "n_skipped": len(skipped),
        "top_k": int(top_k),
        "search_depth": search_depth,
        "page_tolerance": int(page_tolerance),
        "require_page": require_page,
        "embedding_model": retriever.embedding_model_name,
        "n_chunks_indexed": retriever.n_chunks,
        "n_documents_indexed": len(retriever.documents),
    }
    for reason, count in sorted(reason_counts.items()):
        summary[f"skipped__{reason}"] = count

    if len(results):
        for k in k_values:
            summary[f"recall_at_{k}"] = round(float(results[f"hit_at_{k}"].mean()), 4)
        # Reciprocal rank is only observable within the searched depth, so the
        # name carries that depth. Never report a bare "mrr".
        summary[f"mrr_at_{search_depth}"] = round(
            float(results["reciprocal_rank"].mean()), 4
        )
        summary["n_misses"] = int((results["first_relevant_rank"] == "").sum())
    else:
        not_measured = "not measured (no question had usable ground truth)"
        for k in k_values:
            summary[f"recall_at_{k}"] = not_measured
        summary[f"mrr_at_{search_depth}"] = not_measured
        summary["n_misses"] = not_measured

    return EvaluationOutcome(results=results, summary=summary, skipped=skipped)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def save_outcome(
    outcome: EvaluationOutcome,
    results_path: Union[str, Path],
    summary_path: Optional[Union[str, Path]] = None,
    skipped_path: Optional[Union[str, Path]] = None,
) -> None:
    """Write measured results, the summary, and the skipped rows to CSV.

    Skipped rows get their own file so they stay visible instead of vanishing
    into a count.
    """
    results_path = Path(results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    outcome.results.to_csv(results_path, index=False)

    if summary_path is not None:
        summary_path = Path(summary_path)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        outcome.summary_frame().to_csv(summary_path, index=False)

    if skipped_path is not None:
        skipped_path = Path(skipped_path)
        skipped_path.parent.mkdir(parents=True, exist_ok=True)
        outcome.skipped_frame().to_csv(skipped_path, index=False)


# ---------------------------------------------------------------------------
# Manual grounding / failure-analysis template
# ---------------------------------------------------------------------------
GROUNDING_REVIEW_COLUMNS = [
    "question",
    "retrieved_evidence_ids",
    "generated_answer",
    "fully_supported",        # yes / no / partial
    "unsupported_claim",      # quote the specific unsupported sentence
    "citation_correct",       # yes / no
    "failure_description",
]


def create_grounding_review_template(
    path: Union[str, Path], overwrite: bool = False
) -> Path:
    """Create an empty CSV for the manual answer-grounding review.

    This is an engineering failure-analysis exercise filled in by a human. It
    is not a validated hallucination detector and must not be described as one.
    """
    path = Path(path)
    if path.exists() and not overwrite:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=GROUNDING_REVIEW_COLUMNS).to_csv(path, index=False)
    return path


def failure_cases(outcome: EvaluationOutcome) -> pd.DataFrame:
    """Measured questions where no relevant chunk was retrieved (genuine misses).

    Skipped questions are deliberately excluded - they were never measured, so
    they are not failures of retrieval.
    """
    if not len(outcome.results):
        return outcome.results
    return outcome.results[
        outcome.results["first_relevant_rank"] == ""
    ].reset_index(drop=True)
