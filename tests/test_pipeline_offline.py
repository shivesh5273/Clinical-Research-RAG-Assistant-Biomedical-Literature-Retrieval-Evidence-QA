"""Offline integration test for the RAG pipeline - no network, no model download.

Why a stub encoder
------------------
The real encoder (`sentence-transformers/all-MiniLM-L6-v2`) has to be
downloaded on first use. That makes it a bad dependency for a test whose job is
to check *our* logic: chunk metadata, FAISS wiring, save/reload, citation
formatting, evaluation maths, error handling.

So this test swaps in `StubEncoder` - a deterministic hashed bag-of-words
encoder with no dependencies. It produces genuine lexical-overlap similarity,
which is enough to exercise every code path end to end.

IMPORTANT: any retrieval numbers printed by this test are produced by the STUB
encoder on FABRICATED documents. They are not a measurement of this system's
retrieval quality and must never be reported as one. For real numbers, run
`python scripts/run_evaluation.py` with the real encoder and your real papers.

Run with:
    python tests/test_pipeline_offline.py
"""

from __future__ import annotations

import hashlib
import re
import sys
import tempfile
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.chunking import chunk_page_text, chunk_records, make_chunk_id  # noqa: E402
from src.config import EVALUATION_DIR, TEST_FIXTURES_DIR, Settings  # noqa: E402
from src.evaluation import (  # noqa: E402
    SKIP_MALFORMED_PAGE,
    SKIP_MISSING_DOCUMENT,
    SKIP_MISSING_PAGE,
    SKIP_PAGE_NOT_INDEXED,
    SKIP_UNKNOWN_DOCUMENT,
    EvaluationError,
    evaluate_retrieval,
    failure_cases,
    is_relevant,
    load_questions,
    parse_expected_pages,
)
from src.generator import (  # noqa: E402
    INSUFFICIENT_EVIDENCE,
    build_prompt,
    find_unknown_citations,
    format_sources_block,
    generate_answer,
)
from src.ingest import (  # noqa: E402
    load_pdf_directory,
    normalize_text,
    repair_minus_glyphs,
)
from src.retriever import FaissRetriever, RetrieverError, build_index  # noqa: E402

DIMENSION = 256
_TOKEN = re.compile(r"[a-z0-9]+")

_failures: List[str] = []
_checks = 0


def check(condition: bool, label: str) -> None:
    """Tiny assertion helper so one failure does not stop the whole run."""
    global _checks
    _checks += 1
    if condition:
        print(f"  ok   {label}")
    else:
        print(f"  FAIL {label}")
        _failures.append(label)


# ---------------------------------------------------------------------------
# Deterministic stub encoder (duck-types EmbeddingModel)
# ---------------------------------------------------------------------------
class StubEncoder:
    """Hashed bag-of-words encoder. Deterministic, offline, dependency-free."""

    def __init__(
        self,
        model_name: str = "stub-hashing-encoder",
        device: str | None = None,
        dimension: int = DIMENSION,
    ):
        # Signature deliberately mirrors EmbeddingModel so this class can be
        # dropped in wherever EmbeddingModel is constructed.
        self.model_name = model_name
        self.device = device
        self._dimension = dimension

    def load(self) -> "StubEncoder":
        return self

    @property
    def dimension(self) -> int:
        return self._dimension

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self._dimension, dtype="float32")
        tokens = _TOKEN.findall(text.lower())
        for token in tokens:
            if len(token) < 3:
                continue
            digest = hashlib.md5(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % self._dimension
            vec[bucket] += 1.0
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm else vec

    def encode(
        self,
        texts: Sequence[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> np.ndarray:
        if not len(texts):
            return np.zeros((0, self._dimension), dtype="float32")
        return np.vstack([self._vector(t) for t in texts]).astype("float32")

    def encode_query(self, question: str) -> np.ndarray:
        return self.encode([question])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_normalization() -> None:
    print("\n[1] text normalization")
    joined = normalize_text("random-\nized controlled trial")
    check("randomized controlled trial" in joined, "hyphenated line break rejoined")
    check(
        normalize_text("a\n\nb") == "a\n\nb", "paragraph breaks preserved"
    )
    check(normalize_text("a\nb") == "a b", "single line break becomes a space")
    check(normalize_text("") == "", "empty input is safe")
    check("  " not in normalize_text("a     b"), "repeated spaces collapsed")


def test_glyph_repair() -> None:
    """Regression test for the symbol-font minus-sign defect (RAED2 / PMC3308269).

    A publisher font renders the minus sign but decodes to the digit "2", so
    "-0.40%" extracts as "20.40%" - every negative result becomes a positive
    number an order of magnitude too large. These cases are built as synthetic
    character/font streams so the test needs no PDF and no network.
    """
    print("\n[2] symbol-font minus-sign repair")

    TEXT = "TextFont"
    SYM = "SymFont"
    # A real symbol font's whole document-wide inventory is tiny.
    INV = {TEXT: set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,%[]()"),
           SYM: {" ", ",", "2", "8", "."}}

    def stream(*segments):
        return [(ch, font) for seg, font in segments for ch in seg]

    def run(segments, text=None, inventory=INV):
        s = stream(*segments)
        return repair_minus_glyphs(s, text if text is not None else
                                   "".join(seg for seg, _ in segments), inventory)

    # -- the exact published pattern -------------------------------------
    out, n = run([("both groups (", TEXT), ("2", SYM), ("0.40% [95% CI", TEXT),
                  (" 2", SYM), ("0.61 to", TEXT), (" 2", SYM), ("0.18]", TEXT)])
    check(out == "both groups (-0.40% [95% CI -0.61 to -0.18]",
          f"published CI pattern repaired: {out!r}")
    check(n == 3, f"all three minus signs repaired (got {n})")
    check("0.40" in out and "0.61" in out and "0.18" in out,
          "the numbers themselves are preserved exactly")
    check(out.count("2") == 0, "no stray '2' left inside the repaired numbers")

    # -- glued words must not block the repair ---------------------------
    out, n = run([("The mean changewas", TEXT), ("2", SYM), ("0.40%", TEXT)])
    check(out == "The mean changewas-0.40%" and n == 1,
          f"repair works even when the PDF dropped the space: {out!r}")

    # -- things that must NEVER be rewritten -----------------------------
    out, n = run([("We enrolled 20 patients and 25 controls", TEXT)])
    check(n == 0 and "20 patients" in out, "ordinary digits in body text untouched")

    out, n = run([("Table ", TEXT), ("2", SYM), (" shows", TEXT)])
    check(n == 0, "symbol-font '2' NOT followed by a digit is untouched")

    out, n = run([("value ", TEXT), ("20.40", SYM)])
    check(n == 0, "digit run in the same font is untouched (not a font boundary)")

    out, n = run([("was ", TEXT), ("2", SYM), ("0.40", TEXT)],
                 inventory={TEXT: INV[TEXT], SYM: set("abcdefghij0123456789")})
    check(n == 0, "a font with a large inventory is not treated as a symbol font")

    out, n = run([("Value 1", TEXT), ("2", SYM), ("0.40", TEXT)])
    check(n == 0 and out == "Value 120.40",
          f"a number split across a font boundary is never broken: {out!r}")

    # -- alignment safety --------------------------------------------------
    out, n = run([("groups (", TEXT), ("2", SYM), ("0.40", TEXT)],
                 text="completely different text")
    check(n == 0 and out == "completely different text",
          "misaligned stream and text leaves the text untouched")

    # -- whitespace in the extracted text must not break alignment --------
    out, n = run([("groups (", TEXT), ("2", SYM), ("0.40% and", TEXT),
                  (" 2", SYM), ("0.35%", TEXT)],
                 text="groups (20.40%\nand 20.35%")
    check(out == "groups (-0.40%\nand -0.35%" and n == 2,
          f"newlines between spans do not break the repair: {out!r}")

    # -- a negative already extracted correctly is left alone -------------
    out, n = run([("change of -0.40% was seen", TEXT)])
    check(n == 0 and out == "change of -0.40% was seen",
          "correctly-extracted negatives are not touched")


def test_chunking() -> None:
    print("\n[3] chunking")
    text = ". ".join(f"Sentence number {i} about a fabricated cohort" for i in range(120))
    pieces = chunk_page_text(text, chunk_size=600, chunk_overlap=120)
    check(len(pieces) > 1, "long page produces multiple chunks")
    check(
        all(len(p) <= 600 for p in pieces),
        f"no chunk exceeds chunk_size (max={max(len(p) for p in pieces)})",
    )
    check(
        any(pieces[0][-40:].split()[-3:] == pieces[1].split()[:3] for _ in [0])
        or pieces[0].split()[-1] in pieces[1],
        "consecutive chunks share overlapping text",
    )
    check(chunk_page_text("", 600, 120) == [], "empty page produces no chunks")
    check(chunk_page_text("short text", 600, 120) == ["short text"],
          "short page produces exactly one chunk")

    raised = False
    try:
        chunk_page_text("x" * 100, chunk_size=100, chunk_overlap=100)
    except ValueError:
        raised = True
    check(raised, "overlap >= chunk_size raises ValueError")

    check(
        make_chunk_id("My Paper (2020).pdf", 7, 3) == "My_Paper_2020_p7_c3",
        "chunk ids are filesystem-safe and readable",
    )


def test_pipeline(settings: Settings):
    print("\n[4] ingest -> chunk -> embed -> index")
    records, report = load_pdf_directory(TEST_FIXTURES_DIR, settings.min_page_chars)
    check(report.n_found == 3, f"3 fixture PDFs found (got {report.n_found})")
    check(report.n_loaded == 3, f"3 PDFs loaded (got {report.n_loaded})")
    check(len(records) > 0, f"{len(records)} page records extracted")

    chunks = chunk_records(records, settings.chunk_size, settings.chunk_overlap)
    check(len(chunks) > 0, f"{len(chunks)} chunks created")
    check(
        all({"document", "page", "chunk_id", "text"} <= set(c) for c in chunks),
        "every chunk carries document / page / chunk_id / text",
    )
    check(
        len({c["chunk_id"] for c in chunks}) == len(chunks),
        "chunk ids are unique",
    )

    encoder = StubEncoder()
    matrix = encoder.encode([str(c["text"]) for c in chunks])
    check(matrix.shape == (len(chunks), DIMENSION), f"embedding matrix {matrix.shape}")
    norms = np.linalg.norm(matrix, axis=1)
    check(bool(np.allclose(norms, 1.0, atol=1e-5)), "embeddings are unit-normalized")

    retriever = build_index(chunks, matrix, encoder.model_name)
    check(retriever.n_chunks == len(chunks), "index size matches chunk count")
    check(len(retriever.documents) == 3, "three documents in the index")
    return retriever, encoder, chunks


def test_retrieval(retriever: FaissRetriever, encoder: StubEncoder) -> None:
    print("\n[5] retrieval")
    hits = retriever.retrieve(encoder.encode_query("Which biomarkers were examined?"), top_k=5)
    check(len(hits) == 5, f"top_k=5 returns 5 hits (got {len(hits)})")
    check([h.rank for h in hits] == [1, 2, 3, 4, 5], "ranks are 1..k in order")
    check(
        all(hits[i].score >= hits[i + 1].score for i in range(len(hits) - 1)),
        "scores are non-increasing",
    )
    check(
        all(h.document and h.page > 0 and h.chunk_id and h.text for h in hits),
        "every hit carries complete metadata",
    )
    check(
        hits[0].citation().endswith(f"page {hits[0].page}"),
        f"citation string is well formed: '{hits[0].citation()}'",
    )

    top1 = retriever.retrieve(encoder.encode_query("biomarkers"), top_k=1)
    check(len(top1) == 1, "top_k=1 honoured")

    huge = retriever.retrieve(encoder.encode_query("biomarkers"), top_k=9999)
    check(len(huge) == retriever.n_chunks, "top_k larger than the corpus is clamped")

    filtered = retriever.retrieve(encoder.encode_query("biomarkers"), top_k=5, min_score=0.99)
    check(len(filtered) < 5, f"min_score filters weak hits (kept {len(filtered)})")

    raised = False
    try:
        retriever.retrieve(np.zeros((1, DIMENSION + 7), dtype="float32"), top_k=3)
    except RetrieverError:
        raised = True
    check(raised, "wrong query dimension raises a clear RetrieverError")


def test_persistence(retriever: FaissRetriever, encoder: StubEncoder) -> None:
    print("\n[6] index save / reload")
    with tempfile.TemporaryDirectory() as tmp:
        retriever.save(tmp)
        reloaded = FaissRetriever.load(tmp)
        check(reloaded.n_chunks == retriever.n_chunks, "chunk count survives a round trip")

        query = encoder.encode_query("What limitations were reported?")
        before = retriever.retrieve(query, top_k=3)
        after = reloaded.retrieve(query, top_k=3)
        check(
            [h.chunk_id for h in before] == [h.chunk_id for h in after],
            "reloaded index returns identical results",
        )

    raised = False
    try:
        with tempfile.TemporaryDirectory() as empty:
            FaissRetriever.load(empty)
    except RetrieverError:
        raised = True
    check(raised, "loading a missing index raises a clear RetrieverError")


def test_generation(retriever: FaissRetriever, encoder: StubEncoder,
                    settings: Settings) -> None:
    print("\n[7] generation (retrieval-only path)")
    question = "What limitations did the authors report?"
    hits = retriever.retrieve(encoder.encode_query(question), top_k=3)

    prompt = build_prompt(question, hits)
    check("[S1]" in prompt and "[S3]" in prompt, "prompt labels evidence [S1]..[S3]")
    check(question in prompt, "prompt contains the question")
    check(hits[0].text[:60] in prompt, "prompt contains the retrieved passage text")

    sources = format_sources_block(hits)
    check(sources.startswith("[S1] "), "sources block starts at [S1]")
    check(
        sources.count("page") == len(hits),
        "every source line carries a page number",
    )

    offline = Settings.from_env()
    offline.anthropic_api_key = ""
    offline.anthropic_model = ""
    result = generate_answer(question, hits, offline)
    check(result.mode == "retrieval_only", "falls back to retrieval-only with no key")
    check("Retrieval-only mode" in result.answer, "answer says generation is disabled")
    check("[S1]" in result.answer, "retrieval-only answer still cites sources")
    check(len(result.sources) == 3, "sources are attached to the result")

    empty = generate_answer("", hits, offline)
    check(bool(empty.error), "empty question returns an error, not a crash")

    none_found = generate_answer(question, [], offline)
    check(
        none_found.answer == INSUFFICIENT_EVIDENCE,
        "no evidence -> the exact insufficient-evidence sentence",
    )

    check(
        find_unknown_citations("Claim [S1] and claim [S9].", 5) == ["[S9]"],
        "out-of-range citation markers are detected",
    )
    check(
        find_unknown_citations("Claim [S1][S2].", 5) == [],
        "valid citation markers are not flagged",
    )


def test_evaluation(retriever: FaissRetriever, encoder: StubEncoder) -> None:
    print("\n[8] evaluation metrics")

    # -- page parsing: clean / empty / malformed are three distinct outcomes --
    check(parse_expected_pages("4;5") == ({4, 5}, []), "multi-page ground truth parsed")
    check(parse_expected_pages("7") == ({7}, []), "single page parsed")
    check(parse_expected_pages("") == (set(), []), "empty page field -> not specified")
    check(
        parse_expected_pages("p4") == (set(), ["p4"]),
        "malformed page token is reported, not silently dropped",
    )
    check(
        parse_expected_pages("4-5") == (set(), ["4-5"]),
        "range syntax is reported as malformed (use '4;5')",
    )
    check(
        parse_expected_pages("4; nine") == ({4}, ["nine"]),
        "partially malformed input surfaces the bad token",
    )
    check(parse_expected_pages("0") == (set(), ["0"]), "page 0 is malformed (1-based)")

    # -- relevance judgement -------------------------------------------------
    hits = retriever.retrieve(encoder.encode_query("biomarkers"), top_k=1)
    hit = hits[0]
    check(is_relevant(hit, hit.document, {hit.page}), "exact match is relevant")
    check(not is_relevant(hit, "other.pdf", {hit.page}), "wrong document is not relevant")
    check(
        not is_relevant(hit, hit.document, {hit.page + 5}),
        "wrong page is not relevant",
    )
    check(
        is_relevant(hit, hit.document, {hit.page + 1}, page_tolerance=1),
        "page_tolerance widens the match",
    )
    check(
        not is_relevant(hit, hit.document, set(), require_page=True),
        "no expected page never matches under page-grounded rules",
    )
    check(
        is_relevant(hit, hit.document, set(), require_page=False),
        "document-level match only when explicitly allowed",
    )

    # -- a clean run ---------------------------------------------------------
    questions = load_questions(EVALUATION_DIR / "fixture_questions.csv")
    check(len(questions) == 12, f"12 fixture questions loaded (got {len(questions)})")

    outcome = evaluate_retrieval(retriever, encoder, questions, top_k=5)
    check(outcome.n_measured == 12, f"12 measured (got {outcome.n_measured})")
    check(outcome.n_skipped == 0, f"0 skipped (got {outcome.n_skipped})")
    check(
        {"recall_at_1", "recall_at_3", "recall_at_5"} <= set(outcome.summary),
        "summary contains Recall@1/3/5",
    )
    check(
        "mrr_at_5" in outcome.summary,
        f"reciprocal rank is named for its depth: {[k for k in outcome.summary if 'mrr' in k]}",
    )
    check("mrr" not in outcome.summary, "no bare, depth-ambiguous 'mrr' key")
    check(0.0 <= float(outcome.summary["mrr_at_5"]) <= 1.0, "MRR@5 is in [0, 1]")
    check(
        float(outcome.summary["recall_at_1"]) <= float(outcome.summary["recall_at_5"]),
        "Recall@1 <= Recall@5 (monotonicity)",
    )
    check(
        all(outcome.results["match_level"] == "page"),
        "page-grounded rows are recorded as page-level matches",
    )

    # -- depth guard ---------------------------------------------------------
    raised = False
    try:
        evaluate_retrieval(retriever, encoder, questions, top_k=3, k_values=(1, 3, 10))
    except EvaluationError:
        raised = True
    check(raised, "Recall@K deeper than the search depth is an error, not a guess")

    # -- each skip reason, one row at a time ---------------------------------
    good = questions.iloc[[0]].copy()
    good_doc = str(good.iloc[0]["expected_document"])

    def one(doc: str, page: str) -> pd.DataFrame:
        frame = good.copy()
        frame.loc[frame.index[0], "expected_document"] = doc
        frame.loc[frame.index[0], "expected_page"] = page
        return frame

    cases = [
        (one("", "2"), True, SKIP_MISSING_DOCUMENT, "blank expected_document"),
        (one(good_doc, ""), True, SKIP_MISSING_PAGE,
         "blank expected_page under page-grounded rules"),
        (one(good_doc, "p2"), True, SKIP_MALFORMED_PAGE, "malformed expected_page"),
        (one(good_doc, "p2"), False, SKIP_MALFORMED_PAGE,
         "malformed page is skipped even in document-level mode"),
        (one("does_not_exist.pdf", "2"), True, SKIP_UNKNOWN_DOCUMENT,
         "expected_document absent from the index"),
        (one(good_doc, "999"), True, SKIP_PAGE_NOT_INDEXED,
         "expected_page that was never indexed"),
    ]
    for frame, require_page, expected_reason, label in cases:
        result = evaluate_retrieval(
            retriever, encoder, frame, top_k=5, require_page=require_page
        )
        ok = (
            result.n_measured == 0
            and result.n_skipped == 1
            and result.skipped[0].reason == expected_reason
        )
        got = result.skipped[0].reason if result.skipped else "measured instead"
        check(ok, f"skip: {label} -> {expected_reason} (got {got})")

    # -- blank page IS measurable when document-level matching is opted into --
    doc_level = evaluate_retrieval(
        retriever, encoder, one(good_doc, ""), top_k=5, require_page=False
    )
    check(
        doc_level.n_measured == 1
        and doc_level.results.iloc[0]["match_level"] == "document",
        "blank page + require_page=False -> measured as a document-level match",
    )

    # -- skipped rows must not dilute the aggregates -------------------------
    mixed = pd.concat([questions, one("", ""), one("ghost.pdf", "2")], ignore_index=True)
    mixed_outcome = evaluate_retrieval(retriever, encoder, mixed, top_k=5)
    check(
        mixed_outcome.n_measured == 12 and mixed_outcome.n_skipped == 2,
        f"12 measured / 2 skipped (got {mixed_outcome.n_measured}/"
        f"{mixed_outcome.n_skipped})",
    )
    check(
        mixed_outcome.summary["recall_at_5"] == outcome.summary["recall_at_5"],
        "adding unusable rows does not change Recall@5",
    )
    check(
        mixed_outcome.summary["mrr_at_5"] == outcome.summary["mrr_at_5"],
        "adding unusable rows does not change MRR@5",
    )
    check(
        int(mixed_outcome.summary["n_questions_in_file"]) == 14,
        "the file's total row count is still reported honestly",
    )
    check(len(mixed_outcome.skipped_frame()) == 2,
          "skipped rows are preserved as data, not just a count")
    check(
        set(mixed_outcome.skipped_frame()["skip_reason"])
        == {SKIP_MISSING_DOCUMENT, SKIP_UNKNOWN_DOCUMENT},
        "each skipped row carries its reason",
    )
    check(
        all(f"skipped__{r}" in mixed_outcome.summary
            for r in (SKIP_MISSING_DOCUMENT, SKIP_UNKNOWN_DOCUMENT)),
        "the summary breaks skips down by reason",
    )

    # -- an all-unusable file reports 'not measured', never 0.0 --------------
    blanked = questions.copy()
    blanked.loc[:, "expected_document"] = ""
    blank_outcome = evaluate_retrieval(retriever, encoder, blanked, top_k=5)
    check(blank_outcome.n_measured == 0, "rows with no ground truth are all skipped")
    check(
        isinstance(blank_outcome.summary["mrr_at_5"], str)
        and isinstance(blank_outcome.summary["recall_at_1"], str),
        "unmeasured metrics report 'not measured', never 0.0",
    )
    check(
        len(failure_cases(blank_outcome)) == 0,
        "skipped questions are not reported as retrieval failures",
    )

    print("\n  --- STUB-ENCODER numbers on FABRICATED documents ---")
    print("  (a code-correctness check, NOT a measurement of this system)")
    for key in ("n_measured", "n_skipped", "recall_at_1", "recall_at_3",
                "recall_at_5", "mrr_at_5"):
        print(f"    {key:<20} {outcome.summary[key]}")


def test_error_handling(settings: Settings) -> None:
    print("\n[9] error handling")
    with tempfile.TemporaryDirectory() as tmp:
        empty_dir = Path(tmp) / "nothing"
        empty_dir.mkdir()
        records, report = load_pdf_directory(empty_dir, settings.min_page_chars)
        check(records == [] and report.n_found == 0, "empty folder handled cleanly")

        records, report = load_pdf_directory(Path(tmp) / "missing", settings.min_page_chars)
        check(records == [], "missing folder handled cleanly")

        corrupt = Path(tmp) / "corrupt.pdf"
        corrupt.write_bytes(b"%PDF-1.4 this is not really a pdf at all")
        _, report = load_pdf_directory(Path(tmp), settings.min_page_chars)
        check(report.n_failed == 1, "corrupt PDF reported as a failure, not a crash")
        check(
            bool(report.problems()[0].message),
            f"failure carries an explanation: '{report.problems()[0].message[:60]}'",
        )

        zero = Path(tmp) / "zero.pdf"
        zero.write_bytes(b"")
        _, report = load_pdf_directory(Path(tmp), settings.min_page_chars)
        check(report.n_failed == 2, "0-byte PDF also reported")

    raised = False
    try:
        build_index([], np.zeros((0, 8), dtype="float32"))
    except RetrieverError:
        raised = True
    check(raised, "building an index from zero chunks raises a clear error")


def main() -> int:
    settings = Settings.from_env()

    if not list(TEST_FIXTURES_DIR.glob("*.pdf")):
        print("Generating synthetic fixtures first ...")
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        import make_test_pdfs

        make_test_pdfs.main()

    print("=" * 70)
    print("OFFLINE PIPELINE TEST  (stub encoder - no network, no model download)")
    print("=" * 70)

    test_normalization()
    test_glyph_repair()
    test_chunking()
    retriever, encoder, _chunks = test_pipeline(settings)
    test_retrieval(retriever, encoder)
    test_persistence(retriever, encoder)
    test_generation(retriever, encoder, settings)
    test_evaluation(retriever, encoder)
    test_error_handling(settings)

    print("\n" + "=" * 70)
    if _failures:
        print(f"FAILED  {len(_failures)} of {_checks} checks:")
        for failure in _failures:
            print(f"  - {failure}")
        return 1
    print(f"PASSED  all {_checks} checks")
    print("=" * 70)
    print(
        "\nReminder: this verifies the CODE. It does not measure retrieval\n"
        "quality. For that, add real papers and run scripts/run_evaluation.py."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
