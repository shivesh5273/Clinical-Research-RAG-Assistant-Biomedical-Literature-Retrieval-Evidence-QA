"""End-to-end smoke test: proves the pipeline actually runs on this machine.

Run with:
    python scripts/smoke_test.py                 # uses data/sample_papers/
    python scripts/smoke_test.py --fixtures      # uses the synthetic test PDFs

Exit code 0 = every stage succeeded. Anything else = a stage failed, and the
message says which one.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (  # noqa: E402
    SAMPLE_PAPERS_DIR,
    STORAGE_DIR,
    TEST_FIXTURES_DIR,
    Settings,
    ensure_directories,
)
from src.embeddings import EmbeddingModel  # noqa: E402
from src.generator import generation_status  # noqa: E402
from src.pipeline import answer_question, build_pipeline  # noqa: E402

DEMO_QUESTIONS = [
    "What patient population was studied?",
    "What limitations did the authors report?",
    "What biomarkers were discussed?",
]


def rule(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test the RAG pipeline.")
    parser.add_argument(
        "--fixtures",
        action="store_true",
        help="Use the synthetic test PDFs in data/test_fixtures/ instead of "
             "data/sample_papers/.",
    )
    parser.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args()

    ensure_directories()
    settings = Settings.from_env()
    top_k = args.top_k or settings.top_k

    paper_dir = TEST_FIXTURES_DIR if args.fixtures else SAMPLE_PAPERS_DIR

    rule("STAGE 0  Configuration")
    print(f"paper directory : {paper_dir}")
    print(f"embedding model : {settings.embedding_model}")
    print(f"chunk size      : {settings.chunk_size} (overlap {settings.chunk_overlap})")
    print(f"top_k           : {top_k}")
    status = generation_status(settings)
    print(f"generation      : {status['label']} — {status['detail']}")

    pdfs = sorted(paper_dir.glob("*.pdf"))
    if not pdfs:
        print(
            f"\nFAIL: no PDFs in {paper_dir}\n"
            f"  - put open-access papers in data/sample_papers/, or\n"
            f"  - run `python scripts/make_test_pdfs.py` then re-run with --fixtures"
        )
        return 1
    print(f"PDFs found      : {len(pdfs)}")

    rule("STAGE 1-4  Ingest -> chunk -> embed -> index")
    try:
        model = EmbeddingModel(settings.embedding_model, settings.embedding_device)
        outcome = build_pipeline(
            pdf_dir=paper_dir,
            settings=settings,
            model=model,
            progress=lambda m: print(f"  {m}"),
            save_to=STORAGE_DIR,
        )
    except Exception as exc:
        print(f"\nFAIL: unexpected exception during build: {type(exc).__name__}: {exc}")
        return 1

    if outcome.ingest_report is not None:
        print("\n" + outcome.ingest_report.as_text())

    if not outcome.ok:
        print(f"\nFAIL: {outcome.error}")
        return 1

    retriever = outcome.retriever
    print("\nchunk statistics:", outcome.stats)
    print("retriever       :", retriever)

    rule("STAGE 5  Retrieval + grounded answer")
    for question in DEMO_QUESTIONS:
        print(f"\nQ: {question}")
        try:
            hits, result = answer_question(
                question, retriever, model, settings, top_k=top_k
            )
        except Exception as exc:
            print(f"  FAIL: {type(exc).__name__}: {exc}")
            return 1

        if not hits:
            print("  (no passages retrieved)")
            continue

        for hit in hits[:3]:
            print(
                f"  [{hit.rank}] {hit.score:.3f}  {hit.document} p.{hit.page}  "
                f"({hit.chunk_id})"
            )
            print(f"      {hit.text[:140].replace(chr(10), ' ')} ...")

        print(f"  mode: {result.mode}")
        if result.error:
            print(f"  note: {result.error}")
        first_lines = result.answer.strip().splitlines()[:4]
        for line in first_lines:
            print(f"  | {line[:150]}")

    rule("STAGE 6  Save / reload index")
    try:
        from src.retriever import FaissRetriever

        reloaded = FaissRetriever.load(STORAGE_DIR)
        assert reloaded.n_chunks == retriever.n_chunks
        probe = reloaded.retrieve(model.encode_query(DEMO_QUESTIONS[0]), top_k=1)
        print(f"reloaded index OK: {reloaded.n_chunks} chunks, probe hit "
              f"{probe[0].chunk_id} @ {probe[0].score:.3f}")
    except Exception as exc:
        print(f"FAIL: index save/reload broken: {type(exc).__name__}: {exc}")
        return 1

    rule("RESULT")
    print("PASS - every pipeline stage ran end to end.")
    print("Note: this proves the code runs. It says nothing about retrieval")
    print("quality - run `python scripts/run_evaluation.py` for that.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
