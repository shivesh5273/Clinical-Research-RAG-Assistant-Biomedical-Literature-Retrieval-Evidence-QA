"""Run the retrieval evaluation and write real (measured) metrics to CSV.

Run with:
    python scripts/run_evaluation.py
    python scripts/run_evaluation.py --fixtures        # synthetic test corpus
    python scripts/run_evaluation.py --top-k 10 --page-tolerance 1

Outputs:
    evaluation/evaluation_results.csv    one row per scored question
    evaluation/evaluation_summary.csv    Recall@K and MRR

Nothing is written until the retrieval actually runs, and questions with an
empty `expected_document` are skipped rather than scored against a guess.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (  # noqa: E402
    EVAL_QUESTIONS_CSV,
    EVAL_RESULTS_CSV,
    EVAL_SUMMARY_CSV,
    EVALUATION_DIR,
    SAMPLE_PAPERS_DIR,
    TEST_FIXTURES_DIR,
    Settings,
    ensure_directories,
)
from src.embeddings import EmbeddingModel  # noqa: E402
from src.evaluation import (  # noqa: E402
    create_grounding_review_template,
    evaluate_retrieval,
    failure_cases,
    load_questions,
    save_outcome,
)
from src.pipeline import build_pipeline  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality.")
    parser.add_argument("--fixtures", action="store_true",
                        help="Evaluate against the synthetic test PDFs.")
    parser.add_argument("--questions", type=Path, default=None,
                        help="Path to the question CSV.")
    parser.add_argument("--papers", type=Path, default=None,
                        help="Path to the folder of PDFs to index.")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--page-tolerance", type=int, default=0,
                        help="Allow a match within N pages of the expected page.")
    parser.add_argument(
        "--allow-document-level",
        action="store_true",
        help="Permit rows with a blank expected_page to match on document only. "
             "Off by default: a page-grounded evaluation requires page ground "
             "truth, and a blank page field is skipped rather than silently "
             "relaxed. Malformed page values are skipped either way.",
    )
    parser.add_argument("--results", type=Path, default=None)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--skipped", type=Path, default=None)
    args = parser.parse_args()

    ensure_directories()
    settings = Settings.from_env()
    top_k = args.top_k or settings.top_k

    if args.fixtures:
        paper_dir = args.papers or TEST_FIXTURES_DIR
        questions_csv = args.questions or (EVALUATION_DIR / "fixture_questions.csv")
        results_csv = args.results or (EVALUATION_DIR / "fixture_results.csv")
        summary_csv = args.summary or (EVALUATION_DIR / "fixture_summary.csv")
        skipped_csv = args.skipped or (EVALUATION_DIR / "fixture_skipped.csv")
    else:
        paper_dir = args.papers or SAMPLE_PAPERS_DIR
        questions_csv = args.questions or EVAL_QUESTIONS_CSV
        results_csv = args.results or EVAL_RESULTS_CSV
        summary_csv = args.summary or EVAL_SUMMARY_CSV
        skipped_csv = args.skipped or (EVALUATION_DIR / "evaluation_skipped.csv")

    print(f"papers    : {paper_dir}")
    print(f"questions : {questions_csv}")
    print(f"top_k     : {top_k}  (page tolerance {args.page_tolerance})")
    print(f"matching  : {'document-level allowed' if args.allow_document_level else 'page-grounded (strict)'}")

    # ---- Load the question set first: fail fast on a bad ground-truth file.
    try:
        questions = load_questions(questions_csv)
    except Exception as exc:
        print(f"\nFAIL: {exc}")
        return 1
    print(f"loaded {len(questions)} question(s)")

    # ---- Build the index ---------------------------------------------------
    model = EmbeddingModel(settings.embedding_model, settings.embedding_device)
    outcome = build_pipeline(
        pdf_dir=paper_dir, settings=settings, model=model,
        progress=lambda m: print(f"  {m}"),
    )
    if not outcome.ok:
        print(f"\nFAIL: {outcome.error}")
        return 1
    print(f"indexed {outcome.retriever.n_chunks} chunks from "
          f"{len(outcome.retriever.documents)} document(s)")

    # ---- Evaluate ----------------------------------------------------------
    evaluation = evaluate_retrieval(
        retriever=outcome.retriever,
        model=model,
        questions=questions,
        top_k=top_k,
        page_tolerance=args.page_tolerance,
        require_page=not args.allow_document_level,
    )

    print("\n" + evaluation.as_text())

    if not len(evaluation.results):
        print(
            "\nNo questions could be MEASURED (see skip reasons above). Fix the "
            "ground truth in the question CSV, then run this again. "
            "No results file was written."
        )
        return 1

    save_outcome(evaluation, results_csv, summary_csv, skipped_csv)
    print(f"\nwrote {results_csv}")
    print(f"wrote {summary_csv}")
    print(f"wrote {skipped_csv}  ({evaluation.n_skipped} skipped row(s))")

    print("\nPer-question outcome (measured only):")
    for _, row in evaluation.results.iterrows():
        rank = row["first_relevant_rank"] or "MISS"
        print(f"  rank {str(rank):>4}  |  {row['expected_document'][:38]:<38} "
              f"p.{row['expected_page']:<6} | {row['question'][:52]}")

    misses = failure_cases(evaluation)
    if len(misses):
        print(f"\n{len(misses)} MEASURED question(s) retrieved NO relevant passage "
              f"in top-{evaluation.summary['search_depth']} — these are genuine "
              f"failure cases (skipped rows are not counted here):")
        for question in misses["question"]:
            print(f"  - {question}")

    template = create_grounding_review_template(EVALUATION_DIR / "grounding_review.csv")
    print(f"\nmanual answer-grounding review template: {template}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
