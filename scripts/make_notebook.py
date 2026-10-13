"""Generate `notebooks/01_clinical_rag_experiments.ipynb` and a plain-text
copy of its cells (`notebooks/NOTEBOOK_CELLS.md`).

Keeping the notebook's source here means it can be regenerated from scratch
and reviewed as ordinary code in git, instead of arguing with notebook JSON
diffs. Run it whenever you want to reset the notebook:

    python scripts/make_notebook.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "01_clinical_rag_experiments.ipynb"
CELLS_MD_PATH = PROJECT_ROOT / "notebooks" / "NOTEBOOK_CELLS.md"

# (cell_type, source) where cell_type is "markdown" or "code"
Cell = Tuple[str, str]

CELLS: List[Cell] = [
    # ---------------------------------------------------------------- intro
    ("markdown", """# Clinical Research RAG Assistant — pipeline experiments

**Biomedical Literature Retrieval & Evidence QA**

> ⚠️ **Research use only.** This system is for biomedical literature retrieval and
> summarization only. It does not provide medical advice, diagnosis, treatment
> recommendations, clinical decision support, or patient-specific guidance.

This notebook walks through the RAG pipeline one stage at a time so that every
step is visible and inspectable:

`PDFs → ingest → chunk → embed → FAISS index → retrieve → generate → evaluate`

The notebook imports the **same modules the Streamlit app uses** (`src/`), so
anything you verify here is what the app actually does.

**Before running:** put 3–5 open-access biomedical PDFs in `data/sample_papers/`.
If you do not have them yet, run `python scripts/make_test_pdfs.py` and set
`USE_FIXTURES = True` in the next cell — that uses *synthetic, fabricated*
documents which exist only to prove the code runs."""),

    # --------------------------------------------------------- 1. Imports
    ("markdown", """## 1. Imports and configuration

`Settings.from_env()` reads `.env` (chunk size, embedding model, top-k, API
credentials). Nothing is hard-coded, so you can change behaviour without
editing code."""),
    ("code", '''import sys
from pathlib import Path

# Make the project root importable no matter where Jupyter was started from.
CWD = Path.cwd()
PROJECT_ROOT = CWD.parent if CWD.name == "notebooks" else CWD
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.config import (
    SAMPLE_PAPERS_DIR,
    TEST_FIXTURES_DIR,
    EVALUATION_DIR,
    DISCLAIMER,
    Settings,
)
from src.ingest import load_pdf_directory
from src.chunking import chunk_records, chunk_statistics, chunk_page_text
from src.embeddings import EmbeddingModel, embed_chunks
from src.retriever import build_index
from src.generator import generate_answer, generation_status, build_prompt

settings = Settings.from_env()

# Set to True to use the synthetic, fabricated test PDFs instead of real papers.
USE_FIXTURES = False
PAPER_DIR = TEST_FIXTURES_DIR if USE_FIXTURES else SAMPLE_PAPERS_DIR

print("project root   :", PROJECT_ROOT)
print("paper directory:", PAPER_DIR)
print("settings       :", settings)
print("generation     :", generation_status(settings))
print()
print(DISCLAIMER)'''),

    # ------------------------------------------------------- 2. Load PDFs
    ("markdown", """## 2. Load PDFs

`load_pdf_directory` uses PyMuPDF to extract text **page by page**. One record
per page — that is what makes a page-level citation honest later.

The returned `IngestReport` tells you exactly which files failed and why
(corrupt, encrypted, image-only). A scanned PDF produces no text; OCR is out of
scope for this MVP and is listed as future work."""),
    ("code", '''page_records, ingest_report = load_pdf_directory(
    PAPER_DIR, min_page_chars=settings.min_page_chars
)

print(ingest_report.as_text())

if not page_records:
    raise SystemExit(
        "No text extracted. Add PDFs to data/sample_papers/, or set "
        "USE_FIXTURES = True after running scripts/make_test_pdfs.py"
    )'''),

    # -------------------------------------------- 3. Inspect page records
    ("markdown", """## 3. Inspect the extracted pages

Always look at raw extraction before trusting anything downstream. Two-column
layouts, running headers and reference lists are the usual sources of noise."""),
    ("code", '''pages_df = pd.DataFrame(
    [
        {
            "document": r["document"],
            "page": r["page"],
            "title": str(r.get("title", ""))[:60],
            "n_chars": len(str(r["text"])),
        }
        for r in page_records
    ]
)
display(pages_df.head(15))
print("pages per document:")
print(pages_df.groupby("document")["page"].count())

print("\\n--- raw text of the first page ---\\n")
print(str(page_records[0]["text"])[:1200])'''),

    # ------------------------------------------------------- 4. Chunking
    ("markdown", """## 4. Chunk the documents

Chunking rule, in one sentence: pack paragraphs (then sentences, then word
runs) into ~`chunk_size` characters, then seed the next chunk with the previous
chunk's tail so a fact on a boundary is not split away from its context.

Chunks **never cross a page boundary**, which is what keeps `page` a truthful
citation."""),
    ("code", '''chunks = chunk_records(
    page_records,
    chunk_size=settings.chunk_size,
    chunk_overlap=settings.chunk_overlap,
)

stats = chunk_statistics(chunks)
print(stats)

chunks_df = pd.DataFrame(
    [{k: c[k] for k in ("document", "page", "chunk_id", "n_chars")} for c in chunks]
)
display(chunks_df.head(12))'''),

    # --------------------------------------------- 5. Inspect the chunks
    ("markdown", """## 5. Inspect sample chunks (and the overlap)

The cell below prints two consecutive chunks from the same page so you can see
the overlap with your own eyes. This is the single most useful debugging view
when retrieval behaves oddly."""),
    ("code", '''same_page = [c for c in chunks if c["page"] == chunks[0]["page"]
             and c["document"] == chunks[0]["document"]]

for c in same_page[:2]:
    print(f"=== {c['chunk_id']}  ({c['n_chars']} chars) ===")
    print(c["text"])
    print()

# Length distribution — a long tail of tiny chunks usually means the
# sentence splitter is tripping over something in your PDFs.
display(chunks_df["n_chars"].describe())

# Try a different configuration on a single page, without rebuilding anything:
alt = chunk_page_text(str(page_records[0]["text"]), chunk_size=500, chunk_overlap=100)
print(f"\\nSame page at chunk_size=500 -> {len(alt)} chunks "
      f"(lengths: {[len(a) for a in alt]})")'''),

    # ------------------------------------------------------ 6. Embeddings
    ("markdown", """## 6. Generate embeddings

`EmbeddingModel` wraps SentenceTransformers and guarantees **L2-normalized
float32** vectors. That normalization is what lets a FAISS inner-product index
behave as cosine similarity.

The first run downloads the model (~90 MB for MiniLM) and caches it in
`~/.cache/huggingface`."""),
    ("code", '''model = EmbeddingModel(settings.embedding_model, settings.embedding_device).load()
print("model     :", model.model_name)
print("dimension :", model.dimension)

embeddings, chunks = embed_chunks(chunks, model=model, show_progress=True)
print("matrix    :", embeddings.shape, embeddings.dtype)
print("row norms :", np.linalg.norm(embeddings, axis=1)[:5].round(4), "(should be 1.0)")'''),

    # ----------------------------------------------------- 7. FAISS index
    ("markdown", """## 7. Build the FAISS index

`IndexFlatIP` is exact brute-force inner-product search. At a few thousand
chunks it is fast and has **zero recall loss** — no approximation to explain
away. Approximate indexes (IVF, HNSW) only earn their complexity at ~10⁶
vectors, which is future work."""),
    ("code", '''retriever = build_index(chunks, embeddings, settings.embedding_model)
print(retriever)
print("documents indexed:", retriever.documents)'''),

    # -------------------------------------------------- 8. Sample retrieval
    ("markdown", """## 8. Run a sample retrieval

Every hit carries `document`, `page`, `chunk_id`, `score` and the passage text —
everything needed to cite it and to check it by hand."""),
    ("code", '''question = "What limitations did the authors report?"

hits = retriever.retrieve(model.encode_query(question), top_k=settings.top_k)

for h in hits:
    print(f"[{h.rank}] score={h.score:.4f}  {h.citation()}  ({h.chunk_id})")
    print("    ", h.text[:220].replace("\\n", " "), "...")
    print()'''),

    # -------------------------------------------- 9. Similarity inspection
    ("markdown", """## 9. Inspect similarity scores

Two things worth knowing about this embedding family:

* absolute cosine values are **not** calibrated — 0.45 is not "45% relevant";
* what matters is the *gap* between the top hit and the rest, and whether the
  score distribution for a good question differs from a nonsense one.

The nonsense-question control below is a cheap, honest sanity check."""),
    ("code", '''import matplotlib.pyplot as plt

query_vec = model.encode_query(question)
all_scores = (embeddings @ query_vec.T).ravel()

fig, ax = plt.subplots(figsize=(7, 3.2))
ax.hist(all_scores, bins=40)
for h in hits:
    ax.axvline(h.score, linestyle="--", linewidth=1)
ax.set_xlabel("cosine similarity to the query")
ax.set_ylabel("number of chunks")
ax.set_title(f"Score distribution over {len(chunks)} chunks\\n(dashed = retrieved top-k)")
plt.tight_layout()
plt.show()

print(f"top-k scores : {[round(h.score, 4) for h in hits]}")
print(f"corpus mean  : {all_scores.mean():.4f}   max: {all_scores.max():.4f}")

control = "What is the resale value of a 1998 diesel tractor?"
control_hits = retriever.retrieve(model.encode_query(control), top_k=3)
print(f"\\nControl question (should score clearly lower): {control}")
for h in control_hits:
    print(f"  [{h.rank}] {h.score:.4f}  {h.citation()}")'''),

    # -------------------------------------------- 10. Example questions
    ("markdown", """## 10. Ask example biomedical research questions

These are the demo questions the app ships with. Run them all and read the
retrieved passages — this is how you find out whether your chunk size is
working before you ever involve an LLM."""),
    ("code", '''EXAMPLE_QUESTIONS = [
    "What patient population was studied?",
    "What intervention was analyzed?",
    "What comparator was used?",
    "What were the primary outcomes?",
    "What limitations did the authors report?",
    "What dataset or cohort was used?",
    "What biomarkers were discussed?",
    "What evidence supports the conclusion?",
    "What future research did the authors suggest?",
]

rows = []
for q in EXAMPLE_QUESTIONS:
    top = retriever.retrieve(model.encode_query(q), top_k=3)
    rows.append(
        {
            "question": q,
            "top1_score": round(top[0].score, 4) if top else None,
            "top1_source": top[0].citation() if top else "",
            "top3_sources": " | ".join(h.citation() for h in top),
        }
    )

display(pd.DataFrame(rows))'''),

    # ------------------------------------------- 11. Grounded generation
    ("markdown", """## 11. Generate a grounded answer (if an API key is configured)

`generate_answer` uses the Anthropic API when both `ANTHROPIC_API_KEY` and
`ANTHROPIC_MODEL` are set in `.env`. Otherwise it returns **retrieval-only**
output: the top passages quoted verbatim, clearly labelled as not generated.

The cell first prints the exact prompt that would be sent, so the grounding
instructions are visible rather than hidden."""),
    ("code", '''status = generation_status(settings)
print(status["label"], "—", status["detail"], "\\n")

print("=" * 70)
print("PROMPT SENT TO THE MODEL (evidence truncated for display)")
print("=" * 70)
prompt = build_prompt(question, hits)
print(prompt[:2000])
print("\\n... [truncated] ...\\n")

result = generate_answer(question, hits, settings)

print("=" * 70)
print(f"MODE: {result.mode}")
if result.error:
    print("NOTE:", result.error)
print("=" * 70)
print(result.answer)'''),

    # ----------------------------------------------------- 12. Citations
    ("markdown", """## 12. Display citations

Two separate things are shown, on purpose:

1. the **Sources** list, built from retrieval metadata (never from the model);
2. a **string check** for citation markers that point at a source that was
   never supplied.

That second check catches `[S9]` when only five passages exist. It does *not*
verify that a correctly-numbered citation supports the sentence — only a human
reading the passage can say that, which is what the grounding review in
step 13 is for."""),
    ("code", '''print("Sources")
print(result.sources_block())

if result.unknown_citations:
    print("\\n⚠️  Answer cited source IDs that were never supplied:",
          ", ".join(result.unknown_citations))
else:
    print("\\nNo out-of-range citation markers found.")

print("\\n--- retrieved passages, verbatim ---")
for i, h in enumerate(hits, start=1):
    print(f"\\n[S{i}] {h.document} — page {h.page}  (similarity {h.score:.4f})")
    print(h.text[:500], "...")'''),

    # ---------------------------------------------------- 13. Evaluation
    ("markdown", """## 13. Evaluation experiment — Recall@K and MRR

Ground truth lives in `evaluation/evaluation_questions.csv` and **you fill it
in by hand** after reading your papers. Rows with an empty `expected_document`
are skipped, never scored against a guess — which means that until you fill
the file in, this cell will honestly report "0 scored".

* **Recall@K** — the fraction of questions for which a relevant passage appears
  in the top K results.
* **MRR** — mean of 1/rank of the first relevant passage (0 when none is found).

These measure *retrieval*, not answer quality."""),
    ("code", '''from src.evaluation import evaluate_retrieval, load_questions, save_outcome, failure_cases

questions_csv = EVALUATION_DIR / ("fixture_questions.csv" if USE_FIXTURES
                                  else "evaluation_questions.csv")
questions = load_questions(questions_csv)
print(f"loaded {len(questions)} question(s) from {questions_csv.name}")
display(questions.head(12))

outcome = evaluate_retrieval(
    retriever=retriever,
    model=model,
    questions=questions,
    top_k=settings.top_k,
    k_values=(1, 3, 5),
    page_tolerance=0,
)

print(outcome.as_text())

if len(outcome.results):
    display(outcome.results)
    save_outcome(
        outcome,
        EVALUATION_DIR / ("fixture_results.csv" if USE_FIXTURES else "evaluation_results.csv"),
        EVALUATION_DIR / ("fixture_summary.csv" if USE_FIXTURES else "evaluation_summary.csv"),
    )
    print("\\nResults written to evaluation/")
else:
    print("\\nNothing scored yet — fill in expected_document / expected_page "
          "in the CSV, then re-run this cell.")'''),

    # ------------------------------------------------- 14. Failure cases
    ("markdown", """## 14. Failure-case inspection

The questions where retrieval found **nothing** relevant are the most
informative output of the whole notebook. For each one, look at what *was*
retrieved and ask which of these is going on:

* the answer is spread across pages and no single chunk contains it;
* the paper phrases it in vocabulary the general-purpose encoder does not map
  to your question (the classic argument for a biomedical encoder);
* the relevant text is in a table or figure caption that extracted badly;
* the ground-truth page you recorded is simply wrong — re-check it.

Record what you find in `reports/technical_report.md`. Honest failure analysis
is worth more in a portfolio than a clean number."""),
    ("code", '''if len(outcome.results):
    misses = failure_cases(outcome)
    print(f"{len(misses)} of {len(outcome.results)} scored question(s) found "
          f"no relevant passage in top-{settings.top_k}\\n")

    for _, row in misses.iterrows():
        print("=" * 70)
        print("Q            :", row["question"])
        print("expected     :", row["expected_document"], "page", row["expected_page"])
        print("retrieved    :", row["retrieved_documents"])
        print("pages        :", row["retrieved_pages"])
        print("scores       :", row["retrieved_scores"])
        top = retriever.retrieve(model.encode_query(row["question"]), top_k=2)
        for h in top:
            print(f"\\n  [{h.rank}] {h.citation()}  ({h.score:.4f})")
            print("  ", h.text[:300].replace("\\n", " "), "...")
        print()
else:
    print("No scored questions yet — see step 13.")'''),

    # -------------------------------------------------------- 15. Wrap-up
    ("markdown", """## 15. What this notebook does and does not establish

**Established here (if the cells above ran):** the pipeline ingests PDFs,
chunks with page-level metadata, embeds, indexes, retrieves with scores, and
produces a cited answer or a clearly-labelled retrieval-only response.

**Not established here:**

* answer quality — only a manual grounding review says anything about that;
* that a biomedical encoder would do better — untested, and therefore future
  work rather than a claim;
* anything clinical. This is a literature retrieval tool, nothing more.

**Future work** (not implemented): biomedical embeddings (BioBERT /
PubMedBERT / SPECTER) compared on this same evaluation set, hybrid BM25 +
dense retrieval, cross-encoder reranking, OCR for scanned PDFs, structured PICO
extraction, PubMed / PMC ingestion.

> ⚠️ **Research use only.** This system does not provide medical advice,
> diagnosis, treatment recommendations, clinical decision support, or
> patient-specific guidance."""),
]


def build_notebook() -> dict:
    """Assemble the .ipynb JSON structure."""
    cells = []
    for cell_type, source in CELLS:
        lines = source.split("\n")
        # nbformat stores source as a list of lines, each keeping its newline.
        source_lines = [line + "\n" for line in lines[:-1]] + [lines[-1]]
        cell = {
            "cell_type": cell_type,
            "metadata": {},
            "source": source_lines,
        }
        if cell_type == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        cells.append(cell)

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def build_cells_markdown() -> str:
    """Plain-text listing of every cell, in order, for manual reconstruction."""
    out = [
        "# Notebook cells — `01_clinical_rag_experiments.ipynb`",
        "",
        "The notebook file is generated by `scripts/make_notebook.py` and is "
        "already in `notebooks/`. This listing exists so you can rebuild it by "
        "hand in PyCharm (or review it in a diff) if you ever need to.",
        "",
        "Create each block below as a new cell, in order, using the stated "
        "cell type.",
        "",
        "---",
        "",
    ]
    for i, (cell_type, source) in enumerate(CELLS, start=1):
        out.append(f"## Cell {i} — **{cell_type}**")
        out.append("")
        if cell_type == "markdown":
            out.append("````markdown")
            out.append(source)
            out.append("````")
        else:
            out.append("```python")
            out.append(source)
            out.append("```")
        out.append("")
    return "\n".join(out)


def main() -> None:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(
        json.dumps(build_notebook(), indent=1, ensure_ascii=False), encoding="utf-8"
    )
    CELLS_MD_PATH.write_text(build_cells_markdown(), encoding="utf-8")
    print(f"wrote {NOTEBOOK_PATH}  ({len(CELLS)} cells)")
    print(f"wrote {CELLS_MD_PATH}")


if __name__ == "__main__":
    main()
