# Clinical Research RAG Assistant
### Biomedical Literature Retrieval & Evidence QA

A domain-specific Retrieval-Augmented Generation (RAG) system for biomedical and
clinical research literature. Index a small set of open-access papers, ask
research-oriented questions, retrieve the source passages, and get an answer
that cites the paper and page every claim came from.

---

## ⚠️ Safety disclaimer

> **This system is for biomedical literature retrieval and summarization only.
> It does not provide medical advice, diagnosis, treatment recommendations,
> clinical decision support, or patient-specific guidance.**

The disclaimer is shown on every screen of the application, is enforced in the
generation prompt, and is defined in exactly one place (`src/config.py`) so it
cannot drift.

This project uses **open-access published literature only**. It contains no
patient data, no private medical records, and no protected health information,
and it must not be used with any.

---

## 1. Project overview

| | |
|---|---|
| **What it does** | Answers questions about a corpus of biomedical PDFs, grounded in retrieved passages, with paper + page citations |
| **What it is not** | A medical device, a clinical decision-support tool, or a validated system of any kind |
| **Core stack** | PyMuPDF · SentenceTransformers · FAISS · NumPy · pandas · Streamlit |
| **Generation** | Anthropic API when configured; otherwise an explicit retrieval-only mode |
| **Framework choice** | The RAG pipeline is implemented directly, **without** LangChain or LlamaIndex, so every stage is inspectable |

---

## 2. Problem statement

A researcher reading a set of papers repeatedly asks the same structural
questions: who was studied, what was done, what was compared, what was measured,
what did the authors themselves say was wrong with it. Answering those questions
means scanning full texts by hand.

A general-purpose chatbot answers them fluently and sometimes wrongly, because
it draws on parametric memory and has no obligation to point at a page. For
research use that failure mode is disqualifying: an unsourced claim about a
clinical paper is worse than no claim.

This project takes the opposite stance. The retrieval layer is the product; the
language model is an optional formatting step over evidence that has already
been selected and is displayed to the user alongside the answer. If no model is
available, the system still works — it just stops paraphrasing and shows you the
passages.

**Design commitments**

1. Every factual claim carries a `[S#]` marker resolving to a document and page.
2. The retrieved passages are always displayed, separately from the answer, so
   the answer can be checked rather than trusted.
3. When the evidence does not answer the question, the system says so verbatim
   instead of improvising.
4. No LLM is required for the system to be useful.

---

## 3. Architecture

```
data/sample_papers/*.pdf
          │
          ▼
  ┌───────────────┐   PyMuPDF, page by page; page boundaries preserved
  │  src/ingest   │   image-only / corrupt / encrypted PDFs reported, not skipped silently
  └───────┬───────┘
          │  page records: {document, title, page, text}
          ▼
  ┌───────────────┐   paragraph → sentence → word-run packing,
  │ src/chunking  │   ~1000 chars with ~175 char overlap, never across a page
  └───────┬───────┘
          │  chunks: {document, title, page, chunk_id, text, n_chars}
          ▼
  ┌───────────────┐   SentenceTransformers, L2-normalized float32
  │src/embeddings │   model configurable via .env
  └───────┬───────┘
          │  (n_chunks × dim) matrix, row i ↔ chunks[i]
          ▼
  ┌───────────────┐   faiss.IndexFlatIP — exact search;
  │ src/retriever │   normalized vectors ⇒ inner product == cosine similarity
  └───────┬───────┘
          │  top-k hits with score + full metadata
          ▼
  ┌───────────────┐   strict grounding prompt, [S1]..[Sn] evidence labels
  │ src/generator │   Anthropic API  ──or──  retrieval-only fallback
  └───────┬───────┘
          │
          ▼
   app.py (Streamlit)   answer · Sources list · raw passages · scores
```

`src/pipeline.py` wires these stages together so that the Streamlit app, the
notebook and the evaluation scripts all execute the **same** code path.

### Why these choices

| Decision | Reason |
|---|---|
| Page-level records | A chunk that spans pages cannot be cited honestly. Page boundaries are never crossed. |
| Hand-written chunker | Chunking dominates retrieval quality; hiding it behind a framework makes failures unexplainable. |
| L2-normalized vectors | Lets `IndexFlatIP` act as exact cosine similarity — the score in the UI means what it says. |
| `IndexFlatIP`, not IVF/HNSW | Exact search, zero recall loss, no tuning. Approximate indexes matter at ~10⁶ vectors, not 10³. |
| Generation is optional | The retrieval layer is the contribution. An unavailable API key degrades the output, it does not break the system. |
| No LangChain / LlamaIndex | Every stage is a readable function you can step through in a debugger and explain out loud. |

---

## 4. RAG pipeline in detail

**Ingestion** (`src/ingest.py`) extracts text page by page with PyMuPDF.
Normalization is conservative: Unicode NFKC, rejoining words hyphenated across
a line break, unwrapping single newlines while preserving paragraph breaks.
Nothing is deleted — no header stripping, no reference removal — because a wrong
guess silently destroys evidence. Pages with fewer than `min_page_chars`
alphanumeric characters are skipped as figure/blank pages. Corrupt, encrypted,
empty and image-only PDFs each produce a specific status in an `IngestReport`
that the UI displays.

**Chunking** (`src/chunking.py`) packs paragraphs — falling back to sentences,
then word runs — into ~`chunk_size` characters, then seeds each new chunk with
the tail of the previous one (~`chunk_overlap` characters, snapped to a sentence
or word boundary). Both values are configurable. Chunks never cross a page.

**Embedding** (`src/embeddings.py`) wraps SentenceTransformers and guarantees
L2-normalized float32 output. Baseline model: `all-MiniLM-L6-v2`
(384-dimensional, ~90 MB, CPU-friendly). Biomedical encoders are *future work*,
not a claim.

**Retrieval** (`src/retriever.py`) builds a `faiss.IndexFlatIP` and returns
ranked hits carrying `document`, `title`, `page`, `chunk_id`, `score` and text.
An optional `min_score` floor lets the app answer "no relevant evidence" instead
of surfacing weak matches. Index and metadata can be saved and reloaded.

**Generation** (`src/generator.py`) labels each passage `[S1]`…`[Sn]` and sends
a strict grounding prompt: answer only from the evidence, no outside medical
knowledge, no advice, no diagnosis, no treatment recommendations, cite a source
ID for every claim, never invent a citation, and if the evidence is insufficient
reply with one fixed sentence. The **Sources** list is built from retrieval
metadata — never from the model's output, so a citation cannot be fabricated.

---

## 5. Project structure

```
clinical-research-rag-assistant/
├── README.md
├── requirements.txt
├── .env.example                 # copy to .env; never commit .env
├── .gitignore
├── app.py                       # Streamlit application
│
├── src/
│   ├── __init__.py
│   ├── config.py                # paths, settings, disclaimer (single source of truth)
│   ├── ingest.py                # PDF → page records
│   ├── chunking.py              # page records → overlapping chunks + metadata
│   ├── embeddings.py            # chunks → normalized vectors
│   ├── retriever.py             # FAISS index + top-k search
│   ├── generator.py             # grounded prompt, Anthropic + retrieval-only modes
│   ├── evaluation.py            # Recall@K, MRR, manual review template
│   └── pipeline.py              # wires the stages together (used by app + notebook)
│
├── notebooks/
│   ├── 01_clinical_rag_experiments.ipynb
│   └── NOTEBOOK_CELLS.md        # the notebook's cells as plain text
│
├── scripts/
│   ├── make_test_pdfs.py        # generates SYNTHETIC fixtures (not research)
│   ├── make_notebook.py         # regenerates the notebook from source
│   ├── smoke_test.py            # end-to-end pipeline check
│   └── run_evaluation.py        # computes real Recall@K and MRR
│
├── tests/
│   ├── test_pipeline_offline.py # 67 checks, no network required
│   └── test_app_offline.py      # drives the Streamlit app end to end
│
├── data/
│   ├── sample_papers/           # ← YOUR open-access PDFs go here
│   └── test_fixtures/           # synthetic, fabricated test documents
│
├── evaluation/
│   ├── evaluation_questions.csv # ground truth — YOU fill this in by hand
│   ├── evaluation_results.csv   # generated by scripts/run_evaluation.py
│   ├── fixture_questions.csv    # ground truth for the synthetic fixtures
│   └── grounding_review.csv     # manual answer-grounding review (generated)
│
├── reports/
│   └── technical_report.md
├── screenshots/
└── storage/                     # saved FAISS index (git-ignored)
```

---

## 6. Installation (macOS)

```bash
# 1. Get the project and enter it
cd ~/PycharmProjects           # or wherever you keep projects
cd clinical-research-rag-assistant

# 2. Create a virtual environment
#    Verified on Python 3.13 (macOS, Apple Silicon). 3.10+ should work.
python3 -m venv .venv

# 3. Activate it
source .venv/bin/activate

# 4. Upgrade pip, then install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 5. Create your .env from the template
cp .env.example .env
```

Open `.env` and either leave the Anthropic fields blank (retrieval-only mode) or
fill them in:

```
ANTHROPIC_API_KEY=sk-ant-...            # from console.anthropic.com
ANTHROPIC_MODEL=<current model id>      # look up a current identifier in the Anthropic docs
```

**In PyCharm:** *Settings → Project → Python Interpreter → Add Interpreter →
Existing → `.venv/bin/python`*. Then *Settings → Project → Project Structure* →
mark the project root as **Sources Root** so `from src...` resolves.

**Regenerating `requirements.txt`** after you add a package:

```bash
pip freeze > requirements.lock.txt      # exact, reproducible pins
```

Keep `requirements.txt` as the human-readable list and the lock file as the
exact record.

---

## 7. How to run

### Verify the install (no PDFs and no network needed)

```bash
source .venv/bin/activate
python scripts/make_test_pdfs.py        # synthetic fixtures
python tests/test_pipeline_offline.py   # pipeline logic
python tests/test_app_offline.py        # Streamlit app end to end
```

### Add your papers

Put **3–5 open-access biomedical PDFs** (e.g. from PubMed Central) into
`data/sample_papers/`. Use text-based (born-digital) PDFs — scanned documents
produce no text and OCR is not part of this MVP. Do not use paywalled papers and
do not use any document containing patient data.

### Smoke-test the real pipeline

```bash
python scripts/smoke_test.py            # uses data/sample_papers/
python scripts/smoke_test.py --fixtures # uses the synthetic documents
```

The first run downloads the embedding model (~90 MB) into
`~/.cache/huggingface`. Later runs are offline.

### Run the application

```bash
streamlit run app.py
```

It opens at `http://localhost:8501`. In the sidebar, choose the paper source and
press **Build / Rebuild Index**, then ask a question.

### Jupyter (inside PyCharm or standalone)

```bash
python -m ipykernel install --user --name clinical-rag --display-name "Clinical RAG"
jupyter notebook notebooks/01_clinical_rag_experiments.ipynb
```

In PyCharm you can simply open the `.ipynb` and select the `.venv` interpreter.

---

## 8. Example questions

```
What patient population was studied?
What intervention was analyzed?
What comparator was used?
What were the primary outcomes?
What limitations did the authors report?
What dataset or cohort was used?
What biomarkers were discussed?
What evidence supports the conclusion?
What future research did the authors suggest?
```

Answers come only from the indexed PDFs. Questions the papers do not address
return the fixed sentence *"The provided papers do not contain enough evidence
to answer this question."*

---

## 9. Example output

The shape of a response (content depends entirely on your corpus):

```
Evidence-Grounded Answer

The study included <N> participants with <condition> [S1]. The primary
outcome was <outcome> measured at <timepoint> [S2]. The authors reported
several limitations, including <limitation> [S3].

Sources
[S1] paper_name.pdf — page 4
[S2] paper_name.pdf — page 7
[S3] other_paper.pdf — page 12

Retrieved source passages
▸ [S1]  paper_name.pdf — page 4  ·  similarity 0.612  ·  paper_name_p4_c2
▸ [S2]  paper_name.pdf — page 7  ·  similarity 0.588  ·  paper_name_p7_c1
...
```

> No example output is committed to this repository yet, because doing so before
> running the system on real papers would mean inventing it. Replace this block
> with a real transcript and add a screenshot to `screenshots/` after your first
> successful run.

---

## 10. Evaluation methodology

**What is measured:** retrieval quality — does the system put the right passage
in front of the model? **What is not measured automatically:** answer quality.
That is assessed by a manual grounding review, described below.

### Ground truth

`evaluation/evaluation_questions.csv` ships with ~12 research questions and
**empty ground-truth fields**. You fill in `expected_document` and
`expected_page` by hand after reading your papers. Rows left blank are *skipped*
by the evaluator and reported as skipped — never scored against a guess, and
never counted as a miss.

| column | meaning |
|---|---|
| `question` | the research question |
| `expected_document` | exact PDF filename holding the answer |
| `expected_page` | page number as shown in a PDF reader (1-based); `4;5` for two pages; blank = document-level matching |
| `notes` | your justification for the label |

### Metrics

* **Recall@K** — fraction of *measured* questions for which a relevant passage
  appears in the top K results. Reported at K = 1, 3, 5.
* **MRR@depth** — mean of 1/rank of the first relevant passage, 0 when none is
  found. Reciprocal rank is only observable within the depth actually
  searched, so it is reported as `mrr_at_5` (for `top_k=5`), never a bare
  `mrr`. Asking for a K deeper than `top_k` is an error, not a silently
  truncated number.

Per question the outcome is a binary **hit** (`hit_at_1/3/5` in the results
CSV); **recall** is the aggregate mean over questions.

A retrieved chunk counts as relevant when its document matches
`expected_document` **and** its page is in `expected_page`
(`--page-tolerance N` widens this when a claim straddles a page break).

### Measured vs skipped

A question is **measured** only if its ground truth is usable. Otherwise it is
**skipped**, with a reason, and excluded from every aggregate — a skip is never
counted as a retrieval failure, and never silently becomes a zero:

| skip reason | meaning |
|---|---|
| `missing_expected_document` | no document recorded |
| `missing_expected_page` | no page recorded, under page-grounded rules (the default) |
| `malformed_expected_page` | unparseable page value (`p4`, `4-5`, `four`) — always a skip, in any mode |
| `expected_document_not_in_index` | filename typo, or that paper is not indexed |
| `expected_page_not_in_index` | that page was never indexed (ingestion skips near-empty pages), so it can never be retrieved |

A blank page field does **not** quietly degrade to document-level matching.
Pass `--allow-document-level` to opt into that explicitly; malformed values are
still skipped.

### Running it

```bash
python scripts/run_evaluation.py                     # your corpus
python scripts/run_evaluation.py --top-k 10          # deeper retrieval
python scripts/run_evaluation.py --page-tolerance 1  # allow ±1 page
python scripts/run_evaluation.py --fixtures          # synthetic corpus
```

Writes three files:

* `evaluation/evaluation_results.csv` — one row per **measured** question, with
  the retrieved documents, pages, chunk IDs, scores, first relevant rank,
  `hit_at_1/3/5` flags and reciprocal rank;
* `evaluation/evaluation_summary.csv` — Recall@K, MRR@depth, the run
  configuration, and the skip breakdown;
* `evaluation/evaluation_skipped.csv` — every skipped row and its reason,
  preserved rather than reduced to a count.

### Answer grounding review (manual)

`scripts/run_evaluation.py` also creates `evaluation/grounding_review.csv`. For
each generated answer you record, by hand:

| column | |
|---|---|
| `question` | the question asked |
| `retrieved_evidence_ids` | which `[S#]` passages were supplied |
| `generated_answer` | the answer text |
| `fully_supported` | yes / no / partial |
| `unsupported_claim` | quote the specific sentence |
| `citation_correct` | does `[S2]` actually support the sentence it follows? |
| `failure_description` | what went wrong |

The code performs one automatic check only: a **string check** for citation
markers pointing at a source that was never supplied (e.g. `[S9]` when five
passages were given). That is not hallucination detection, and this project does
not claim to have built one.

---

## 11. Results

**No results have been generated yet.**

Real numbers require your chosen papers and your hand-labelled ground truth.
Once you have run `python scripts/run_evaluation.py`, paste the contents of
`evaluation/evaluation_summary.csv` here:

| metric | value |
|---|---|
| Number of test questions | *to be measured* |
| Recall@1 | *to be measured* |
| Recall@3 | *to be measured* |
| Recall@5 | *to be measured* |
| MRR | *to be measured* |
| Embedding model | *to be recorded* |
| Chunk size / overlap | *to be recorded* |

Record alongside them: number of papers, number of chunks, top-k, page
tolerance, and the date. A metric without its configuration is not reproducible.

### Verification status

Last updated: **2026-09-23**. Full run-by-run detail in
[`reports/verification_log.md`](reports/verification_log.md).

**Verified**

| | evidence |
|---|---|
| Environment installs cleanly | Python 3.13 venv on macOS; `pip install -r requirements.txt` succeeded; `python -m pip check` → *No broken requirements found* |
| Pipeline logic (ingest → chunk → embed → index → retrieve → cite) | `tests/test_pipeline_offline.py` — 67/67 checks (offline stub encoder) |
| Streamlit app end-to-end journey | `tests/test_app_offline.py` — 25/25 checks via Streamlit's AppTest harness |
| Error handling (no PDFs, corrupt PDF, 0-byte PDF, empty question, missing index, dimension mismatch, no API key) | covered by the two suites above — 92/92 total |
| Real embedding model loads and runs | `all-MiniLM-L6-v2` downloaded from Hugging Face and produced real embeddings; dimension confirmed **384** |
| Real FAISS `IndexFlatIP` build, search, save and reload | `scripts/smoke_test.py --fixtures` → *PASS - every pipeline stage ran end to end* |
| Retrieval is semantically sensible | on the synthetic fixture questions only (3 fixture PDFs → 14 chunks) |

**Not verified**

| | what is still needed |
|---|---|
| Any real biomedical corpus | no open-access papers added to `data/sample_papers/` yet |
| Ingestion of real paper PDFs (two-column layouts, reference lists, tables) | run `python scripts/smoke_test.py` once papers are in place |
| Streamlit against real papers | run `streamlit run app.py` with a real corpus |
| Recall@1 / Recall@3 / Recall@5 / MRR | ground truth not labelled; `scripts/run_evaluation.py` not yet run on real papers |
| Anthropic generation path | app is in retrieval-only mode; no `ANTHROPIC_API_KEY` / `ANTHROPIC_MODEL` configured |
| Answer grounding / failure analysis | requires generated answers, which requires the above |
| Screenshots and demo material | `screenshots/` is empty |
| GitHub publication | repository not published |

> **Current honest summary:** the pipeline has been shown to run end to end with
> the real embedding model, on *synthetic test fixtures only*. It has not been
> exercised on real biomedical literature, and no retrieval quality has been
> measured. This is not a completed MVP by the project's own definition of done.

---

## 12. Failure cases

*To be completed after running the system on real papers.* Record them in
`reports/technical_report.md` and summarise here. Expected categories, based on
how this class of system is built rather than on measurements taken:

* answers spread across pages, so no single chunk contains the full answer;
* vocabulary mismatch between a lay question and clinical phrasing (the standard
  argument for a biomedical encoder — untested here);
* tables and figure captions extracting as disordered text;
* reference lists retrieved instead of body text;
* ground-truth labels that turn out to be wrong on re-reading.

---

## 13. Limitations

* **Not a medical device.** No clinical validation of any kind.
* **Small corpus.** Designed for 3–5 papers; not tested at scale.
* **General-purpose embeddings.** `all-MiniLM-L6-v2` is not biomedically trained.
* **No OCR.** Scanned/image-only PDFs yield no text and are reported as such.
* **Text only.** Tables, figures and equations are not parsed structurally.
* **Retrieval metrics only.** Answer quality is reviewed by hand, on a small
  sample, by a non-clinician.
* **Single-vector dense retrieval.** No BM25, no reranking, no query expansion.
* **Page-level ground truth** is coarser than passage-level relevance judgement.
* **LLM dependency is optional but real.** With a model configured, the answer
  is only as good as the model's instruction-following. The displayed passages
  exist so the reader can catch it when that fails.

---

## 14. Future work

Not implemented — listed as direction, not as capability:

* **Biomedical embeddings** — compare BioBERT / PubMedBERT / SciBERT / SPECTER
  against the MiniLM baseline on this same evaluation set.
* **Hybrid retrieval** — BM25 + dense fusion for exact identifiers and rare terms.
* **Cross-encoder reranking** of the top-k candidates.
* **OCR** for scanned PDFs.
* **Structured PICO extraction** (Population, Intervention, Comparator, Outcome,
  Evidence, Source) — deliberately deferred until the base pipeline is measured.
* **PubMed / PMC / ClinicalTrials.gov ingestion.**
* **Passage-level relevance judgements** instead of page-level.
* **LangChain / LlamaIndex** as an alternative implementation to compare against.
* **Deployment** — containerisation, authentication, persistent multi-user index.

---

## 15. Screenshots / demo

`screenshots/` is currently empty. After your first successful run, capture:

1. The app with the disclaimer and an empty index.
2. The sidebar after indexing, showing papers and chunk counts.
3. A full answer with the Sources list.
4. An expanded source passage with its similarity score.
5. A question the papers cannot answer, showing the insufficient-evidence reply.

Then reference them here.

---

## 16. Reproducibility

* No secrets in source. `ANTHROPIC_API_KEY` / `ANTHROPIC_MODEL` come from `.env`,
  which is git-ignored; `.env.example` documents the contract.
* No hard-coded paths. Everything derives from the repo root via `pathlib`.
* No hard-coded model identifier. The Anthropic model comes from `.env` so it
  cannot silently go stale.
* Deterministic generation (`temperature=0`).
* Offline tests, so a reviewer can verify the pipeline without a key or a
  network connection.

## Licensing note on papers

`data/sample_papers/*.pdf` is git-ignored by default. Open-access papers are
generally redistributable under their licence (commonly CC-BY), but licences
differ per paper — check each one before committing it, and record the source
URL and licence for every paper you use.
