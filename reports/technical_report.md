# Clinical Research RAG Assistant — Technical Report

**Biomedical Literature Retrieval & Evidence QA**

> **Research use only.** This system is for biomedical literature retrieval and
> summarization only. It does not provide medical advice, diagnosis, treatment
> recommendations, clinical decision support, or patient-specific guidance.

**Status:** MVP implementation complete and verified by offline tests.
Retrieval quality on real papers **has not been measured yet** — see §10 and §11,
where every unmeasured item is marked as such.

---

## 1. Problem

Reading a set of biomedical papers means repeatedly answering the same
structural questions — population, intervention, comparator, outcomes, cohort,
biomarkers, limitations, suggested future work — by scanning full texts.

A general-purpose language model answers these fluently, from parametric
memory, with no obligation to point at a page. For research use that is
disqualifying: an unsourced claim about a clinical paper cannot be checked, and
an unverifiable claim about clinical literature is worse than no claim.

The requirement is therefore not "a chatbot that discusses papers" but **a
retrieval system whose output is auditable at the page level**, with generation
as an optional presentation layer over evidence the user can see.

## 2. Motivation

This is a portfolio/engineering project, so two constraints shaped it beyond
functionality:

1. **Explainability over capability.** The pipeline is implemented directly on
   PyMuPDF / SentenceTransformers / FAISS rather than through LangChain or
   LlamaIndex, because every stage has to be defensible in conversation. A
   framework would have produced the same demo with less understanding.
2. **Honest scoping.** The system claims retrieval and citation. It does not
   claim clinical utility, does not claim validation, and reports unmeasured
   things as unmeasured.

## 3. Architecture

```
PDFs → ingest → chunk → embed → FAISS index → retrieve → generate → UI
```

| Module | Responsibility |
|---|---|
| `src/config.py` | Paths, env-driven settings, the disclaimer text (single source of truth) |
| `src/ingest.py` | PyMuPDF extraction, page records, failure reporting |
| `src/chunking.py` | Paragraph/sentence-aware chunking with overlap and metadata |
| `src/embeddings.py` | SentenceTransformers wrapper; L2-normalized float32 output |
| `src/retriever.py` | `faiss.IndexFlatIP`, top-k search, save/load |
| `src/generator.py` | Grounding prompt, Anthropic backend, retrieval-only fallback |
| `src/evaluation.py` | Recall@K, MRR, manual grounding review template |
| `src/pipeline.py` | Orchestration shared by the app, the notebook and the scripts |
| `app.py` | Streamlit UI |

`src/pipeline.py` exists specifically so the demo and the evaluation cannot
diverge: the app, the notebook and `run_evaluation.py` call the same functions.

## 4. Document ingestion

PyMuPDF (`page.get_text("text")`), one record per page:

```python
{"document": "paper1.pdf", "title": "...", "page": 5, "text": "..."}
```

**Normalization is deliberately minimal** — Unicode NFKC (fixes ligatures such
as `ﬁ`), rejoining words hyphenated across a line break, unwrapping single
newlines while preserving blank-line paragraph breaks, collapsing repeated
whitespace. No header/footer stripping and no reference removal: a wrong
heuristic silently deletes evidence, and silent deletion is the worst failure
mode for a system whose value proposition is auditability.

Pages with fewer than `min_page_chars` alphanumeric characters are skipped
(cover pages, figure-only pages).

**Failure handling.** Each document gets a status — `ok`, `no_text`, `empty`,
`error` — with a message, surfaced in the UI rather than swallowed:

| Condition | Behaviour |
|---|---|
| Corrupt / not a PDF | `error` with the exception type; other PDFs still load |
| Password-protected | `error`, explicitly named |
| 0 bytes / 0 pages | `empty` |
| Image-only (scanned) | `no_text`, with an explanation that OCR is out of scope |
| One unreadable page | Warned, page skipped, document still ingested |

**Title extraction** prefers PDF metadata, falls back to the longest plausible
line on page 1, and otherwise returns `None` (the filename is then used). A
title is never invented.

## 5. Chunking strategy

Target ~1000 characters, ~175 characters overlap, both configurable.

```
paragraphs → (if too long) sentences → (if still too long) word runs
        ↓
greedy packing to chunk_size
        ↓
next chunk seeded with the previous chunk's tail, snapped to a
sentence boundary where possible, otherwise a word boundary
```

Sentence splitting uses a regex with negative lookbehinds for the abbreviations
that break naive splitters in biomedical prose (`et al.`, `e.g.`, `i.e.`,
`Fig.`, `vs.`, `Eq.`, `Ref.`, `No.`, `ca.`, `cf.`).

**Chunks never cross a page boundary.** This is the design decision that makes
`paper.pdf — page 7` a fact rather than an approximation. It costs some
coherence when a paragraph straddles a page break, which is precisely the
failure mode that `--page-tolerance 1` exists to diagnose during evaluation.

Every chunk carries `document`, `title`, `page`, `chunk_id`, `text`, `n_chars`.
IDs are human-readable and stable: `paper1_p5_c2`.

A trailing fragment shorter than `min_chunk_chars` (120) is merged back into its
predecessor when the result stays within 1.3 × `chunk_size`. Nothing is ever
discarded — verified by a test asserting no content loss across a page.

## 6. Embedding model

Baseline: `sentence-transformers/all-MiniLM-L6-v2` — 384 dimensions, ~90 MB,
fast on CPU, configurable via `EMBEDDING_MODEL` in `.env`.

Chosen for the MVP because it is small enough to run on a laptop, reliable, and
widely understood. It is **not** biomedically pretrained, which is a genuine
limitation (§10) and the most obvious next experiment (§11).

All vectors are L2-normalized to unit length, with a defensive re-normalization
after encoding so the guarantee holds even if a library version ignores
`normalize_embeddings`. That single property is what makes the FAISS
inner-product score a cosine similarity.

## 7. Vector search

`faiss.IndexFlatIP` — exact, brute-force inner product.

| Considered | Decision |
|---|---|
| `IndexFlatIP` | **Chosen.** Exact, no recall loss, no parameters to tune or explain |
| `IndexIVFFlat` | Rejected. Needs training and an `nprobe` tradeoff; meaningless at 10³ vectors |
| HNSW | Rejected. Same reasoning, plus higher memory |

A corpus of 3–5 papers yields roughly 10²–10³ chunks; exact search is
sub-millisecond. Approximate indexes earn their complexity at ~10⁶ vectors, and
adopting one here would be fake sophistication.

Retrieval returns `rank`, `score`, `document`, `title`, `page`, `chunk_id`,
`text`. Default `top_k = 5`, adjustable in the UI. An optional `min_score` floor
lets the system decline to answer rather than surface weak matches.

Index and metadata persist to `storage/` (`faiss.index`, `chunks.json`,
`index_manifest.json`). A query whose dimension does not match the index raises
a specific error naming the likely cause (the index was built with a different
embedding model).

## 8. Grounded generation

Retrieved passages are labelled `[S1]`…`[Sn]` with their document, page and
chunk ID, and sent with a system prompt whose binding rules are:

1. Answer **only** from the supplied passages.
2. Do not use outside medical knowledge to fill gaps.
3. No medical advice, diagnosis, treatment recommendation or patient-specific
   guidance — if asked for one, say the system does not provide it.
4. Every factual claim ends with one or more `[S#]` markers.
5. Never invent a source ID; never invent study details or citations.
6. If the evidence is insufficient, reply with one fixed sentence:
   *"The provided papers do not contain enough evidence to answer this question."*
7. Partial evidence → answer the supported part, state what is not covered.
8. Prefer the paper's own wording for numbers and outcome definitions.

`temperature=0`. No model identifier appears in source code — it comes from
`ANTHROPIC_MODEL` in `.env`, so a model name cannot silently go stale.

**Retrieval-only fallback.** Without both `ANTHROPIC_API_KEY` and
`ANTHROPIC_MODEL`, and on *any* API failure (bad key, unknown model, rate limit,
network), the system returns the top passages **verbatim** with their scores and
citations, under an explicit "no language model was used" header. It does not
paraphrase, because paraphrasing without a model would be sentence copying
dressed up as a summary.

## 9. Citation design

Three separate mechanisms, deliberately not merged:

1. **The Sources list is built from retrieval metadata, never parsed from the
   model's output.** `[S2]` resolves to a document and page the retriever
   recorded, so a citation *cannot* be fabricated by the generator — at worst
   the model attaches the wrong (real) ID to a sentence.
2. **The retrieved passages are always displayed separately** from the answer,
   with similarity score, document, page and chunk ID. The answer is checkable
   without leaving the screen.
3. **An out-of-range citation check** flags markers pointing at a source that
   was never supplied (`[S9]` when five passages were given) and warns the user
   that the answer is unreliable.

Mechanism 3 is a **string check, not a hallucination detector**. It says nothing
about whether a correctly-numbered citation supports the sentence it follows.
Only a human reading the passage can judge that, which is the purpose of the
manual grounding review (§10).

## 10. Safety design

| Layer | Mechanism |
|---|---|
| UI | Disclaimer under the title on every screen and in the footer |
| Prompt | Explicit prohibition on advice, diagnosis, treatment recommendations, clinical decision support |
| Scope | Answers restricted to supplied passages; fixed refusal sentence when evidence is insufficient |
| Data | Open-access published literature only; no patient data, no PHI, no private records |
| Code | Disclaimer defined once in `src/config.py` and imported everywhere, so it cannot drift |
| Secrets | API key read from git-ignored `.env`; `.env.example` documents the contract |

## 11. Evaluation methodology

**Scope: retrieval only.** Whether the right passage reaches the model. Answer
quality is reviewed by hand and is not scored automatically — doing so would
need an LLM judge or human reference answers, neither of which is in the MVP.

**Ground truth** is hand-labelled in `evaluation/evaluation_questions.csv`
(`question`, `expected_document`, `expected_page`, `notes`). The shipped file has
empty label fields on purpose. Unlabelled rows are **skipped and reported as
skipped** — never scored against a guess, never counted as misses. With nothing
labelled, the summary prints *"not measured"* rather than `0.0`, so an empty run
can never be mistaken for a measured failure.

**Metrics**

* **Recall@K** (K = 1, 3, 5) — fraction of questions with a relevant passage in
  the top K.
* **MRR** — mean of 1/rank of the first relevant passage; 0 when none is found.

**Relevance rule** — document match (case-insensitive basename) AND page in
`expected_page`; blank page field means document-level matching;
`--page-tolerance N` allows ±N pages.

**Automated verification already performed** (offline, stub encoder, no network):

| Suite | Checks | Covers |
|---|---|---|
| `tests/test_pipeline_offline.py` | 67 | normalization, chunk sizing/overlap/IDs, metadata integrity, unit-norm embeddings, rank ordering, top-k clamping, min-score filtering, save/reload identity, prompt construction, sources formatting, citation checking, metric maths and monotonicity, skip-on-blank-ground-truth, and error paths (missing folder, corrupt PDF, 0-byte PDF, empty question, missing index, dimension mismatch, zero-chunk index) |
| `tests/test_app_offline.py` | 25 | the Streamlit journey via AppTest: render, disclaimer, empty state, ask-before-index, build, metrics, empty question, full answer with sources and scores, and the no-evidence path |

These verify **code correctness**. They are run with a deterministic hashed
bag-of-words stub encoder on synthetic fixtures, so any numbers they print are
not a measurement of this system's retrieval quality and are never reported as
one.

## 12. Results

**Not yet measured.**

Real Recall@K and MRR require the chosen open-access papers and hand-labelled
ground truth. After running `python scripts/run_evaluation.py`, record here:

| metric | value |
|---|---|
| Papers indexed | *to be measured* |
| Chunks indexed | *to be measured* |
| Questions scored | *to be measured* |
| Recall@1 | *to be measured* |
| Recall@3 | *to be measured* |
| Recall@5 | *to be measured* |
| MRR | *to be measured* |
| Embedding model | *to be recorded* |
| Chunk size / overlap | *to be recorded* |
| top-k / page tolerance | *to be recorded* |
| Date | *to be recorded* |

A metric reported without its configuration is not reproducible; record all rows.

## 13. Observed failures

**No failure analysis has been performed yet** — it requires a real corpus.

After running the evaluation, fill in one block per failure:

```
### Failure N — <short label>

Question:
Expected:            <document>, page <n>
Retrieved (top-5):   <documents/pages/scores>
Category:            chunk boundary | vocabulary mismatch | table/figure text |
                     reference-list retrieval | wrong ground-truth label | other
What happened:
What would fix it:
```

Do the same for answer grounding, using `evaluation/grounding_review.csv`:
question, evidence IDs supplied, generated answer, fully supported (yes/no/
partial), the unsupported sentence quoted, whether the citation was correct, and
a description. This is an engineering failure-analysis exercise on a small
sample, not a validated hallucination benchmark.

## 14. Limitations

* **Not a medical device**; no clinical validation of any kind.
* **Small corpus** (3–5 papers); untested at scale.
* **General-purpose embeddings**; no biomedical pretraining.
* **No OCR**; scanned PDFs are reported as unusable.
* **Text only**; tables, figures and equations are not parsed structurally.
* **Page-level ground truth** is coarser than passage-level judgement.
* **Retrieval metrics only**; answer quality is a small manual sample judged by
  a non-clinician.
* **Single-vector dense retrieval**; no lexical matching, so rare identifiers,
  gene symbols and exact dosages are a known weak spot for this class of model.
* **Chunks never cross pages**, so an answer split across a page break is
  structurally hard to retrieve in one chunk.
* **English only**; not tested on other languages.
* **LLM instruction-following** is the residual risk when generation is enabled;
  displaying the passages is the mitigation, not a guarantee.

## 15. Future work

In rough order of expected value per unit of effort:

1. **Biomedical embeddings** — BioBERT / PubMedBERT / SciBERT / SPECTER against
   the MiniLM baseline on the same question set and the same ground truth. This
   is the experiment the project is set up for: one `.env` change, one rerun.
2. **Chunk-size sweep** — Recall@K as a function of `chunk_size` and
   `chunk_overlap`, which the notebook already supports.
3. **Hybrid retrieval** — BM25 + dense fusion, targeting the exact-identifier
   weakness above.
4. **Cross-encoder reranking** of top-k candidates.
5. **Passage-level relevance judgements** instead of page-level.
6. **OCR** for scanned PDFs.
7. **Structured PICO extraction** (Population, Intervention, Comparator,
   Outcome, Evidence, Source) — deliberately deferred until base retrieval is
   measured.
8. **Corpus ingestion** from PubMed / PMC APIs.
9. **LangChain / LlamaIndex reimplementation** as a comparison, now that the
   hand-written baseline exists to compare against.
10. **Deployment** — containerisation, authentication, persistent shared index.

None of the above is implemented. They are listed as direction, not capability.
