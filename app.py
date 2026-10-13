"""Clinical Research RAG Assistant - Streamlit application.

Run with:
    streamlit run app.py

The app is a thin UI over `src/pipeline.py`; all retrieval and generation logic
lives in `src/` so that the notebook, the evaluation scripts and this app share
one code path.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import streamlit as st

from src.config import (
    SAMPLE_PAPERS_DIR,
    SHORT_DISCLAIMER,
    STORAGE_DIR,
    UPLOADED_PAPERS_DIR,
    Settings,
    ensure_directories,
)
from src.embeddings import EmbeddingModel
from src.generator import generation_status
from src.pipeline import answer_question, build_pipeline

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Clinical Research RAG Assistant",
    page_icon="🔬",
    layout="wide",
)

ensure_directories()

EXAMPLE_QUESTIONS = [
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


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_embedding_model(model_name: str, device: Optional[str]) -> EmbeddingModel:
    """Load the SentenceTransformer once per (model, device) and reuse it.

    Without this cache Streamlit would reload the encoder on every widget
    interaction, which makes the app feel broken.
    """
    return EmbeddingModel(model_name=model_name, device=device).load()


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
def init_state() -> None:
    defaults = {
        "retriever": None,
        "chunks": [],
        "stats": {},
        "ingest_report": None,
        "build_error": "",
        "indexed_source": "",
        "hits": None,
        "result": None,
        "asked_question": "",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


init_state()
settings = Settings.from_env()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def save_uploads(uploaded_files) -> List[Path]:
    """Persist Streamlit uploads to data/uploaded_papers/ and return their paths."""
    UPLOADED_PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    # Clear previous uploads so the index matches exactly what is on screen.
    for stale in UPLOADED_PAPERS_DIR.glob("*.pdf"):
        try:
            stale.unlink()
        except OSError:
            pass

    paths: List[Path] = []
    for item in uploaded_files:
        destination = UPLOADED_PAPERS_DIR / Path(item.name).name
        destination.write_bytes(item.getbuffer())
        paths.append(destination)
    return paths


def run_build(pdf_dir: Optional[Path], pdf_paths: Optional[List[Path]]) -> None:
    """Build the index and store the outcome in session state."""
    status_box = st.sidebar.empty()

    try:
        model = get_embedding_model(settings.embedding_model, settings.embedding_device)
    except Exception as exc:
        st.session_state.build_error = (
            f"Could not load the embedding model "
            f"'{settings.embedding_model}'.\n\n{exc}"
        )
        st.session_state.retriever = None
        return

    outcome = build_pipeline(
        pdf_dir=pdf_dir,
        pdf_paths=pdf_paths,
        settings=settings,
        model=model,
        progress=lambda msg: status_box.info(msg),
        save_to=STORAGE_DIR,
    )
    status_box.empty()

    st.session_state.retriever = outcome.retriever
    st.session_state.chunks = outcome.chunks
    st.session_state.stats = outcome.stats
    st.session_state.ingest_report = outcome.ingest_report
    st.session_state.build_error = outcome.error
    st.session_state.hits = None
    st.session_state.result = None


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("Clinical Research RAG Assistant")
st.caption("Biomedical Literature Retrieval & Evidence QA")

st.error(f"**Research use only.** {SHORT_DISCLAIMER}", icon="⚠️")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Corpus")

    source = st.radio(
        "Paper source",
        options=["Folder: data/sample_papers/", "Upload PDFs"],
        help=(
            "Use the folder for a fixed, reproducible corpus (recommended for "
            "evaluation), or upload PDFs for an ad-hoc session."
        ),
    )

    uploaded_files = None
    if source == "Upload PDFs":
        uploaded_files = st.file_uploader(
            "Upload open-access biomedical PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            help="Open-access papers only. Do not upload patient data.",
        )
    else:
        existing = sorted(SAMPLE_PAPERS_DIR.glob("*.pdf"))
        st.caption(f"`data/sample_papers/` contains **{len(existing)}** PDF(s).")
        if existing:
            with st.expander("Files in folder"):
                for path in existing:
                    st.write(f"- {path.name}")

    if st.button("Build / Rebuild Index", type="primary", use_container_width=True):
        if source == "Upload PDFs":
            if not uploaded_files:
                st.session_state.build_error = (
                    "No files uploaded. Choose one or more PDFs first."
                )
                st.session_state.retriever = None
            else:
                with st.spinner("Building index from uploaded PDFs ..."):
                    run_build(None, save_uploads(uploaded_files))
                st.session_state.indexed_source = "uploaded files"
        else:
            with st.spinner("Building index from data/sample_papers/ ..."):
                run_build(SAMPLE_PAPERS_DIR, None)
            st.session_state.indexed_source = "data/sample_papers/"

    st.divider()
    st.header("Index status")

    retriever = st.session_state.retriever
    if retriever is None:
        st.warning("No index built yet.")
    else:
        stats = st.session_state.stats or {}
        col_a, col_b = st.columns(2)
        col_a.metric("Papers indexed", len(retriever.documents))
        col_b.metric("Chunks", retriever.n_chunks)
        st.caption(
            f"Source: {st.session_state.indexed_source} · "
            f"mean chunk {stats.get('mean_chars', 0)} chars · "
            f"embedding dim {retriever.index.d}"
        )

    report = st.session_state.ingest_report
    if report is not None and report.problems():
        with st.expander(f"⚠️ {len(report.problems())} PDF(s) had problems"):
            for doc in report.problems():
                st.write(f"**{doc.document}** — `{doc.status}`")
                st.caption(doc.message)

    st.divider()
    st.header("Retrieval")
    top_k = st.slider(
        "Passages to retrieve (top-k)",
        min_value=1,
        max_value=15,
        value=int(settings.top_k),
        help="How many chunks are retrieved and shown to the generator.",
    )
    min_score = st.slider(
        "Minimum similarity",
        min_value=0.0,
        max_value=0.8,
        value=0.0,
        step=0.05,
        help=(
            "Cosine-similarity floor. Raise it to make the system say "
            "'not enough evidence' instead of returning weak matches."
        ),
    )

    st.divider()
    st.header("Generation")
    status = generation_status(settings)
    if status["mode"] == "anthropic":
        st.success(status["label"])
    else:
        st.info(status["label"])
    st.caption(str(status["detail"]))

    with st.expander("Active configuration"):
        st.write(
            {
                "embedding_model": settings.embedding_model,
                "embedding_device": settings.embedding_device or "auto",
                "chunk_size": settings.chunk_size,
                "chunk_overlap": settings.chunk_overlap,
                "top_k (default)": settings.top_k,
            }
        )
        st.caption("Change these in your `.env` file, then restart the app.")

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
if st.session_state.build_error:
    st.warning(st.session_state.build_error)

st.subheader("Ask a research question")

with st.expander("Example questions"):
    st.write("\n".join(f"- {q}" for q in EXAMPLE_QUESTIONS))

question = st.text_input(
    "Question",
    placeholder="What limitations did the authors report?",
    label_visibility="collapsed",
)

asked = st.button("Ask Research Question", type="primary")

if asked:
    if st.session_state.retriever is None:
        st.warning("Build the index first (sidebar → **Build / Rebuild Index**).")
    elif not question.strip():
        st.warning("Please type a question.")
    else:
        try:
            model = get_embedding_model(
                settings.embedding_model, settings.embedding_device
            )
            with st.spinner("Retrieving evidence ..."):
                hits, result = answer_question(
                    question=question,
                    retriever=st.session_state.retriever,
                    model=model,
                    settings=settings,
                    top_k=top_k,
                    min_score=min_score if min_score > 0 else None,
                )
            st.session_state.hits = hits
            st.session_state.result = result
            st.session_state.asked_question = question.strip()
        except Exception as exc:
            st.session_state.hits = None
            st.session_state.result = None
            st.error(f"Something went wrong during retrieval: {exc}")

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
result = st.session_state.result
hits = st.session_state.hits

if result is not None:
    st.divider()
    st.subheader("Evidence-Grounded Answer")
    st.caption(f"Question: *{st.session_state.asked_question}*")

    if result.error:
        st.warning(result.error)

    if not hits:
        st.info(
            "No passages passed the similarity threshold, so there is no "
            "evidence to answer from. Try lowering the minimum similarity, "
            "rephrasing the question, or indexing more papers."
        )
    else:
        if result.mode == "anthropic":
            st.success(f"Generated with `{result.model}` from retrieved evidence only.")
        st.markdown(result.answer)

        if result.unknown_citations:
            st.warning(
                "The generated answer cited source IDs that were not supplied: "
                + ", ".join(result.unknown_citations)
                + ". Treat this answer as unreliable and report it as a failure case."
            )

        st.markdown("#### Sources")
        st.code(result.sources_block(), language="text")

        st.markdown("#### Retrieved source passages")
        st.caption(
            "These are the exact passages retrieved from the indexed PDFs, "
            "before any generation. Check the answer against them."
        )
        for i, hit in enumerate(hits):
            label = (
                f"[S{i + 1}]  {hit.document} — page {hit.page}  ·  "
                f"similarity {hit.score:.3f}  ·  {hit.chunk_id}"
            )
            with st.expander(label, expanded=(i == 0)):
                st.caption(f"Paper title (best effort): {hit.title}")
                st.write(hit.text)

st.divider()
st.caption(SHORT_DISCLAIMER)
