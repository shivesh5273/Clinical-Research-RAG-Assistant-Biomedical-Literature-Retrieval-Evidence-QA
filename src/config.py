"""Central configuration for the Clinical Research RAG Assistant.

Everything that might reasonably change - model names, chunk sizes, top-k,
API credentials - is read from environment variables (loaded from a local
`.env` file) so that no secret and no machine-specific path is ever written
into the source code.

All paths are derived from the repository root using `pathlib`, so the project
works no matter where you clone it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# config.py lives at <repo>/src/config.py, so parents[1] is the repo root.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

DATA_DIR: Path = PROJECT_ROOT / "data"
SAMPLE_PAPERS_DIR: Path = DATA_DIR / "sample_papers"
UPLOADED_PAPERS_DIR: Path = DATA_DIR / "uploaded_papers"
TEST_FIXTURES_DIR: Path = DATA_DIR / "test_fixtures"

STORAGE_DIR: Path = PROJECT_ROOT / "storage"          # saved FAISS index + metadata
EVALUATION_DIR: Path = PROJECT_ROOT / "evaluation"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
NOTEBOOKS_DIR: Path = PROJECT_ROOT / "notebooks"

EVAL_QUESTIONS_CSV: Path = EVALUATION_DIR / "evaluation_questions.csv"
EVAL_RESULTS_CSV: Path = EVALUATION_DIR / "evaluation_results.csv"
EVAL_SUMMARY_CSV: Path = EVALUATION_DIR / "evaluation_summary.csv"

# Load .env from the repository root (no-op if the file does not exist).
load_dotenv(PROJECT_ROOT / ".env")


# ---------------------------------------------------------------------------
# Safety text (single source of truth - imported by app.py, README, report)
# ---------------------------------------------------------------------------
DISCLAIMER: str = (
    "This system is for biomedical literature retrieval and summarization only. "
    "It does not provide medical advice, diagnosis, treatment recommendations, "
    "clinical decision support, or patient-specific guidance."
)

SHORT_DISCLAIMER: str = (
    "For research-literature retrieval and summarization only. This application "
    "does not provide medical advice, diagnosis, treatment recommendations, "
    "clinical decision support, or patient-specific guidance."
)


# ---------------------------------------------------------------------------
# Helpers for reading typed environment variables
# ---------------------------------------------------------------------------
def _env_str(name: str, default: str = "") -> str:
    """Read a string env var, treating empty/whitespace values as unset."""
    raw = os.getenv(name)
    if raw is None:
        return default
    raw = raw.strip()
    return raw if raw else default


def _env_int(name: str, default: int) -> int:
    """Read an int env var, falling back to `default` if unset or malformed."""
    raw = _env_str(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


# ---------------------------------------------------------------------------
# Settings object
# ---------------------------------------------------------------------------
@dataclass
class Settings:
    """Runtime settings for the RAG pipeline.

    Construct with `Settings.from_env()` to pick up values from `.env`, or
    instantiate directly (useful in the notebook when you want to sweep
    chunk sizes or top-k without editing files).
    """

    # Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 175

    # Embedding
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: Optional[str] = None  # None => let the library decide

    # Retrieval
    top_k: int = 5

    # Generation (Anthropic). Empty strings mean "not configured".
    anthropic_api_key: str = field(default="", repr=False)
    anthropic_model: str = ""
    anthropic_max_tokens: int = 1024

    # Ingestion
    min_page_chars: int = 50  # pages with less usable text than this are skipped

    @classmethod
    def from_env(cls) -> "Settings":
        """Build a Settings object from environment variables / `.env`."""
        return cls(
            chunk_size=_env_int("CHUNK_SIZE", 1000),
            chunk_overlap=_env_int("CHUNK_OVERLAP", 175),
            embedding_model=_env_str(
                "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
            ),
            embedding_device=_env_str("EMBEDDING_DEVICE") or None,
            top_k=_env_int("TOP_K", 5),
            anthropic_api_key=_env_str("ANTHROPIC_API_KEY"),
            anthropic_model=_env_str("ANTHROPIC_MODEL"),
            anthropic_max_tokens=_env_int("ANTHROPIC_MAX_TOKENS", 1024),
            min_page_chars=_env_int("MIN_PAGE_CHARS", 50),
        )

    # -- derived properties -------------------------------------------------
    @property
    def anthropic_enabled(self) -> bool:
        """True only when BOTH an API key and a model identifier are present.

        Deliberately strict: we never guess a model name, because model
        identifiers change over time and a hard-coded one would silently rot.
        """
        return bool(self.anthropic_api_key) and bool(self.anthropic_model)

    def generation_mode(self) -> str:
        """Return 'anthropic' or 'retrieval_only'."""
        return "anthropic" if self.anthropic_enabled else "retrieval_only"


def ensure_directories() -> None:
    """Create the directories the app writes to, if they do not exist yet."""
    for directory in (
        SAMPLE_PAPERS_DIR,
        UPLOADED_PAPERS_DIR,
        TEST_FIXTURES_DIR,
        STORAGE_DIR,
        EVALUATION_DIR,
        REPORTS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
