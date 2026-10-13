"""Embeddings: chunk text -> normalized dense vectors (SentenceTransformers).

Key detail for the rest of the pipeline
---------------------------------------
Every vector produced here is L2-normalized. That single choice is what lets
FAISS's inner-product index behave exactly like cosine similarity, so the
scores you see in the UI are cosine similarities in [-1, 1] (in practice
roughly [0, 1] for this model family).

The model identifier is configurable. The MVP baseline is a small
general-purpose encoder; swapping in a biomedical encoder (BioBERT /
PubMedBERT / SPECTER) is a one-line change and is listed as future work
precisely because it has not been measured yet.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingError(RuntimeError):
    """Raised when the embedding model cannot be loaded or used."""


class EmbeddingModel:
    """Thin, explicit wrapper around a SentenceTransformer encoder.

    The wrapper exists for three reasons:
      1. It loads the heavy library lazily, so importing `src.embeddings`
         is cheap and gives a clear error message if the install is broken.
      2. It guarantees float32, L2-normalized output everywhere.
      3. It gives the rest of the code one stable interface, which makes
         swapping the encoder (future work) a single-file change.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, device: Optional[str] = None):
        self.model_name = model_name
        self.device = device
        self._model = None  # loaded on first use

    # -- loading ------------------------------------------------------------
    def load(self) -> "EmbeddingModel":
        """Load the underlying model. Safe to call repeatedly."""
        if self._model is not None:
            return self

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EmbeddingError(
                "sentence-transformers is not installed. Run:\n"
                "    pip install sentence-transformers"
            ) from exc

        try:
            self._model = SentenceTransformer(self.model_name, device=self.device)
        except Exception as exc:
            raise EmbeddingError(
                f"Could not load embedding model '{self.model_name}'.\n"
                f"Reason: {type(exc).__name__}: {exc}\n"
                "Common causes: no internet connection on first run (the model "
                "is downloaded once and then cached in ~/.cache/huggingface), "
                "or a typo in EMBEDDING_MODEL in your .env file."
            ) from exc
        return self

    @property
    def dimension(self) -> int:
        """Embedding dimensionality (384 for all-MiniLM-L6-v2)."""
        self.load()
        return int(self._model.get_sentence_embedding_dimension())

    # -- encoding -----------------------------------------------------------
    def encode(
        self,
        texts: Sequence[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Encode texts into an L2-normalized float32 matrix.

        Returns an array of shape `(len(texts), dimension)`.
        """
        self.load()
        if len(texts) == 0:
            return np.zeros((0, self.dimension), dtype="float32")

        try:
            vectors = self._model.encode(
                list(texts),
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,   # <- cosine via inner product
                show_progress_bar=show_progress,
            )
        except Exception as exc:
            raise EmbeddingError(
                f"Embedding failed ({type(exc).__name__}: {exc})."
            ) from exc

        vectors = np.asarray(vectors, dtype="float32")
        # Defensive re-normalization: guarantees unit norm even if a future
        # model or library version ignores `normalize_embeddings`.
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (vectors / norms).astype("float32")

    def encode_query(self, question: str) -> np.ndarray:
        """Encode a single question into shape `(1, dimension)`."""
        return self.encode([question], batch_size=1, show_progress=False)


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------
def embed_chunks(
    chunks: Sequence[Dict[str, object]],
    model: Optional[EmbeddingModel] = None,
    model_name: str = DEFAULT_MODEL,
    device: Optional[str] = None,
    batch_size: int = 32,
    show_progress: bool = True,
) -> Tuple[np.ndarray, List[Dict[str, object]]]:
    """Embed a list of chunk dicts.

    Returns `(embedding_matrix, chunks)`. The chunk list is returned unchanged
    and in the same order as the matrix rows - row *i* of the matrix is the
    embedding of `chunks[i]`. That positional contract is what the FAISS
    retriever relies on to map a hit back to its metadata.
    """
    if not chunks:
        raise ValueError("No chunks to embed - check ingestion and chunking first.")

    encoder = model or EmbeddingModel(model_name=model_name, device=device)
    texts = [str(c["text"]) for c in chunks]
    matrix = encoder.encode(texts, batch_size=batch_size, show_progress=show_progress)

    if matrix.shape[0] != len(chunks):
        raise EmbeddingError(
            f"Embedding count mismatch: {matrix.shape[0]} vectors for "
            f"{len(chunks)} chunks."
        )
    return matrix, list(chunks)


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity between two sets of already-normalized vectors.

    Used in the notebook for sanity checks (e.g. "are near-duplicate chunks
    actually scoring high?"). With normalized inputs this is just a dot product.
    """
    return np.asarray(a, dtype="float32") @ np.asarray(b, dtype="float32").T


# ---------------------------------------------------------------------------
# Manual check:  python -m src.embeddings
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from src.config import Settings

    settings = Settings.from_env()
    model = EmbeddingModel(settings.embedding_model, settings.embedding_device).load()
    sample = [
        "The trial enrolled 842 adult participants with type 2 diabetes.",
        "Eight hundred forty-two adults with T2D were randomised.",
        "The catalytic converter reduces nitrogen oxide emissions.",
    ]
    vecs = model.encode(sample)
    print(f"model      : {settings.embedding_model}")
    print(f"dimension  : {model.dimension}")
    print(f"matrix     : {vecs.shape}, dtype={vecs.dtype}")
    print(f"row norms  : {np.linalg.norm(vecs, axis=1).round(4)}")
    sims = cosine_similarity_matrix(vecs, vecs)
    print("similarity matrix:\n", sims.round(3))
