"""Vector search over chunk embeddings using FAISS.

Index choice
------------
`faiss.IndexFlatIP` - an exact, brute-force inner-product index. For a corpus
of a few thousand chunks this is fast (milliseconds) and, unlike an
approximate index (IVF/HNSW), it has no recall loss and no tuning knobs to
explain away. Because `src.embeddings` L2-normalizes every vector, inner
product == cosine similarity.

Approximate indexes only start to matter at ~10^6 vectors, which is documented
as future work rather than pretended to be needed now.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

try:
    import faiss
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "faiss-cpu is required for retrieval. Install it with:\n"
        "    pip install faiss-cpu"
    ) from exc

INDEX_FILENAME = "faiss.index"
METADATA_FILENAME = "chunks.json"
MANIFEST_FILENAME = "index_manifest.json"


class RetrieverError(RuntimeError):
    """Raised when the index is missing, malformed, or queried incorrectly."""


@dataclass
class RetrievedChunk:
    """One search hit, carrying everything needed for an honest citation."""

    rank: int              # 1-based position in the result list
    score: float           # cosine similarity in [-1, 1]
    document: str
    title: str
    page: int
    chunk_id: str
    text: str

    def as_dict(self) -> Dict[str, object]:
        return {
            "rank": self.rank,
            "score": self.score,
            "document": self.document,
            "title": self.title,
            "page": self.page,
            "chunk_id": self.chunk_id,
            "text": self.text,
        }

    def citation(self) -> str:
        """Short citation string, e.g. `paper1.pdf - page 7`."""
        return f"{self.document} - page {self.page}"


class FaissRetriever:
    """A FAISS index plus the chunk metadata that gives its rows meaning."""

    def __init__(
        self,
        index: "faiss.Index",
        chunks: Sequence[Dict[str, object]],
        embedding_model_name: str = "",
    ):
        if index.ntotal != len(chunks):
            raise RetrieverError(
                f"Index/metadata mismatch: index holds {index.ntotal} vectors "
                f"but {len(chunks)} chunks were supplied."
            )
        self.index = index
        self.chunks: List[Dict[str, object]] = list(chunks)
        self.embedding_model_name = embedding_model_name

    # -- construction -------------------------------------------------------
    @classmethod
    def build(
        cls,
        chunks: Sequence[Dict[str, object]],
        embeddings: np.ndarray,
        embedding_model_name: str = "",
    ) -> "FaissRetriever":
        """Build an exact inner-product index from normalized embeddings."""
        if len(chunks) == 0:
            raise RetrieverError("Cannot build an index from zero chunks.")

        matrix = np.ascontiguousarray(np.asarray(embeddings, dtype="float32"))
        if matrix.ndim != 2:
            raise RetrieverError(
                f"Embeddings must be 2-D, got shape {matrix.shape}."
            )
        if matrix.shape[0] != len(chunks):
            raise RetrieverError(
                f"Got {matrix.shape[0]} embeddings for {len(chunks)} chunks."
            )

        index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)
        return cls(index, chunks, embedding_model_name)

    # -- search -------------------------------------------------------------
    def retrieve(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        min_score: Optional[float] = None,
    ) -> List[RetrievedChunk]:
        """Return the `top_k` most similar chunks to a query embedding.

        Parameters
        ----------
        query_embedding:
            Shape `(dim,)` or `(1, dim)`. Must already be L2-normalized -
            `EmbeddingModel.encode_query` does this for you.
        top_k:
            Number of chunks to return (clipped to the index size).
        min_score:
            Optional similarity floor. Hits below it are dropped, which is how
            the app can honestly say "no relevant evidence found" instead of
            showing unrelated text with a 0.05 score.
        """
        if self.index.ntotal == 0:
            raise RetrieverError("The index is empty - build it before searching.")

        query = np.asarray(query_embedding, dtype="float32")
        if query.ndim == 1:
            query = query.reshape(1, -1)
        if query.shape[0] != 1:
            raise RetrieverError(
                f"retrieve() expects a single query vector, got {query.shape[0]}."
            )
        if query.shape[1] != self.index.d:
            raise RetrieverError(
                f"Query dimension {query.shape[1]} does not match index "
                f"dimension {self.index.d}. This usually means the index was "
                f"built with a different embedding model - rebuild the index."
            )

        k = max(1, min(int(top_k), self.index.ntotal))
        scores, indices = self.index.search(np.ascontiguousarray(query), k)

        hits: List[RetrievedChunk] = []
        rank = 0
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:  # FAISS pads with -1 when fewer than k results exist
                continue
            if min_score is not None and float(score) < min_score:
                continue
            chunk = self.chunks[int(idx)]
            rank += 1
            hits.append(
                RetrievedChunk(
                    rank=rank,
                    score=float(score),
                    document=str(chunk["document"]),
                    title=str(chunk.get("title") or chunk["document"]),
                    page=int(chunk["page"]),
                    chunk_id=str(chunk["chunk_id"]),
                    text=str(chunk["text"]),
                )
            )
        return hits

    # -- persistence --------------------------------------------------------
    def save(self, directory: Union[str, Path]) -> Path:
        """Write the index, chunk metadata and a manifest to `directory`."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(directory / INDEX_FILENAME))
        (directory / METADATA_FILENAME).write_text(
            json.dumps(self.chunks, ensure_ascii=False), encoding="utf-8"
        )
        manifest = {
            "n_chunks": len(self.chunks),
            "dimension": int(self.index.d),
            "embedding_model": self.embedding_model_name,
            "documents": sorted({str(c["document"]) for c in self.chunks}),
        }
        (directory / MANIFEST_FILENAME).write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return directory

    @classmethod
    def load(cls, directory: Union[str, Path]) -> "FaissRetriever":
        """Load a previously saved index + metadata."""
        directory = Path(directory)
        index_path = directory / INDEX_FILENAME
        meta_path = directory / METADATA_FILENAME

        if not index_path.exists() or not meta_path.exists():
            raise RetrieverError(
                f"No saved index found in {directory}. Build one first "
                f"(Streamlit sidebar -> 'Build / Rebuild Index')."
            )

        index = faiss.read_index(str(index_path))
        chunks = json.loads(meta_path.read_text(encoding="utf-8"))

        model_name = ""
        manifest_path = directory / MANIFEST_FILENAME
        if manifest_path.exists():
            try:
                model_name = json.loads(
                    manifest_path.read_text(encoding="utf-8")
                ).get("embedding_model", "")
            except json.JSONDecodeError:
                model_name = ""

        return cls(index, chunks, model_name)

    # -- introspection ------------------------------------------------------
    @property
    def n_chunks(self) -> int:
        return len(self.chunks)

    @property
    def documents(self) -> List[str]:
        return sorted({str(c["document"]) for c in self.chunks})

    def __repr__(self) -> str:  # pragma: no cover - debugging convenience
        return (
            f"FaissRetriever(chunks={self.n_chunks}, dim={self.index.d}, "
            f"documents={len(self.documents)}, model='{self.embedding_model_name}')"
        )


# ---------------------------------------------------------------------------
# Function-style API (mirrors the shapes asked for in the project spec)
# ---------------------------------------------------------------------------
def build_index(
    chunks: Sequence[Dict[str, object]],
    embeddings: np.ndarray,
    embedding_model_name: str = "",
) -> FaissRetriever:
    """Build a `FaissRetriever` from chunks + their embeddings."""
    return FaissRetriever.build(chunks, embeddings, embedding_model_name)


def retrieve(
    retriever: FaissRetriever,
    query_embedding: np.ndarray,
    top_k: int = 5,
    min_score: Optional[float] = None,
) -> List[RetrievedChunk]:
    """Search an existing retriever (functional wrapper around `.retrieve`)."""
    return retriever.retrieve(query_embedding, top_k=top_k, min_score=min_score)


# ---------------------------------------------------------------------------
# Manual check:  python -m src.retriever
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from src.chunking import chunk_records
    from src.config import SAMPLE_PAPERS_DIR, STORAGE_DIR, Settings
    from src.embeddings import EmbeddingModel, embed_chunks
    from src.ingest import load_pdf_directory

    settings = Settings.from_env()
    records, report = load_pdf_directory(SAMPLE_PAPERS_DIR, settings.min_page_chars)
    print(report.as_text())
    if not records:
        raise SystemExit("No text extracted - add PDFs to data/sample_papers/ first.")

    chunks = chunk_records(records, settings.chunk_size, settings.chunk_overlap)
    model = EmbeddingModel(settings.embedding_model, settings.embedding_device)
    matrix, chunks = embed_chunks(chunks, model=model)

    retriever = build_index(chunks, matrix, settings.embedding_model)
    retriever.save(STORAGE_DIR)
    print(retriever)

    question = "What limitations did the authors report?"
    for hit in retriever.retrieve(model.encode_query(question), top_k=settings.top_k):
        print(f"\n[{hit.rank}] {hit.score:.3f}  {hit.citation()}  ({hit.chunk_id})")
        print(hit.text[:220], "...")
