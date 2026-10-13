"""Clinical Research RAG Assistant - core pipeline package.

Modules
-------
config      : paths and environment-driven settings
ingest      : PDF -> page records (PyMuPDF)
chunking    : page records -> overlapping text chunks with metadata
embeddings  : chunks -> normalized dense vectors (SentenceTransformers)
retriever   : vectors -> FAISS inner-product index + top-k search
generator   : question + retrieved chunks -> grounded, cited answer
evaluation  : retrieval quality metrics (Recall@K, MRR)
"""

__version__ = "0.1.0"
