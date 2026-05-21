"""
retriever.py — RAG retrieval from ChromaDB for the GROMACS Agent.

Queries the local vector store for relevant documentation chunks.
On any failure, returns an empty list and logs a warning — never raises.
"""

import os
from typing import List, Optional, Tuple

import chromadb
from chromadb.config import Settings

from config import CHROMA_DIR, SIMILARITY_THRESHOLD, TOP_K
from display import show_error, show_warning
from gemini import embed_text

# Type alias for retrieved chunks: (text, source_file, similarity_score)
ChunkResult = Tuple[str, str, float]

# Module-level client (lazy init)
_client: Optional[chromadb.ClientAPI] = None
_collection: Optional[chromadb.Collection] = None


def _get_collection() -> Optional[chromadb.Collection]:
    """Get or initialize the ChromaDB collection in read-only mode.

    Returns:
        The ChromaDB collection, or None if unavailable.
    """
    global _client, _collection

    if _collection is not None:
        return _collection

    if not CHROMA_DIR.exists():
        show_error("ChromaDB not found. Run: python ingest.py")
        return None

    try:
        _client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
        _collection = _client.get_collection(name="gromacs_docs")
        return _collection
    except Exception as exc:
        show_error(f"Failed to connect to ChromaDB: {exc}")
        return None


def get_chunk_count() -> int:
    """Return the number of chunks in the collection.

    Returns:
        Number of indexed chunks, or 0 on failure.
    """
    collection = _get_collection()
    if collection is None:
        return 0
    try:
        return collection.count()
    except Exception:
        return 0


def retrieve(query: str, k: int = TOP_K) -> List[ChunkResult]:
    """Retrieve the most relevant documentation chunks for a query.

    Args:
        query: The user's natural language question.
        k: Number of chunks to retrieve (default: TOP_K).

    Returns:
        List of (text, source_file, score) tuples, sorted by relevance.
        Returns empty list on any failure.
    """
    collection = _get_collection()
    if collection is None:
        show_warning("Retrieval skipped — ChromaDB unavailable.")
        return []

    # Embed the query
    query_embedding = embed_text(query)
    if query_embedding is None:
        show_warning("Retrieval skipped — embedding failed.")
        return []

    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as exc:
        show_warning(f"ChromaDB query failed: {exc}")
        return []

    if not results or not results.get("documents"):
        return []

    # Parse results
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    chunks: List[ChunkResult] = []
    for doc, meta, dist in zip(documents, metadatas, distances):
        # ChromaDB returns L2 distance by default; convert to similarity
        # For cosine distance: similarity = 1 - distance
        # ChromaDB uses squared L2 by default, but with cosine embeddings
        # we treat distance as cosine distance
        similarity = 1.0 - dist

        if similarity >= SIMILARITY_THRESHOLD:
            source = meta.get("source", "unknown")
            chunks.append((doc, source, similarity))

    # Sort by score descending
    chunks.sort(key=lambda x: x[2], reverse=True)

    return chunks


def get_source_names(chunks: List[ChunkResult]) -> List[str]:
    """Extract unique source filenames from retrieval results.

    Args:
        chunks: List of (text, source, score) tuples.

    Returns:
        List of unique source filenames.
    """
    seen = set()
    sources: List[str] = []
    for _, source, _ in chunks:
        if source not in seen:
            seen.add(source)
            sources.append(source)
    return sources


def reset_connection() -> None:
    """Reset the ChromaDB connection (e.g. after reindexing)."""
    global _client, _collection
    _client = None
    _collection = None
