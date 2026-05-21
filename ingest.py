"""
ingest.py — Orchestrates scraping GROMACS docs and embedding into ChromaDB.

Phase 1: Scrape official GROMACS documentation → save as markdown to docs/
Phase 2: Chunk docs/ → embed with text-embedding-004 → store in ChromaDB

Run once before using agent.py. Re-running is safe (caches prevent re-work).
"""

import hashlib
import os
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chromadb
import tiktoken
from chromadb.config import Settings

from config import (
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DOCS_DIR,
    EMBEDDINGS_CACHE,
    EMBEDDING_MODEL,
)
from display import (
    console,
    create_progress,
    inline_status,
    show_error,
    show_success,
    show_warning,
    step_status,
)
from gemini import embed_text
from scraper import scrape_all_pages

# Tiktoken encoder for chunking
_encoder = tiktoken.get_encoding("cl100k_base")


def _tokenize(text: str) -> List[int]:
    """Tokenize text using tiktoken.

    Args:
        text: Input text.

    Returns:
        List of token IDs.
    """
    return _encoder.encode(text)


def _detokenize(tokens: List[int]) -> str:
    """Convert token IDs back to text.

    Args:
        tokens: List of token IDs.

    Returns:
        Decoded text string.
    """
    return _encoder.decode(tokens)


def _chunk_text(
    text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> List[str]:
    """Split text into overlapping chunks by token count.

    Args:
        text: The input text to chunk.
        chunk_size: Maximum tokens per chunk.
        overlap: Token overlap between consecutive chunks.

    Returns:
        List of text chunks.
    """
    tokens = _tokenize(text)
    chunks: List[str] = []
    start = 0

    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = _detokenize(chunk_tokens)
        if chunk_text.strip():
            chunks.append(chunk_text)
        start += chunk_size - overlap

    return chunks


def _file_hash(path: Path) -> str:
    """Compute MD5 hash of a file for cache invalidation.

    Args:
        path: Path to the file.

    Returns:
        Hex digest string.
    """
    return hashlib.md5(path.read_bytes()).hexdigest()


def _load_embeddings_cache() -> Dict[str, Any]:
    """Load the embeddings cache from disk.

    Returns:
        Cache dict mapping file hashes to their chunk data.
    """
    if EMBEDDINGS_CACHE.exists():
        try:
            with open(EMBEDDINGS_CACHE, "rb") as f:
                return pickle.load(f)
        except Exception:
            return {}
    return {}


def _save_embeddings_cache(cache: Dict[str, Any]) -> None:
    """Save the embeddings cache to disk.

    Args:
        cache: Cache dict to persist.
    """
    with open(EMBEDDINGS_CACHE, "wb") as f:
        pickle.dump(cache, f)


def _get_or_create_collection(
    client: chromadb.ClientAPI,
) -> chromadb.Collection:
    """Get or create the gromacs_docs ChromaDB collection.

    Args:
        client: ChromaDB client instance.

    Returns:
        The ChromaDB collection.
    """
    return client.get_or_create_collection(
        name="gromacs_docs",
        metadata={"hnsw:space": "cosine"},
    )


def run_ingest() -> None:
    """Run the full ingestion pipeline: scrape → chunk → embed → store."""
    start_time = time.time()

    console.print()
    console.rule("[bold cyan]GROMACS Documentation Ingestion[/bold cyan]")
    console.print()

    # ── Step 1: Check scrape cache ───────────────────────────────────────
    step_status("1", "6", "Checking scrape cache...", "...", "info")

    # ── Step 2: Scrape documentation ─────────────────────────────────────
    step_status("2", "6",
                f"Scraping GROMACS documentation ({len(scrape_all_pages.__code__.co_varnames)} pages)...",
                "...", "info")
    console.print()
    scrape_results = scrape_all_pages()
    pages_scraped = sum(1 for v in scrape_results.values() if v)
    console.print()

    # ── Step 3: Load and process markdown files ──────────────────────────
    step_status("3", "6", "Cleaning and saving markdown files...",
                "✓", "success")

    # Collect all markdown files
    md_files = sorted(DOCS_DIR.glob("*.md"))
    if not md_files:
        show_error("No markdown files found in docs/. Aborting.")
        raise SystemExit(1)

    # ── Step 4: Check embedding cache ────────────────────────────────────
    step_status("4", "6", "Checking embedding cache...", "...", "info")
    embed_cache = _load_embeddings_cache()

    # Determine which files need re-embedding
    all_chunks: List[Tuple[str, str, str]] = []  # (chunk_text, source, chunk_id)
    cached_chunks: List[Tuple[str, str, str, List[float]]] = []  # + embedding

    for md_file in md_files:
        file_hash = _file_hash(md_file)
        source_name = md_file.name  # e.g. 'mdp_options.md'
        text = md_file.read_text(encoding="utf-8")

        chunks = _chunk_text(text)

        if file_hash in embed_cache:
            # File unchanged — reuse cached embeddings
            cached_data = embed_cache[file_hash]
            for i, chunk_text in enumerate(chunks):
                chunk_id = f"{source_name}_{i}"
                if i < len(cached_data.get("embeddings", [])):
                    cached_chunks.append((
                        chunk_text,
                        source_name,
                        chunk_id,
                        cached_data["embeddings"][i],
                    ))
                else:
                    all_chunks.append((chunk_text, source_name, chunk_id))
        else:
            # File changed or new — needs embedding
            for i, chunk_text in enumerate(chunks):
                chunk_id = f"{source_name}_{i}"
                all_chunks.append((chunk_text, source_name, chunk_id))

    total_chunks = len(all_chunks) + len(cached_chunks)
    console.print(
        f"  [dim]{len(cached_chunks)} chunks cached, "
        f"{len(all_chunks)} new/changed[/dim]"
    )

    # ── Step 5: Embed new/changed chunks ─────────────────────────────────
    step_status("5", "6",
                f"Chunking and embedding new content...",
                "...", "info")

    new_embeddings: Dict[str, Dict[str, Any]] = {}  # file_hash → data

    if all_chunks:
        progress = create_progress()
        with progress:
            task = progress.add_task(
                "  Embedding chunks", total=len(all_chunks)
            )

            for chunk_text, source_name, chunk_id in all_chunks:
                embedding = embed_text(chunk_text)
                if embedding is not None:
                    # Group by source file for cache
                    md_file = DOCS_DIR / source_name
                    fhash = _file_hash(md_file)

                    if fhash not in new_embeddings:
                        new_embeddings[fhash] = {"embeddings": []}
                    new_embeddings[fhash]["embeddings"].append(embedding)

                    cached_chunks.append((
                        chunk_text, source_name, chunk_id, embedding
                    ))

                progress.update(task, advance=1)
    else:
        console.print("  [dim]All chunks already cached — nothing to embed.[/dim]")

    # Update embedding cache
    embed_cache.update(new_embeddings)
    _save_embeddings_cache(embed_cache)

    # ── Step 6: Store in ChromaDB ────────────────────────────────────────
    step_status("6", "6", "Saving to ChromaDB...", "...", "info")

    # Create fresh ChromaDB
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )

    # Delete existing collection if present, then recreate
    try:
        client.delete_collection("gromacs_docs")
    except Exception:
        pass

    collection = _get_or_create_collection(client)

    # Batch upsert (ChromaDB has a batch limit)
    batch_size = 100
    valid_chunks = [c for c in cached_chunks if len(c) == 4]

    for i in range(0, len(valid_chunks), batch_size):
        batch = valid_chunks[i : i + batch_size]
        ids = [c[2] for c in batch]
        documents = [c[0] for c in batch]
        metadatas = [{"source": c[1]} for c in batch]
        embeddings = [c[3] for c in batch]

        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

    step_status("6", "6", "Saving to ChromaDB...", "✓", "success")

    # ── Summary ──────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    console.print()
    console.rule("[bold green]Done[/bold green]")
    console.print(
        f"  [success]{pages_scraped}[/success] pages scraped, "
        f"[success]{len(valid_chunks)}[/success] chunks indexed "
        f"in [cyan]{elapsed:.0f}s[/cyan]."
    )
    console.print()


if __name__ == "__main__":
    run_ingest()
