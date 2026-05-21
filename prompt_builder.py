"""
prompt_builder.py — Assembles the RAG prompt within a hard token budget.

Format: SYSTEM → CONTEXT → HISTORY → QUESTION
Trims oldest history first, then lowest-scored chunks if over budget.
"""

from typing import List, Optional, Tuple

import tiktoken

from config import MAX_HISTORY, PROMPT_BUDGET, SYSTEM_PROMPT

# Initialize tiktoken encoder (cl100k_base works for Gemini-class models)
try:
    _encoder = tiktoken.get_encoding("cl100k_base")
except Exception:
    _encoder = tiktoken.get_encoding("cl100k_base")

# Type alias matching retriever.py
ChunkResult = Tuple[str, str, float]


def _count_tokens(text: str) -> int:
    """Count the number of tokens in a text string.

    Args:
        text: Input text.

    Returns:
        Token count.
    """
    return len(_encoder.encode(text))


def _format_context(chunks: List[ChunkResult]) -> str:
    """Format retrieved chunks into a context block.

    Args:
        chunks: List of (text, source, score) tuples.

    Returns:
        Formatted context string.
    """
    if not chunks:
        return "No relevant documentation found for this query."

    parts: List[str] = []
    for i, (text, source, score) in enumerate(chunks, 1):
        parts.append(f"[Source: {source}]\n{text}")

    return "\n\n---\n\n".join(parts)


def _format_history(history: List[dict]) -> str:
    """Format conversation history into a text block.

    Args:
        history: List of {"role": str, "content": str} dicts.

    Returns:
        Formatted history string.
    """
    if not history:
        return ""

    parts: List[str] = []
    for entry in history:
        role = entry.get("role", "user").upper()
        content = entry.get("content", "")
        parts.append(f"{role}: {content}")

    return "\n\n".join(parts)


def build_prompt(
    question: str,
    chunks: List[ChunkResult],
    history: Optional[List[dict]] = None,
) -> str:
    """Assemble the full RAG prompt within the token budget.

    Strategy:
    1. Start with SYSTEM + QUESTION (always included).
    2. Add CONTEXT chunks (highest relevance first).
    3. Add HISTORY (most recent first).
    4. If over budget: trim oldest history first, then lowest-scored chunk.

    Args:
        question: The user's current question.
        chunks: Retrieved context chunks from the vector store.
        history: Conversation history (most recent MAX_HISTORY turns).

    Returns:
        The assembled prompt string.
    """
    if history is None:
        history = []

    # Trim history to MAX_HISTORY turns
    trimmed_history = history[-MAX_HISTORY:]

    # Build components
    system_block = f"=== SYSTEM ===\n{SYSTEM_PROMPT}"
    question_block = f"=== QUESTION ===\n{question}"

    # Calculate base token cost (system + question are always included)
    base_tokens = _count_tokens(system_block) + _count_tokens(question_block)

    # Sort chunks by score descending (best first)
    sorted_chunks = sorted(chunks, key=lambda c: c[2], reverse=True)

    # Budget remaining for context + history
    remaining = PROMPT_BUDGET - base_tokens

    if remaining <= 0:
        # Extremely long question — just use system + question
        return f"{system_block}\n\n{question_block}"

    # Try to fit all context and history; trim if needed
    context_chunks = list(sorted_chunks)
    hist_entries = list(trimmed_history)

    # First pass: calculate total tokens needed
    context_text = _format_context(context_chunks)
    history_text = _format_history(hist_entries)

    context_tokens = _count_tokens(context_text) if context_text else 0
    history_tokens = _count_tokens(history_text) if history_text else 0

    total_extra = context_tokens + history_tokens

    # Trim oldest history entries first
    while total_extra > remaining and hist_entries:
        hist_entries.pop(0)  # remove oldest
        history_text = _format_history(hist_entries)
        history_tokens = _count_tokens(history_text) if history_text else 0
        total_extra = context_tokens + history_tokens

    # Then trim lowest-scored chunks
    while total_extra > remaining and context_chunks:
        context_chunks.pop()  # remove lowest-scored (last in sorted list)
        context_text = _format_context(context_chunks)
        context_tokens = _count_tokens(context_text) if context_text else 0
        total_extra = context_tokens + history_tokens

    # Assemble final prompt
    parts = [system_block]

    if context_chunks:
        parts.append(f"=== CONTEXT ===\n{_format_context(context_chunks)}")

    if hist_entries:
        parts.append(
            f"=== CONVERSATION HISTORY ===\n{_format_history(hist_entries)}"
        )

    parts.append(question_block)

    return "\n\n".join(parts)


def get_prompt_token_count(prompt: str) -> int:
    """Return the token count of an assembled prompt.

    Args:
        prompt: The assembled prompt string.

    Returns:
        Token count.
    """
    return _count_tokens(prompt)
