"""
config.py — Central configuration for the GROMACS Simulation Agent.

All constants, target URLs, file paths, and the rate limiter live here.
"""

import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

# ─── Directory Paths ──────────────────────────────────────────────────────────
BASE_DIR: Path = Path(__file__).resolve().parent
DOCS_DIR: Path = BASE_DIR / "docs"
CACHE_DIR: Path = BASE_DIR / "scrape_cache"
CHROMA_DIR: Path = BASE_DIR / "chroma_db"
STRUCTURES_DIR: Path = BASE_DIR / "input_structures"
EMBEDDINGS_CACHE: Path = BASE_DIR / "embeddings_cache.pkl"
ENV_FILE: Path = BASE_DIR / ".env"

# ─── Model Configuration ─────────────────────────────────────────────────────
GENERATION_MODEL: str = "gemini-2.0-flash-lite"
EMBEDDING_MODEL: str = "gemini-embedding-001"

# ─── Gemini API Base URL ─────────────────────────────────────────────────────
GEMINI_API_BASE: str = "https://generativelanguage.googleapis.com/v1beta"

# ─── RAG Parameters ──────────────────────────────────────────────────────────
CHUNK_SIZE: int = 150          # tokens per chunk
CHUNK_OVERLAP: int = 30        # token overlap between chunks
TOP_K: int = 3                 # number of retrieved chunks
SIMILARITY_THRESHOLD: float = 0.75  # cosine similarity floor
MAX_HISTORY: int = 4           # conversation turns to keep
PROMPT_BUDGET: int = 3000      # hard token budget for assembled prompt
VERBOSE: bool = False          # set True to see detailed API warnings

# ─── Reliability ──────────────────────────────────────────────────────────────
MAX_RETRIES: int = 3
MAX_CONSECUTIVE_ERRORS: int = 5
REQUEST_TIMEOUT: int = 15      # seconds for all HTTP requests
CACHE_MAX_AGE_DAYS: int = 7    # scrape cache expiry

# ─── Backoff Schedule (seconds) ──────────────────────────────────────────────
BACKOFF_SCHEDULE: Tuple[int, ...] = (10, 20, 40)

# ─── Target GROMACS Documentation URLs ────────────────────────────────────────
SCRAPE_URLS: Dict[str, str] = {
    "mdp_options":
        "https://manual.gromacs.org/2024.6/user-guide/mdp-options.html",
    "mdrun_performance":
        "https://manual.gromacs.org/2024.6/user-guide/mdrun-performance.html",
    "getting_started":
        "https://manual.gromacs.org/2024.6/user-guide/getting-started.html",
    "system_preparation":
        "https://manual.gromacs.org/2024.6/user-guide/system-preparation.html",
    "force_fields":
        "https://manual.gromacs.org/2024.6/user-guide/force-fields.html",
    "molecular_dynamics":
        "https://manual.gromacs.org/2024.6/reference-manual/algorithms/molecular-dynamics.html",
    "analysis_tools":
        "https://manual.gromacs.org/2024.6/reference-manual/analysis.html",
    "fep":
        "https://manual.gromacs.org/2024.6/reference-manual/special-topics/free-energy-implementation.html",
    "gmx_commands":
        "https://manual.gromacs.org/2024.6/onlinehelp/gmx.html",
    "water_models":
        "https://manual.gromacs.org/2024.6/reference-manual/topologies/water-models.html",
    "error_messages":
        "https://manual.gromacs.org/2024.6/user-guide/known-issues.html",
}

# ─── Slash Commands ──────────────────────────────────────────────────────────
COMMANDS: List[str] = [
    "/help", "/clear", "/save", "/sources",
    "/reindex", "/structures", "/exit", "/key",
]

# ─── System Prompt ────────────────────────────────────────────────────────────
SYSTEM_PROMPT: str = """You are a GROMACS Molecular Dynamics simulation expert assistant.

Rules you MUST follow:
1. Answer ONLY from the retrieved context provided below. Never invent or hallucinate
   GROMACS parameters, flags, or commands that are not in the context.
2. Be concise — no lengthy preambles or unnecessary filler.
3. When asked for simulation setups, generate COMPLETE .mdp files and bash pipelines
   that the user can run directly.
4. Always reference files in the input_structures/ directory in generated commands.
5. Cite the source document(s) you used inline, e.g. [mdp_options.md].
6. Use GROMACS 2023/2024 syntax exclusively.
7. Never suggest downloading structure files — the user provides them in input_structures/.
8. If the context does not contain enough information to answer, say so honestly
   rather than guessing.
"""


class RateLimiter:
    """Simple rate limiter for Gemini API calls."""

    def __init__(self, min_interval: float) -> None:
        self._last_call: float = 0.0
        self._min_interval: float = min_interval

    def wait_if_needed(self) -> None:
        """Block until the minimum interval has elapsed since the last call."""
        elapsed = time.time() - self._last_call
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_call = time.time()

    def record_call(self) -> None:
        """Record the timestamp of a successful API call."""
        self._last_call = time.time()


# Embedding model: 100 RPM free tier → 1 request every ~0.7 s (use 1.0 s to be safe)
embedding_rate_limiter = RateLimiter(min_interval=1.0)

# Generation model: 15 RPM free tier → 1 request every ~4.5 s (use 5.0 s to be safe)
generation_rate_limiter = RateLimiter(min_interval=5.0)

# Backwards-compatible alias (used by any code that still imports `rate_limiter`)
rate_limiter = generation_rate_limiter
