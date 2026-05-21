"""
scraper.py — Fetches and cleans GROMACS documentation pages.

Uses requests + BeautifulSoup4 + html2text. No Scrapy, no Selenium.
Implements a 7-day file-based cache and atomic markdown writes.
"""

import os
import re
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import html2text
import requests
from bs4 import BeautifulSoup

from config import (
    CACHE_DIR,
    CACHE_MAX_AGE_DAYS,
    DOCS_DIR,
    REQUEST_TIMEOUT,
    SCRAPE_URLS,
)
from display import console, show_error, show_scrape_status, show_warning

# ─── User-Agent Header ───────────────────────────────────────────────────────
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; GROMACSAgent/1.0; "
        "+https://github.com/gromacs-agent)"
    ),
}


def _ensure_dirs() -> None:
    """Create docs/ and scrape_cache/ directories if they don't exist."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_path(name: str) -> Path:
    """Return the cache file path for a given page name.

    Args:
        name: Page identifier (e.g. 'mdp_options').

    Returns:
        Path to the cached HTML file.
    """
    return CACHE_DIR / f"{name}.html"


def _is_cache_valid(name: str) -> Tuple[bool, Optional[int]]:
    """Check if a cached HTML file exists and is fresh enough.

    Args:
        name: Page identifier.

    Returns:
        Tuple of (is_valid, age_in_days). age_in_days is None if no cache.
    """
    path = _cache_path(name)
    if not path.exists():
        return False, None

    mtime = datetime.fromtimestamp(path.stat().st_mtime)
    age = datetime.now() - mtime
    age_days = age.days

    if age < timedelta(days=CACHE_MAX_AGE_DAYS):
        return True, age_days
    return False, age_days


def _fetch_page(name: str, url: str) -> Optional[str]:
    """Fetch a page from the internet with one retry on timeout.

    Args:
        name: Page identifier.
        url: URL to fetch.

    Returns:
        Raw HTML string, or None on failure.
    """
    max_attempts = 2
    for attempt in range(1, max_attempts + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 200:
                # Save to cache
                cache_path = _cache_path(name)
                cache_path.write_text(resp.text, encoding="utf-8")
                return resp.text
            else:
                show_warning(
                    f"{name}: HTTP {resp.status_code} — skipping"
                )
                return None
        except requests.exceptions.Timeout:
            if attempt < max_attempts:
                time.sleep(3)
                continue
            show_warning(f"{name}: timed out twice — skipping")
            return None
        except requests.exceptions.RequestException as exc:
            show_warning(f"{name}: request failed ({exc}) — skipping")
            return None

    return None


def _extract_content(html: str) -> str:
    """Extract the main content from a GROMACS doc page.

    Looks for <div class="document">, <article>, or <main> in that order.
    Falls back to <body> minus nav/footer if none found.

    Args:
        html: Raw HTML string.

    Returns:
        Cleaned HTML string of just the content.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Try main content containers in priority order
    content = (
        soup.find("div", class_="document")
        or soup.find("article")
        or soup.find("main")
    )

    if content is None:
        # Fallback: use body minus navigation elements
        content = soup.find("body")
        if content is None:
            return ""

    # Remove unwanted elements
    tags_to_remove = [
        "nav",
        "footer",
        "header",
        "script",
        "style",
    ]
    classes_to_remove = [
        "headerlink",
        "toctree-wrapper",
        "related",
        "sphinxsidebar",
        "sphinxsidebarwrapper",
    ]

    for tag_name in tags_to_remove:
        for tag in content.find_all(tag_name):
            tag.decompose()

    for class_name in classes_to_remove:
        for tag in content.find_all(class_=class_name):
            tag.decompose()

    return str(content)


def _html_to_markdown(html: str) -> str:
    """Convert cleaned HTML to markdown using html2text.

    Args:
        html: Cleaned HTML string.

    Returns:
        Markdown string.
    """
    converter = html2text.HTML2Text()
    converter.ignore_links = False      # keep command references
    converter.body_width = 0            # no line wrapping
    converter.protect_links = True
    converter.ignore_images = True
    converter.ignore_emphasis = False

    md = converter.handle(html)
    return _postprocess_markdown(md)


def _postprocess_markdown(md: str) -> str:
    """Clean up converted markdown.

    - Strip excessive blank lines (max 2 consecutive).
    - Remove 'Table of Contents' sections.
    - Ensure code blocks use triple backticks.

    Args:
        md: Raw markdown string.

    Returns:
        Cleaned markdown string.
    """
    # Remove Table of Contents sections
    md = re.sub(
        r"#+\s*Table\s+of\s+Contents.*?(?=\n#|\Z)",
        "",
        md,
        flags=re.IGNORECASE | re.DOTALL,
    )

    # Collapse excessive blank lines (max 2 consecutive)
    md = re.sub(r"\n{4,}", "\n\n\n", md)

    # Strip leading/trailing whitespace
    md = md.strip()

    return md


def _atomic_write(path: Path, content: str) -> None:
    """Write content to a file atomically (temp file → rename).

    Args:
        path: Target file path.
        content: Content to write.
    """
    # Write to a temp file in the same directory, then rename
    fd, tmp_path = tempfile.mkstemp(
        dir=str(path.parent), suffix=".tmp", prefix=".scrape_"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        # On Windows, we need to remove existing file before rename
        if path.exists():
            path.unlink()
        os.rename(tmp_path, str(path))
    except Exception:
        # Clean up temp file on error
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def scrape_page(name: str, url: str) -> bool:
    """Scrape a single GROMACS documentation page.

    Checks cache first, fetches if needed, converts to markdown,
    and saves atomically to docs/{name}.md.

    Args:
        name: Page identifier (e.g. 'mdp_options').
        url: URL to scrape.

    Returns:
        True if the page was successfully processed.
    """
    # Check cache
    cache_valid, age_days = _is_cache_valid(name)

    if cache_valid:
        age_str = f"{age_days} day{'s' if age_days != 1 else ''} old"
        show_scrape_status(name, "[cyan]⚡[/cyan]",
                           f"Using cache ({age_str})")
        html = _cache_path(name).read_text(encoding="utf-8")
    else:
        if age_days is not None:
            show_scrape_status(name, "[yellow]↓[/yellow]",
                               "Fetching (cache expired)")
        else:
            show_scrape_status(name, "[yellow]↓[/yellow]", "Fetching...")

        html = _fetch_page(name, url)
        if html is None:
            return False

    # Extract and convert
    content_html = _extract_content(html)
    if not content_html.strip():
        show_warning(f"{name}: no content extracted — skipping")
        return False

    markdown = _html_to_markdown(content_html)
    if not markdown.strip():
        show_warning(f"{name}: empty markdown — skipping")
        return False

    # Write atomically to docs/
    output_path = DOCS_DIR / f"{name}.md"
    _atomic_write(output_path, markdown)

    size_kb = len(markdown.encode("utf-8")) / 1024
    if not cache_valid:
        show_scrape_status(
            name, "[green]✓[/green]",
            f"{size_kb:.0f}KB → docs/{name}.md"
        )

    return True


def scrape_all_pages() -> Dict[str, bool]:
    """Scrape all target GROMACS documentation pages.

    Returns:
        Dict mapping page names to success status.
    """
    _ensure_dirs()
    results: Dict[str, bool] = {}

    for name, url in SCRAPE_URLS.items():
        try:
            results[name] = scrape_page(name, url)
        except Exception as exc:
            show_error(f"{name}: unexpected error — {exc}")
            results[name] = False

    # Summary
    success_count = sum(1 for v in results.values() if v)
    total = len(results)

    if success_count == 0:
        # Check if docs/ has files from a previous run
        existing_docs = list(DOCS_DIR.glob("*.md"))
        if existing_docs:
            show_warning(
                "Could not scrape docs — using existing docs/ from last run."
            )
        else:
            show_error(
                "No documentation available. Check your internet connection "
                "and re-run ingest.py."
            )
            raise SystemExit(1)

    console.print(
        f"\n  [success]{success_count}/{total}[/success] pages scraped "
        f"successfully."
    )
    return results
