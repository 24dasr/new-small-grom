"""
display.py — Rich terminal UI helpers for the GROMACS Simulation Agent.

Provides ASCII banner, status spinners, progress bars, streaming renderer,
and source citation formatting.
"""

import sys
from typing import List, Optional

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.text import Text
from rich.theme import Theme

# ─── Console Setup ────────────────────────────────────────────────────────────
custom_theme = Theme({
    "info": "cyan",
    "success": "green",
    "warning": "yellow",
    "error": "red bold",
    "step": "bold blue",
    "source": "dim cyan",
})

console = Console(theme=custom_theme)

# ─── ASCII Art Banner ────────────────────────────────────────────────────────
BANNER = r"""
[bold cyan]
   ██████╗ ██████╗  ██████╗ ███╗   ███╗ █████╗  ██████╗███████╗
  ██╔════╝ ██╔══██╗██╔═══██╗████╗ ████║██╔══██╗██╔════╝██╔════╝
  ██║  ███╗██████╔╝██║   ██║██╔████╔██║███████║██║     ███████╗
  ██║   ██║██╔══██╗██║   ██║██║╚██╔╝██║██╔══██║██║     ╚════██║
  ╚██████╔╝██║  ██║╚██████╔╝██║ ╚═╝ ██║██║  ██║╚██████╗███████║
   ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚═╝     ╚═╝╚═╝  ╚═╝ ╚═════╝╚══════╝
[/bold cyan]
[bold white]  Molecular Dynamics Simulation Agent[/bold white]
[dim]  Powered by Gemini Flash · RAG-enhanced · GROMACS 2023/2024[/dim]
"""

SETUP_BANNER = r"""
[bold cyan]
  ╔═══════════════════════════════════════════╗
  ║   GROMACS Agent — First-Run Setup Wizard  ║
  ╚═══════════════════════════════════════════╝
[/bold cyan]
"""


def show_banner() -> None:
    """Display the main ASCII art banner."""
    console.print(BANNER)


def show_setup_banner() -> None:
    """Display the setup wizard banner."""
    console.print(SETUP_BANNER)


def step_status(step: str, total: str, message: str, status: str = "...",
                style: str = "info") -> None:
    """Print a numbered step status line.

    Args:
        step: Current step number (e.g. "1").
        total: Total steps (e.g. "4").
        message: Description of the step.
        status: Status indicator (e.g. "✓", "✗", "⚠").
        style: Rich style name for the status.
    """
    console.print(
        f"  [step]\\[{step}/{total}][/step] {message:<45} "
        f"[{style}]{status}[/{style}]"
    )


def inline_status(message: str, status: str = "...",
                  style: str = "info") -> None:
    """Print a bullet-point status line (for per-query steps).

    Args:
        message: Description of the operation.
        status: Status indicator.
        style: Rich style name.
    """
    console.print(f"  [dim]●[/dim] {message:<45} [{style}]{status}[/{style}]")


def check_item(label: str, ok: bool, detail: str = "",
               warn: bool = False) -> None:
    """Print a checklist item with ✓ / ✗ / ⚠.

    Args:
        label: Description of the item.
        ok: Whether the check passed.
        detail: Additional detail text.
        warn: If True, show ⚠ instead of ✗ on failure.
    """
    if ok:
        icon = "[success]✓[/success]"
    elif warn:
        icon = "[warning]⚠[/warning]"
    else:
        icon = "[error]✗[/error]"

    detail_str = f"  [dim]({detail})[/dim]" if detail else ""
    console.print(f"  {icon} {label}{detail_str}")


def show_sources(sources: List[str]) -> None:
    """Print a compact source citation line.

    Args:
        sources: List of source document filenames.
    """
    if not sources:
        return
    unique = list(dict.fromkeys(sources))  # preserve order, deduplicate
    citation = " · ".join(unique)
    console.print(f"\n  [source]📄 Sources: {citation}[/source]")


def show_error(message: str) -> None:
    """Print a formatted error message.

    Args:
        message: Error description.
    """
    console.print(f"  [error]✗[/error] {message}")


def show_warning(message: str) -> None:
    """Print a formatted warning message.

    Args:
        message: Warning description.
    """
    console.print(f"  [warning]⚠[/warning] {message}")


def show_success(message: str) -> None:
    """Print a formatted success message.

    Args:
        message: Success description.
    """
    console.print(f"  [success]✓[/success] {message}")


def show_help() -> None:
    """Print the list of available slash commands."""
    help_text = """
[bold]Available Commands:[/bold]

  [cyan]/help[/cyan]        Show this help message
  [cyan]/clear[/cyan]       Clear conversation history
  [cyan]/save[/cyan]        Save the last response to a file
  [cyan]/sources[/cyan]     Show sources from last retrieval
  [cyan]/reindex[/cyan]     Re-run ingest.py to rebuild the index
  [cyan]/structures[/cyan]  Rescan input_structures/ directory
  [cyan]/key[/cyan]         Update your Gemini API key
  [cyan]/exit[/cyan]        Exit the agent

[dim]Press Enter twice to submit a multi-line query.[/dim]
"""
    console.print(Panel(help_text.strip(), title="Help", border_style="cyan"))


def stream_markdown(text: str) -> None:
    """Render a complete response as markdown.

    Args:
        text: The full response text to render.
    """
    console.print()
    console.print(Markdown(text))
    console.print()


class StreamPrinter:
    """Prints streamed tokens to the console character-by-character."""

    def __init__(self) -> None:
        self._buffer: str = ""

    def print_token(self, token: str) -> None:
        """Print a single token chunk to stdout.

        Args:
            token: The text chunk to print.
        """
        sys.stdout.write(token)
        sys.stdout.flush()
        self._buffer += token

    def finish(self) -> str:
        """Finalize the stream, print a newline, and return the full text.

        Returns:
            The complete streamed text.
        """
        sys.stdout.write("\n")
        sys.stdout.flush()
        result = self._buffer
        self._buffer = ""
        return result


def create_progress() -> Progress:
    """Create a Rich progress bar for embedding operations.

    Returns:
        A configured Progress instance.
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    )


def show_scrape_status(name: str, status: str, detail: str = "") -> None:
    """Print a scrape status line for a documentation page.

    Args:
        name: The page identifier (e.g. 'mdp_options').
        status: Status icon (e.g. '✓', '⚡', '↓').
        detail: Additional detail text.
    """
    detail_str = f"  ({detail})" if detail else ""
    console.print(f"  Scraping {name:<25} {status}{detail_str}")


def show_rate_limit(retry: int, max_retries: int, wait: int) -> None:
    """Print a rate limit warning with retry countdown.

    Args:
        retry: Current retry attempt number.
        max_retries: Maximum retry attempts.
        wait: Seconds to wait before retry.
    """
    console.print(
        f"  [warning]⚠[/warning] Rate limit hit (429). "
        f"Waiting {wait}s... retry {retry}/{max_retries}"
    )
