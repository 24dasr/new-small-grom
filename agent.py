"""
agent.py — Main chat loop for the GROMACS Simulation Agent.

Interactive CLI chatbot with RAG-enhanced responses, slash commands,
multi-line input, streaming output, and a 5-error circuit breaker.
"""

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory

from config import CHROMA_DIR, COMMANDS, MAX_CONSECUTIVE_ERRORS, STRUCTURES_DIR
from display import (
    console,
    inline_status,
    show_banner,
    show_error,
    show_help,
    show_sources,
    show_success,
    show_warning,
    step_status,
    stream_markdown,
    StreamPrinter,
)
from gemini import stream_generate
from prompt_builder import build_prompt, get_prompt_token_count
from retriever import (
    get_chunk_count,
    get_source_names,
    reset_connection,
    retrieve,
)

load_dotenv()


def _scan_structures() -> List[str]:
    """Scan input_structures/ for PDB files.

    Returns:
        List of PDB filenames found.
    """
    if not STRUCTURES_DIR.exists():
        STRUCTURES_DIR.mkdir(parents=True, exist_ok=True)
        return []
    return [f.name for f in STRUCTURES_DIR.glob("*.pdb")]


def _startup() -> bool:
    """Run the 4-step startup sequence.

    Returns:
        True if startup succeeded, False otherwise.
    """
    show_banner()
    console.print()

    # Step 1: Load configuration
    step_status("1", "4", "Loading configuration...", "✓", "success")

    # Step 2: Connect to ChromaDB
    if not CHROMA_DIR.exists():
        step_status("2", "4", "Connecting to ChromaDB...", "✗", "error")
        show_error("ChromaDB not found. Run: python ingest.py")
        return False

    chunk_count = get_chunk_count()
    if chunk_count == 0:
        step_status("2", "4", "Connecting to ChromaDB...", "⚠", "warning")
        show_warning("ChromaDB is empty. Run: python ingest.py")
        return False

    step_status(
        "2", "4", "Connecting to ChromaDB...",
        f"✓  ({chunk_count:,} chunks loaded)", "success",
    )

    # Step 3: Scan input_structures
    pdb_files = _scan_structures()
    if pdb_files:
        step_status(
            "3", "4", "Scanning input_structures/...",
            f"✓  ({len(pdb_files)} PDB file{'s' if len(pdb_files) != 1 else ''} found)",
            "success",
        )
    else:
        step_status(
            "3", "4", "Scanning input_structures/...",
            "✓  (no PDB files yet)", "success",
        )

    # Step 4: Ready
    step_status("4", "4", "Ready.", "✓", "success")
    console.print()
    console.print(
        "  [dim]Type your question and press Enter twice to submit. "
        "Type /help for commands.[/dim]\n"
    )

    return True


def _handle_command(
    command: str,
    history: List[dict],
    last_response: Optional[str],
    last_sources: List[str],
) -> Optional[str]:
    """Handle a slash command.

    Args:
        command: The command string (e.g. '/help').
        history: Current conversation history.
        last_response: The last agent response text.
        last_sources: Source files from the last retrieval.

    Returns:
        'exit' to quit, 'reindex' to reindex, or None to continue.
    """
    cmd = command.strip().lower()

    if cmd == "/help":
        show_help()

    elif cmd == "/clear":
        history.clear()
        show_success("Conversation history cleared.")

    elif cmd == "/save":
        if not last_response:
            show_warning("No response to save.")
        else:
            save_path = Path("last_response.md")
            save_path.write_text(last_response, encoding="utf-8")
            show_success(f"Response saved to {save_path}")

    elif cmd == "/sources":
        if last_sources:
            show_sources(last_sources)
        else:
            show_warning("No sources from last query.")

    elif cmd == "/reindex":
        console.print("  Launching ingest.py...")
        try:
            subprocess.run(
                [sys.executable, "ingest.py"],
                cwd=str(Path(__file__).resolve().parent),
            )
            reset_connection()
            show_success("Reindex complete. Connection refreshed.")
        except Exception as exc:
            show_error(f"Reindex failed: {exc}")

    elif cmd == "/structures":
        pdb_files = _scan_structures()
        if pdb_files:
            console.print(f"\n  [bold]PDB files ({len(pdb_files)}):[/bold]")
            for f in pdb_files:
                console.print(f"    • {f}")
            console.print()
        else:
            show_warning("No PDB files in input_structures/")

    elif cmd == "/key":
        console.print("  Run [cyan]python setup.py[/cyan] to update your API key.")

    elif cmd == "/exit":
        return "exit"

    else:
        show_warning(
            f"Unknown command: {cmd}. Type /help for available commands."
        )

    return None


def _read_multiline(session: PromptSession) -> Optional[str]:
    """Read multi-line input: single Enter for newline, double Enter to submit.

    Args:
        session: prompt_toolkit session.

    Returns:
        The user's input text, or None on EOF/empty.
    """
    lines: List[str] = []

    try:
        while True:
            prompt_str = "You > " if not lines else "...  "
            line = session.prompt(prompt_str)

            if line == "" and lines and lines[-1] == "":
                # Double enter — submit (remove trailing empty lines)
                while lines and lines[-1] == "":
                    lines.pop()
                break

            lines.append(line)

    except EOFError:
        if lines:
            return "\n".join(lines).strip()
        console.print("\nGoodbye!")
        raise SystemExit(0)
    except KeyboardInterrupt:
        if lines:
            return None
        console.print("\n  [dim]Use /exit to quit.[/dim]")
        return None

    text = "\n".join(lines).strip()
    return text if text else None


def main() -> None:
    """Run the main chat loop."""
    if not _startup():
        sys.exit(1)

    # State
    history: List[dict] = []       # {"role": str, "content": str}
    last_response: Optional[str] = None
    last_sources: List[str] = []
    consecutive_errors: int = 0

    session = PromptSession(history=InMemoryHistory())

    try:
        while True:
            # ── Read Input ───────────────────────────────────────────
            try:
                user_input = _read_multiline(session)
            except SystemExit:
                break

            if user_input is None:
                continue

            # ── Handle Commands ──────────────────────────────────────
            if user_input.startswith("/"):
                result = _handle_command(
                    user_input, history, last_response, last_sources
                )
                if result == "exit":
                    break
                continue

            # ── RAG Pipeline ─────────────────────────────────────────
            try:
                # Step 1: Retrieve context
                t0 = time.time()
                inline_status("Embedding query...", "...", "info")
                chunks = retrieve(user_input)
                t1 = time.time()

                source_names = get_source_names(chunks)
                last_sources = source_names

                source_str = " · ".join(source_names) if source_names else "none"
                inline_status(
                    f"Retrieving context (K={len(chunks)})...",
                    f"✓  ({source_str})", "success",
                )

                # Step 2: Build prompt
                prompt = build_prompt(user_input, chunks, history)
                token_count = get_prompt_token_count(prompt)
                inline_status(
                    "Assembling prompt...",
                    f"✓  ({token_count:,} tokens)", "success",
                )

                # Step 3: Stream response
                inline_status(
                    "Streaming response from Gemini...", "", "info"
                )

                printer = StreamPrinter()
                console.print()

                for token in stream_generate(prompt):
                    printer.print_token(token)

                full_response = printer.finish()

                if not full_response.strip():
                    show_warning("Empty response from Gemini.")
                    consecutive_errors += 1
                else:
                    last_response = full_response

                    # Show sources
                    show_sources(source_names)

                    # Update history
                    history.append({"role": "user", "content": user_input})
                    history.append({
                        "role": "assistant", "content": full_response
                    })

                    # Reset error counter on success
                    consecutive_errors = 0

                console.print()

            except KeyboardInterrupt:
                console.print("\n  [dim]Response interrupted.[/dim]\n")
                continue

            except Exception as exc:
                show_error(f"Error: {exc}")
                consecutive_errors += 1

            # ── Circuit Breaker ──────────────────────────────────────
            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                console.print()
                show_error(
                    f"{MAX_CONSECUTIVE_ERRORS} consecutive errors. "
                    f"Please check your API key and internet connection."
                )
                console.print(
                    "  Run [cyan]python setup.py[/cyan] to reconfigure.\n"
                )
                break

    except KeyboardInterrupt:
        console.print("\n  [dim]Use /exit to quit.[/dim]")
    except EOFError:
        pass

    console.print("\n  [bold]Goodbye![/bold] 👋\n")


if __name__ == "__main__":
    main()
