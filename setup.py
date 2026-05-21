"""
setup.py — First-run setup wizard for the GROMACS Simulation Agent.

Run once on any new system. Collects the Gemini API key, validates it,
checks system requirements, and prepares the environment.
"""

import getpass
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv, set_key

from config import ENV_FILE, STRUCTURES_DIR
from display import (
    check_item,
    console,
    show_error,
    show_setup_banner,
    show_success,
    show_warning,
)

# Maximum API key validation attempts
MAX_KEY_ATTEMPTS = 3


def _collect_api_key() -> bool:
    """Collect and validate the Gemini API key from the user.

    Returns:
        True if a valid key was saved.
    """
    console.print("\n[bold]Step 1: Gemini API Key[/bold]\n")
    console.print(
        "  Get your free key at: [cyan]https://aistudio.google.com[/cyan]\n"
    )

    for attempt in range(1, MAX_KEY_ATTEMPTS + 1):
        try:
            key = getpass.getpass(
                f"  Enter your Gemini API key (attempt {attempt}/{MAX_KEY_ATTEMPTS}): "
            )
        except (EOFError, KeyboardInterrupt):
            console.print()
            show_error("Setup cancelled.")
            return False

        key = key.strip()

        # Format validation
        if not key.startswith("AIza"):
            show_warning("Key should start with 'AIza'. Please try again.")
            continue

        if len(key) < 35:
            show_warning("Key seems too short (min 35 characters). Please try again.")
            continue

        # Live test call
        console.print("  Testing API key...", end=" ")
        try:
            from gemini import test_api_key

            if test_api_key(key):
                # Save to .env
                if not ENV_FILE.exists():
                    ENV_FILE.write_text("", encoding="utf-8")
                set_key(str(ENV_FILE), "GEMINI_API_KEY", key)
                show_success("API key validated and saved to .env")
                return True
            else:
                show_warning("Key test failed. Please check your key.")
        except Exception as exc:
            show_error(f"Key test error: {exc}")

    show_error(
        f"{MAX_KEY_ATTEMPTS} attempts failed. Check your key at "
        f"aistudio.google.com then re-run setup.py."
    )
    return False


def _check_python_version() -> bool:
    """Check if Python version is 3.9+.

    Returns:
        True if version is adequate.
    """
    major, minor = sys.version_info[:2]
    version_str = f"{major}.{minor}.{sys.version_info[2]}"
    ok = (major, minor) >= (3, 9)
    check_item(
        f"Python version: {version_str}",
        ok,
        detail="" if ok else "3.9+ recommended",
        warn=not ok,
    )
    return ok


def _check_pip_packages() -> bool:
    """Check if required pip packages are installed.

    Returns:
        True if all packages are installed.
    """
    requirements_file = Path(__file__).resolve().parent / "requirements.txt"
    if not requirements_file.exists():
        check_item("requirements.txt", False, detail="file not found")
        return False

    packages = []
    for line in requirements_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            # Extract package name (before ==, >=, etc.)
            name = line.split("==")[0].split(">=")[0].split("<=")[0].strip()
            packages.append(name)

    missing = []
    for pkg in packages:
        try:
            __import__(pkg.replace("-", "_"))
        except ImportError:
            missing.append(pkg)

    if missing:
        check_item(
            "pip packages",
            False,
            detail=f"missing: {', '.join(missing)}",
            warn=True,
        )
        console.print()
        install = input("  Install missing packages now? [Y/n]: ").strip().lower()
        if install in ("", "y", "yes"):
            try:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", "-r",
                     str(requirements_file)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                show_success("Packages installed successfully.")
                return True
            except subprocess.CalledProcessError:
                show_error("Failed to install packages. Run manually:")
                console.print(f"  pip install -r {requirements_file}")
                return False
        return False
    else:
        check_item("pip packages", True, detail="all installed")
        return True


def _check_disk_space() -> bool:
    """Check available disk space (warn if below 300MB).

    Returns:
        True if adequate space is available.
    """
    try:
        usage = shutil.disk_usage(Path(__file__).resolve().parent)
        free_mb = usage.free / (1024 * 1024)
        ok = free_mb >= 300

        check_item(
            f"Disk space: {free_mb:.0f}MB free",
            ok,
            detail="" if ok else "300MB+ recommended for scraping",
            warn=not ok,
        )
        return ok
    except Exception:
        check_item("Disk space", True, detail="could not check", warn=True)
        return True


def _check_internet() -> bool:
    """Check internet connectivity to manual.gromacs.org.

    Returns:
        True if the site is reachable.
    """
    import requests

    try:
        resp = requests.head(
            "https://manual.gromacs.org", timeout=5, allow_redirects=True
        )
        ok = resp.status_code < 400
        check_item("Internet connectivity", ok,
                    detail="manual.gromacs.org reachable" if ok else "unreachable")
        return ok
    except requests.exceptions.RequestException:
        check_item(
            "Internet connectivity", False,
            detail="cannot reach manual.gromacs.org", warn=True,
        )
        return False


def _check_gromacs() -> bool:
    """Check if GROMACS is installed by running `gmx --version`.

    Returns:
        True if GROMACS is found.
    """
    try:
        result = subprocess.run(
            ["gmx", "--version"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            # Extract version from output
            for line in result.stdout.splitlines():
                if "GROMACS version" in line or "gromacs" in line.lower():
                    version = line.strip()
                    check_item("GROMACS", True, detail=version)
                    return True
            check_item("GROMACS", True, detail="installed")
            return True
        else:
            check_item(
                "GROMACS", False,
                detail="not found (optional — needed only for running simulations)",
                warn=True,
            )
            return False
    except FileNotFoundError:
        check_item(
            "GROMACS", False,
            detail="not found (optional — needed only for running simulations)",
            warn=True,
        )
        return False
    except Exception:
        check_item("GROMACS", False, detail="check failed", warn=True)
        return False


def _check_structures_dir() -> bool:
    """Create input_structures/ directory if missing.

    Returns:
        True always (directory is created if needed).
    """
    if not STRUCTURES_DIR.exists():
        STRUCTURES_DIR.mkdir(parents=True, exist_ok=True)
        check_item("input_structures/", True, detail="created")
    else:
        pdb_files = list(STRUCTURES_DIR.glob("*.pdb"))
        count = len(pdb_files)
        detail = f"{count} PDB file{'s' if count != 1 else ''} found"
        check_item("input_structures/", True, detail=detail)
    return True


def main() -> None:
    """Run the first-run setup wizard."""
    show_setup_banner()

    # ── API Key ──────────────────────────────────────────────────────────
    # Check if key already exists
    load_dotenv(str(ENV_FILE))
    existing_key = os.getenv("GEMINI_API_KEY", "").strip()

    if existing_key and existing_key != "your_key_here":
        console.print(
            "\n[dim]  API key found in .env. "
            "Use /key in the agent to change it.[/dim]"
        )
        key_ok = True
    else:
        key_ok = _collect_api_key()

    if not key_ok:
        sys.exit(1)

    # ── System Checks ────────────────────────────────────────────────────
    console.print("\n[bold]Step 2: System Requirements[/bold]\n")

    _check_python_version()
    _check_pip_packages()
    _check_disk_space()
    _check_internet()
    _check_gromacs()
    _check_structures_dir()

    # ── Summary ──────────────────────────────────────────────────────────
    console.print()
    console.rule("[bold green]Setup Complete[/bold green]")
    console.print(
        "\n  [bold]Next steps:[/bold]\n"
        "  1. Run [cyan]python ingest.py[/cyan]  "
        "  — scrape docs + build vector DB (first run only)\n"
        "  2. Run [cyan]python agent.py[/cyan]   "
        "  — start the chatbot\n"
    )


if __name__ == "__main__":
    main()
