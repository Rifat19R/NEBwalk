"""Run existing NEBwalk development gates without altering scientific data.

Quick mode is a focused local regression check, not a release or physics
certification. Full mode mirrors the core quality gates in CI, without its
Python-version matrix or wheel-installation smoke test.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable

FOCUSED_TESTS = (
    "tests/test_architecture.py",
    "tests/test_public_api.py",
    "tests/test_ase_comparison_benchmark.py",
    "tests/test_neb.py",
    "tests/test_forces.py",
    "tests/test_interpolate.py",
    "tests/test_recovery.py",
    "tests/test_label.py",
    "tests/test_labeling.py",
)


def run(label: str, command: list[str], list_only: bool = False) -> None:
    """Print, then execute one command; stop immediately on any failure."""
    print(f"\n[{label}] {shlex.join(command)}", flush=True)
    if list_only:
        return
    try:
        result = subprocess.run(command, cwd=ROOT, check=False)
    except FileNotFoundError as exc:
        print(
            f"Missing command: {exc.filename}. "
            'Install dependencies with: python -m pip install -e ".[dev,docs]"',
            file=sys.stderr,
        )
        raise SystemExit(2) from exc
    if result.returncode:
        print(f"FAILED: {label} (exit {result.returncode})", file=sys.stderr)
        raise SystemExit(result.returncode)


def standard_checks(full: bool, list_only: bool) -> None:
    """Run common gates, and the expanded gates when requested."""
    run("Ruff lint", ["ruff", "check", "."], list_only)
    run("Ruff format", ["ruff", "format", "--check", "."], list_only)
    if full:
        run("mypy", ["mypy", "src/NEBwalk"], list_only)
        run(
            "Complete pytest coverage",
            [
                PYTHON,
                "-m",
                "pytest",
                "tests/",
                "-v",
                "--cov=NEBwalk",
                "--cov-report=term-missing",
                "--cov-report=xml",
            ],
            list_only,
        )
    else:
        run(
            "Focused scientific/API regression",
            [PYTHON, "-m", "pytest", "-q", *FOCUSED_TESTS],
            list_only,
        )


def full_build_checks(list_only: bool) -> None:
    """Validate clean build artifacts and docs in an isolated temp folder."""
    if list_only:
        run(
            "Package build",
            [PYTHON, "-m", "build", "--outdir", "<TEMP_DIST>"],
            True,
        )
        run(
            "Twine metadata",
            [PYTHON, "-m", "twine", "check", "<BUILT_ARCHIVES>"],
            True,
        )
        run(
            "Strict documentation",
            [
                "mkdocs",
                "build",
                "--strict",
                "--site-dir",
                "<TEMP_SITE>",
            ],
            True,
        )
        return

    with tempfile.TemporaryDirectory(prefix="nebwalk-verify-") as temporary:
        dist_dir = Path(temporary) / "dist"
        site_dir = Path(temporary) / "site"
        run(
            "Package build",
            [PYTHON, "-m", "build", "--outdir", str(dist_dir)],
        )
        archives = sorted(str(path) for path in dist_dir.iterdir() if path.is_file())
        if not archives:
            print("FAILED: build produced no archives", file=sys.stderr)
            raise SystemExit(1)
        run("Twine metadata", [PYTHON, "-m", "twine", "check", *archives])
        run(
            "Strict documentation",
            ["mkdocs", "build", "--strict", "--site-dir", str(site_dir)],
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--quick", action="store_true", help="Focused regression gate (default)."
    )
    modes.add_argument(
        "--full", action="store_true", help="Full local quality/build gate."
    )
    parser.add_argument(
        "--list", action="store_true", help="List commands without running them."
    )
    args = parser.parse_args()

    print(f"NEBwalk root: {ROOT}", flush=True)
    print(f"Mode: {'full' if args.full else 'quick'}", flush=True)
    standard_checks(full=args.full, list_only=args.list)
    if args.full:
        full_build_checks(list_only=args.list)
    print("\nCommands listed." if args.list else "\nAll selected checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
