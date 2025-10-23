#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate a simple "Project Framework" file by concatenating discovered source files.

This script automatically discovers **all** Python files under the `app/` and `scripts/`
directories (recursively), in a stable, alphabetical order. It no longer requires you to
manually list paths. Optionally, it also appends `requirements.txt` and `README.md` if found.

Usage:
    python generate_project_framework.py [--output PROJECT_FRAMEWORK.md]

Notes:
- Excludes `__pycache__/` directories by default.
- Treats file paths relative to the repository root (the directory this script lives in).
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

# ---- Configuration ---------------------------------------------------------

ROOTS = ("app", "scripts")
INCLUDE_EXTRA = ("requirements.txt", "README.md")

BANNER = """\n# ============================================================================
# File: {path}
# ============================================================================
"""

# ---- Core ------------------------------------------------------------------

def discover_python_files(repo_root: Path) -> list[Path]:
    files: list[Path] = []
    for root in ROOTS:
        root_dir = repo_root / root
        if not root_dir.exists():
            continue
        # Collect *.py recursively, excluding __pycache__
        for p in sorted(root_dir.rglob("*.py")):
            if "__pycache__" in p.parts:
                continue
            files.append(p)
    return files

def collect_files(repo_root: Path) -> list[Path]:
    items = discover_python_files(repo_root)
    for extra in INCLUDE_EXTRA:
        p = repo_root / extra
        if p.exists():
            items.append(p)
    return items

def build_framework(repo_root: Path, targets: list[Path]) -> str:
    parts: list[str] = []
    for abs_path in targets:
        try:
            text = abs_path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            print(f"[WARN] Failed to read {abs_path}: {e}", file=sys.stderr)
            continue
        rel_path = abs_path.relative_to(repo_root).as_posix()
        parts.append(BANNER.format(path=rel_path))
        parts.append(text.rstrip() + "\n")
    if not parts:
        return ""
    return "".join(parts)

def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a Project Framework document.")
    parser.add_argument(
        "--output", "-o",
        default="PROJECT_FRAMEWORK.md",
        help="Output file path (default: PROJECT_FRAMEWORK.md)"
    )
    return parser.parse_args(argv)

def main(argv=None) -> None:
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parent

    targets = collect_files(repo_root)
    if not targets:
        print("[WARN] No files were found under 'app/' or 'scripts/'. Nothing to write.")
        return

    output_path = (repo_root / args.output).resolve()
    content = build_framework(repo_root, targets)
    if not content:
        print("[WARN] Nothing to write (empty content).")
        return

    output_path.write_text(content, encoding="utf-8")
    print(f"[OK] Project Framework generated → {output_path}")

if __name__ == "__main__":
    main()
