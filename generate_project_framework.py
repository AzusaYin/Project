#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate a simple "Project Framework" file by concatenating specific source files.

It collects the following files in this exact order (if they exist):
    app/settings.py
    app/schemas.py
    app/utils.py
    app/rag.py
    app/llm_client.py
    app/main.py
    scripts/ingest.py
    requirements.txt
    README.md

Output: Project Framework.py
"""

from pathlib import Path

ORDER = [
    "app/settings.py",
    "app/schemas.py",
    "app/utils.py",
    "app/rag.py",
    "app/llm_client.py",
    "app/main.py",
    "app/security.py",
    "scripts/ingest.py",
    "scripts/build_penalty.py",
    "scripts/tune_thresholds.py",
    "requirements.txt",
    "README.md",
]

BANNER = "# ========================= {path} ========================="
OUT_FILE = "Project Framework.py"


def main():
    root = Path(".").resolve()
    output_path = root / OUT_FILE
    parts = []

    for rel_path in ORDER:
        p = root / rel_path
        if not p.exists():
            print(f"[SKIP] {rel_path} not found.")
            continue

        print(f"[ADD]  {rel_path}")
        try:
            text = p.read_text(encoding="utf-8", errors="ignore").rstrip()
        except Exception as e:
            text = f"[ERROR reading file: {e}]"

        banner = BANNER.format(path=rel_path)
        parts.append(f"{banner}\n{text}\n")

    if not parts:
        print("[WARN] No files were found. Nothing to write.")
        return

    output_path.write_text("\n".join(parts) + "\n", encoding="utf-8")
    print(f"\n[OK] Project Framework generated → {output_path}")


if __name__ == "__main__":
    main()
