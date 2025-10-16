import os, re, json
from pathlib import Path
from typing import List, Dict

def read_markdown_files(docs_dir: str) -> List[Dict]:
    paths = list(Path(docs_dir).glob("**/*.md")) + list(Path(docs_dir).glob("**/*.markdown"))
    docs = []
    for p in paths:
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
            docs.append({"path": str(p), "text": text})
        except Exception as e:
            print(f"[WARN] Failed to read {p}: {e}")
    return docs

_PAGE_RE = re.compile(r"(?:^|\n)\s*Page\s*(\d+)\s*(?:\n|$)", re.IGNORECASE)

def infer_page_map(md_text: str) -> List[Dict]:
    """Best-effort: detect 'Page N' markers in markdown; returns ranges with page numbers.
    If none, return a single segment with page=None.
    """
    pages = []
    matches = list(_PAGE_RE.finditer(md_text))
    if not matches:
        return [{"page": None, "start": 0, "end": len(md_text)}]
    for i, m in enumerate(matches):
        start = m.start()
        page_no = int(m.group(1))
        end = matches[i+1].start() if i+1 < len(matches) else len(md_text)
        pages.append({"page": page_no, "start": start, "end": end})
    return pages