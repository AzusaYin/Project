from __future__ import annotations
from multiprocessing import Process
from pathlib import Path
import json, os, tempfile, time, traceback
from typing import Dict, Any, Optional
from .rag import ingest_corpus
from .settings import settings

STATUS_PATH = Path("data/status.json")
STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)

_proc: Optional[Process] = None
_note: str = ""
_start_ts: Optional[int] = None

def _write_status(payload: Dict[str, Any]) -> None:
    data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    with tempfile.NamedTemporaryFile(dir=str(STATUS_PATH.parent), delete=False) as tf:
        tf.write(data)
        tmpname = tf.name
    os.replace(tmpname, STATUS_PATH)

def _target(note: str):
    # 子进程里执行真正的重建
    _write_status({"status":"indexing", "note":note, "start_ts":int(time.time())})
    try:
        ingest_corpus(settings.docs_dir, settings.index_dir)
        _write_status({"status":"ready", "note":note, "last_built":int(time.time())})
    except Exception:
        err = traceback.format_exc()
        _write_status({"status":"error", "note":f"{note}: {err}", "ts":int(time.time())})

def start(note: str) -> bool:
    global _proc, _note, _start_ts
    if _proc is not None and _proc.is_alive():
        return False  # 已有任务在跑，返回 False
    _note = note
    _start_ts = int(time.time())
    _proc = Process(target=_target, args=(note,), daemon=True)
    _proc.start()
    return True

def cancel() -> bool:
    global _proc
    if _proc is None:
        return False
    alive = _proc.is_alive()
    if alive:
        _proc.terminate()
        _proc.join(timeout=5)
        _write_status({"status":"canceled", "note":"canceled by user", "ts":int(time.time())})
    _proc = None
    return alive

def status() -> Dict[str, Any]:
    s: Dict[str, Any]
    if STATUS_PATH.exists():
        try:
            s = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
        except Exception:
            s = {"status":"unknown"}
    else:
        s = {"status":"ready"}
    if _proc is not None:
        s["running"] = _proc.is_alive()
        s["pid"] = _proc.pid
    return s