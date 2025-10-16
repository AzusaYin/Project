import os
import json
import faiss
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from .settings import settings
from .utils import read_markdown_files, infer_page_map

@dataclass
class Chunk:
    text: str
    meta: Dict

class Embedder:
    def __init__(self, model_name: str, device: str = "cpu"):
        self.model = SentenceTransformer(model_name, device=device)
    def encode(self, texts: List[str], *, batch_size: Optional[int] = None, show_progress: Optional[bool] = None) -> np.ndarray:
        bs = batch_size if batch_size is not None else settings.embed_batch_size
        sp = settings.encode_show_progress if show_progress is None else show_progress
        return np.array(
            self.model.encode(
                texts,
                normalize_embeddings=True,
                batch_size=bs,
                show_progress_bar=sp,
            )
        )

class Index:
    def __init__(self, index_dir: str):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.faiss = None
        self.meta: List[Dict] = []
        self.bm25 = None
        self.bm25_corpus_tokens: List[List[str]] = []

    def build_empty(self, dim: int):
        self.faiss = faiss.IndexFlatIP(dim)

    def save(self):
        if self.faiss is not None:
            faiss.write_index(self.faiss, str(self.index_dir / "faiss.index"))
        (self.index_dir / "meta.json").write_text(json.dumps(self.meta, ensure_ascii=False), encoding="utf-8")
        if self.bm25 is not None:
            # Persist tokens (lightweight); BM25 can be rebuilt quickly
            (self.index_dir / "bm25.json").write_text(json.dumps({"tokens": self.bm25_corpus_tokens}, ensure_ascii=False))

    def load(self):
        faiss_path = self.index_dir / "faiss.index"
        meta_path = self.index_dir / "meta.json"
        if faiss_path.exists() and meta_path.exists():
            self.faiss = faiss.read_index(str(faiss_path))
            self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        bm25_path = self.index_dir / "bm25.json"
        if bm25_path.exists():
            data = json.loads(bm25_path.read_text())
            self.bm25_corpus_tokens = data.get("tokens", [])
            if self.bm25_corpus_tokens:
                self.bm25 = BM25Okapi(self.bm25_corpus_tokens)

    def build(self, embeddings: np.ndarray, meta: List[Dict], 
              bm25_tokens: Optional[List[List[str]]] = None):
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings.astype(np.float32))
        self.faiss = index
        self.meta = meta
        if bm25_tokens:
            self.bm25_corpus_tokens = bm25_tokens
            self.bm25 = BM25Okapi(self.bm25_corpus_tokens)

    def search(self, query_emb: np.ndarray, k: int) -> List[Tuple[int, float]]:
        D, I = self.faiss.search(query_emb.astype(np.float32), k)
        return list(zip(I[0].tolist(), D[0].tolist()))

# --- Chunking ---

def simple_char_chunk(text: str, chunk_size: int, overlap: int) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        start = end - overlap
        if start < 0:
            start = 0
    return chunks

# --- Ingestion ---

def ingest_corpus(docs_dir: str, index_dir: str) -> Tuple[int, int]:
    docs = read_markdown_files(docs_dir)
    embedder = Embedder(settings.embedding_model_name, settings.embedding_device)

    # ---- First pass: count chunks for progress ----
    total_chunks = 0
    for doc in docs:
        for pr in infer_page_map(doc["text"]):
            page_text = doc["text"][pr["start"]:pr["end"]]
            total_chunks += len(simple_char_chunk(page_text, settings.chunk_size, settings.chunk_overlap))

    # ---- Second pass: embed in small batches and build index incrementally ----
    index = Index(index_dir)
    dim_inited = False

    processed = 0
    buf_texts: List[str] = []
    buf_meta: List[Dict] = []
    buf_tokens: List[List[str]] = []

    def flush():
        nonlocal buf_texts, buf_meta, buf_tokens, dim_inited, processed, index
        if not buf_texts:
            return
        embs = embedder.encode(buf_texts, batch_size=settings.embed_batch_size, show_progress=False)
        embs = embs.astype(np.float32)
        if not dim_inited:
            index.build_empty(embs.shape[1])
            dim_inited = True
        index.faiss.add(embs)
        index.meta.extend(buf_meta)
        index.bm25_corpus_tokens.extend(buf_tokens)
        processed += len(buf_texts)
        print(f"[ingest] progress: {processed}/{total_chunks} ({processed*100.0/total_chunks:.1f}%)")
        buf_texts, buf_meta, buf_tokens = [], [], []

    for doc in docs:
        for pr in infer_page_map(doc["text"]):
            page_text = doc["text"][pr["start"]:pr["end"]]
            for i, piece in enumerate(simple_char_chunk(page_text, settings.chunk_size, settings.chunk_overlap)):
                buf_texts.append(piece)
                buf_meta.append({"file": doc["path"], "page": pr["page"], "chunk_id": i})
                buf_tokens.append(piece.lower().split())
                if len(buf_texts) >= settings.embed_batch_size:
                    flush()
    flush()  # final

    if settings.enable_bm25 and index.bm25_corpus_tokens:
        index.bm25 = BM25Okapi(index.bm25_corpus_tokens)

    index.save()
    return len(docs), total_chunks

# --- Retrieval ---

def hybrid_retrieve(query: str, index: Index, embedder: Embedder, k: int) -> List[Dict]:
    # Vector search
    q_emb = embedder.encode([query])
    vec_hits = index.search(q_emb, max(k, 5)) if index.faiss is not None else []

    # BM25 search (if enabled)
    bm25_hits = []
    if index.bm25 is not None:
        scores = index.bm25.get_scores(query.lower().split())
        top_ids = np.argsort(scores)[::-1][:max(k, 5)]
        bm25_hits = [(int(i), float(scores[i])) for i in top_ids if scores[i] > 0]

    # Merge by reciprocal rank fusion
    scores: Dict[int, float] = {}
    for rank, (idx, s) in enumerate(vec_hits):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (60 + rank)
    for rank, (idx, s) in enumerate(bm25_hits):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (60 + rank)

    merged = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
    results = []
    for idx, _ in merged:
        meta = index.meta[idx]
        results.append({
            "text": None,  # lazily load from disk if needed
            "meta": meta,
            "idx": idx
        })
    return results

# --- Build prompt with citations ---

def build_prompt(messages: List[Dict], contexts: List[Dict]) -> List[Dict]:
    citation_blocks = []
    for i, c in enumerate(contexts, 1):
        file = Path(c["meta"]["file"]).name
        page = c["meta"].get("page")
        ref = f"[Source {i}] {file}"
        if page is not None:
            ref += f", page {page}"
        snippet = "(content omitted in prompt to save tokens)"
        citation_blocks.append(f"{ref}\n{snippet}")

    system = {
        "role": "system",
        "content": (
            "You are ElderlyCare HK, a helpful assistant that answers questions using the provided Hong Kong Social Welfare Department documents. "
            "Cite sources at the end of your answer like [Source 1], [Source 2]. "
            "If you don't find the answer in the provided sources, say you couldn't find it in the documents."
        )
    }
    context_msg = {
        "role": "system",
        "content": (
            "Relevant sources:\n\n" + "\n\n".join(citation_blocks)
        )
    }
    return [system, context_msg] + messages

# --- Format final citations (for API response) ---

def format_citations(contexts: List[Dict]) -> List[Dict]:
    out = []
    for c in contexts:
        file = Path(c["meta"]["file"]).name
        page = c["meta"].get("page")
        out.append({"file": file, "page": page, "snippet": None})
    return out
