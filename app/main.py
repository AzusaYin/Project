import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from typing import List

from .settings import settings
from .schemas import ChatRequest, ChatAnswer, IngestResponse
from .rag import Index, Embedder, hybrid_retrieve, build_prompt, format_citations
from .rag import ingest_corpus

app = FastAPI(title="ElderlyCare HK — Backend")

# Lazy singletons
_index: Index | None = None
_embedder: Embedder | None = None

def get_index() -> Index:
    global _index
    if _index is None:
        idx = Index(settings.index_dir)
        idx.load()
        if idx.faiss is None:
            raise HTTPException(status_code=503, detail="Index not built yet. Run /ingest or scripts/ingest.py")
        _index = idx
    return _index

def get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Embedder(settings.embedding_model_name, settings.embedding_device)
    return _embedder

@app.get("/healthz")
async def healthz():
    return {"status": "ok"}

@app.post("/ingest", response_model=IngestResponse)
async def ingest():
    docs_count, chunks_count = ingest_corpus(settings.docs_dir, settings.index_dir)
    # reset singletons
    global _index, _embedder
    _index = None
    _embedder = None
    return IngestResponse(documents_indexed=docs_count, chunks_indexed=chunks_count)

from .llm_client import smartcare_chat

@app.post("/chat", response_model=ChatAnswer)
async def chat(req: ChatRequest):
    idx = get_index()
    emb = get_embedder()

    contexts = hybrid_retrieve(req.messages[-1].content, idx, emb, settings.top_k)
    prompt_msgs = build_prompt([m.dict() for m in req.messages], contexts)

    if req.stream:
        async def event_stream():
            async for token in smartcare_chat(prompt_msgs, stream=True):
                # Forward raw lines as SSE or NDJSON; here we simply yield text
                yield token + "\n"
        # We add citations in 'X-Citations' header after full completion in real app
        return StreamingResponse(event_stream(), media_type="text/plain")

    else:
        data = await smartcare_chat(prompt_msgs, stream=False)
        # Extract final answer text (OpenAI-style schema assumed)
        try:
            answer_text = data["choices"][0]["message"]["content"]
        except Exception:
            raise HTTPException(status_code=502, detail="Unexpected response from SmartCare API")
        citations = format_citations(contexts)
        return ChatAnswer(answer=answer_text, citations=citations)

if __name__ == "__main__":
    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=True)