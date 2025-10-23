import uvicorn
import re, json, time
from pathlib import Path
from typing import List, Dict
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import StreamingResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from .settings import settings
from .schemas import ChatRequest, ChatAnswer, IngestResponse, FeedbackIn
from .rag import Index, Embedder, hybrid_retrieve, build_prompt, format_citations, ingest_corpus
from .llm_client import smartcare_chat, smartcare_chat_stream, smartcare_translate_to_en
from .security import require_bearer
from .admin_docs import router as admin_docs_router

app = FastAPI(title="ElderlyCare HK — Backend")
app.include_router(admin_docs_router)
_CITE_TAG_RE = re.compile(r"\[Source\s+(\d+)\]")

# 英/中政策名与常见后缀
_ENTITY_PAT = re.compile(
    r"\b([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,6}\s(?:Allowance|Scheme|Manual|Programme|Grant|Service|Subvention|System))\b"
    r"|(?:Old Age Allowance|Old Age Living Allowance|Operating Subvented Welfare|LSG Subvention Manual)"
    r"|(?:長者生活津貼|高齡津貼|老年津貼|資助福利服務|統一撥款|資助手冊|計劃|津貼|手冊)",
    re.I
)
_PRONOUN_PAT = re.compile(r"\b(it|its|this|that|they|their)\b|[它其這該]")

# 简单别名映射（可继续补充）
ALIASES = {
    "OAA": "Old Age Allowance",
    "OALA": "Old Age Living Allowance",
    "LSG": "Lump Sum Grant",
    "LSGSS": "Lump Sum Grant Subvention System",
}

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")  # 基本漢字

# —— 从文本中抽取“政策/計劃/津貼/手冊”等名稱 —— 
_ENTITY_EXTRACT_RE = re.compile(
    r"\b([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,7}\s(?:Allowance|Scheme|Manual|Programme|Grant|Service|Subvention|System|Policy))\b"
    r"|(?:Old Age Allowance|Old Age Living Allowance|Disability Allowance|Comprehensive Social Security Assistance)"
    r"|(?:長者生活津貼|高齡津貼|老年津貼|傷殘津貼|綜合社會保障援助|資助福利服務|統一撥款|資助手冊|撥款制度|政策|計劃|津貼|手冊)",
    re.I
)

def _extract_entities_from_text(text: str) -> list[str]:
    if not text: 
        return []
    return [m.group(0).strip() for m in _ENTITY_EXTRACT_RE.finditer(text)]

def _suggest_entities_for(query: str, idx: "Index", emb: "Embedder", top_k: int = 5) -> list[str]:
    """
    用當前查詢在索引裡做一次輕量檢索，從命中的 chunk 文本中抽取實體名，按頻次去重排序返回。
    """
    try:
        contexts = hybrid_retrieve(query, idx, emb, k=20, soft=True)
    except Exception:
        return []
    freq: dict[str, int] = {}
    for c in contexts:
        t = c["meta"].get("text") or ""
        for ent in _extract_entities_from_text(t):
            # 合併大小寫/空白差異
            key = re.sub(r"\s+", " ", ent).strip()
            freq[key] = freq.get(key, 0) + 1
    # 排序 + 去掉過於籠統的詞
    bad = {"Allowance","Scheme","Manual","Programme","Grant","Service","Subvention","System","Policy","計劃","津貼","手冊","政策"}
    items = [e for e,_ in sorted(freq.items(), key=lambda kv: kv[1], reverse=True) if e not in bad]
    # 保留不同前綴的前 top_k 個
    out = []
    seen_lower = set()
    for e in items:
        low = e.lower()
        if low in seen_lower: 
            continue
        seen_lower.add(low)
        out.append(e)
        if len(out) >= top_k:
            break
    return out

def _looks_cjk(s: str) -> bool:
    return bool(_CJK_RE.search(s or ""))

import numpy as np
# 轻量启发式参数
_MIN_TOKENS = 3
_GENERIC_Q_RE = re.compile(
    r"^(what|how|why|tell me|can you|could you|explain|give me|i want to know|說說|介紹|解釋|請講講|我想知道)\b",
    re.I,
)

def _tokenize_simple(s: str) -> list[str]:
    return [t for t in re.findall(r"\w+|[\u4e00-\u9fff]", s or "") if t.strip()]

def _is_ambiguous_heuristic(q: str) -> bool:
    q = (q or "").strip()
    if not q:
        return True
    toks = _tokenize_simple(q)
    if len(toks) <= _MIN_TOKENS:
        return True
    uniq_ratio = len(set(toks)) / max(1, len(toks))
    if uniq_ratio < 0.5:
        return True
    has_named = bool(re.search(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,4})\b", q))
    if _GENERIC_Q_RE.search(q) and not has_named:
        return True
    return False

# 语义模板（极小集合，捕捉“泛问/扩写/解释一下”这类）
_GENERIC_TEMPLATES = [
    "Tell me about it",
    "Tell me about this",
    "What is it",
    "Explain this",
    "Give me details",
    "I want to know more",
    "What about it",
    "請介紹一下",
    "這是什麼",
    "說說看",
]

_GENERIC_EMB: np.ndarray | None = None  # 懒加载缓存

def _ensure_generic_emb() -> np.ndarray:
    global _GENERIC_EMB
    if _GENERIC_EMB is None:
        emb = get_embedder()  # 复用现有的 SentenceTransformer（已做 normalize）
        _GENERIC_EMB = emb.encode(_GENERIC_TEMPLATES)
        if _GENERIC_EMB.ndim == 1:
            _GENERIC_EMB = _GENERIC_EMB.reshape(1, -1)
    return _GENERIC_EMB

def _is_ambiguous_semantic(q: str, thr: float = 0.78) -> bool:
    q = (q or "").strip()
    if not q:
        return True
    emb = get_embedder()
    qe = emb.encode([q])[0]  # 已归一化
    G = _ensure_generic_emb()  # (m, d)
    sims = (G @ qe)  # 余弦相似度
    return float(np.max(sims)) >= thr

def _should_clarify_smart(user_query: str) -> bool:
    # A：启发式先判
    if _is_ambiguous_heuristic(user_query):
        return True
    # B：与“泛问模板”相似则判模糊
    if _is_ambiguous_semantic(user_query):
        return True
    return False

def _clarify_question(user_query: str, lang: str | None) -> str:
    """
    当检测到用户问题模糊时，生成一条追问句，用于提示用户具体化问题。
    """
    if not user_query:
        user_query = "your question"

    # 中文界面
    if lang == "zh-Hant":
        return (
            f"你的問題（「{user_query}」）目前範圍過大。"
            "我在相關的官方文檔中找不到有關該主題的信息。"
        )

    # 英文界面
    return (
        f"Your question (“{user_query}”) is a bit broad. "
        "I couldn't find any information on that topic in documents related to elderly care."
    )

def _clarify_question_smart(user_query: str, lang: str | None, idx: "Index", emb: "Embedder") -> str:
    q = (user_query or "").strip()
    # 先做候選：比如 "allowance", "policy", "scheme", "津貼", "政策"
    keywords = ["allowance", "policy", "scheme", "manual", "programme", "津貼", "政策", "計劃", "手冊"]
    need_list = any(kw in q.lower() for kw in keywords) or _is_ambiguous_semantic(q)

    if need_list:
        cands = _suggest_entities_for(q, idx, emb, top_k=5)
        if cands:
            if lang == "zh-Hant":
                opts = "、".join(cands[:4])
                return f"你的問題較為籠統（「{q}」）。你是在問 {opts}，還是其他？"
            else:
                opts = ", ".join(cands[:4])
                return f'Your question (“{q}”) is a bit broad. Are you asking about {opts}, or something else?'

    # 候選空時，回退到原來的通用提示
    return _clarify_question(user_query, lang)

def _expand_aliases(text: str) -> str:
    out = text
    for k, v in ALIASES.items():
        out = re.sub(rf"\b{k}\b", v, out, flags=re.I)
    return out

def _guess_entity_from_history(msgs: list[dict]) -> str | None:
    # 从后往前找最近出现的“明确名词”，优先 assistant，再到 user
    for m in reversed(msgs):
        txt = _expand_aliases((m.get("content") or "").strip())
        if not txt:
            continue
        hit = _ENTITY_PAT.search(txt)
        if hit:
            return hit.group(0)
    # 如果还没有，最后退回到首问中的名词
    for m in msgs:
        txt = _expand_aliases((m.get("content") or "").strip())
        hit = _ENTITY_PAT.search(txt)
        if hit:
            return hit.group(0)
    return None

_FEEDBACK_DIR = Path("data/feedback")
_FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
_FEEDBACK_PATH = _FEEDBACK_DIR / "feedback.jsonl"
_METRICS_PATH  = _FEEDBACK_DIR / "metrics.json"   # 用于极简在线指标

def _append_jsonl(path: Path, obj: dict):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def _bump_metrics(label: str):
    # 超迷你在线指标：累计 up/down 次数；可扩展为分桶(相似度区间)统计
    m = {"up": 0, "down": 0}
    if _METRICS_PATH.exists():
        try: m = json.loads(_METRICS_PATH.read_text(encoding="utf-8"))
        except Exception: pass
    m[label] = m.get(label, 0) + 1
    _METRICS_PATH.write_text(json.dumps(m), encoding="utf-8")

@app.post("/feedback")
async def feedback_in(body: FeedbackIn, _auth=Depends(require_bearer)):
    rec = body.model_dump()
    rec["ts"] = int(time.time() * 1000)
    try:
        _append_jsonl(_FEEDBACK_PATH, rec)
        _bump_metrics(body.label)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"write feedback failed: {e}")
    return {"ok": True}

@app.middleware("http")
async def add_pna_header(request: Request, call_next):
    # 让 Chrome 的 Private Network Access 预检通过
    response: Response = await call_next(request)
    response.headers["Access-Control-Allow-Private-Network"] = "true"
    return response

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_index: Index | None = None
_embedder: Embedder | None = None

def _extract_used_indices(answer_text: str, max_k: int) -> List[int]:
    """从答案正文里提取实际被使用的 [Source n]，去重并按首次出现顺序返回。"""
    seen = set()
    order: List[int] = []
    for m in _CITE_TAG_RE.finditer(answer_text or ""):
        try:
            n = int(m.group(1))
        except Exception:
            continue
        if 1 <= n <= max_k and n not in seen:
            seen.add(n)
            order.append(n)
    return order

def _citations_by_usage(answer_text: str, contexts: List[Dict]) -> List[Dict]:
    """只返回正文里实际使用到的 [Source n] 所对应的 citations。"""
    from .rag import format_citations
    all_cites = format_citations(contexts)  # 顺序与 [Source 1..K] 对应
    used = _extract_used_indices(answer_text, len(all_cites))
    return [all_cites[i - 1] for i in used]  # i 从 1 开始

def _sanitize_inline_citations(text: str, max_k: int) -> str:
    def repl(m):
        n = int(m.group(1))
        return m.group(0) if 1 <= n <= max_k else ""
    return _CITE_TAG_RE.sub(repl, text)

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
async def ingest(_auth=Depends(require_bearer)):
    docs_count, chunks_count = ingest_corpus(settings.docs_dir, settings.index_dir)
    global _index, _embedder
    _index = None
    _embedder = None
    return IngestResponse(documents_indexed=docs_count, chunks_indexed=chunks_count)

def _extract_answer_text(data: dict) -> str:
    # 新增：处理 "answer" 是嵌套字符串 JSON 的情况
    if "answer" in data and isinstance(data["answer"], str):
        try:
            parsed = json.loads(data["answer"].replace("'", '"'))  # 兼容单引号
            if isinstance(parsed, dict) and "response" in parsed:
                return parsed["response"]
        except Exception:
            pass  # fallback below

    try:
        return data["choices"][0]["message"]["content"]
    except Exception:
        pass
    if isinstance(data, dict) and "choices" in data and data["choices"]:
        c0 = data["choices"][0]
        if isinstance(c0, dict) and "text" in c0 and isinstance(c0["text"], str):
            return c0["text"]
    for key in ("content", "answer", "message", "data"):
        v = data.get(key)
        if isinstance(v, str) and v.strip():
            return v
        if isinstance(v, dict):
            for kk in ("content", "text", "answer"):
                vv = v.get(kk)
                if isinstance(vv, str) and vv.strip():
                    return vv
    return str(data)

def _normalize_answer_text(text: str) -> str:
    if not isinstance(text, str):
        return str(text)

    # 1) 解析 {"response": "..."} 或 {'response': '...'}
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and isinstance(obj.get("response"), str):
            text = obj["response"]
    except Exception:
        m = re.match(r"\s*\{[^}]*'response'\s*:\s*'(.*?)'\s*\}\s*$", text, flags=re.S)
        if m:
            text = m.group(1)

    # 2) 去掉文末内嵌的 Sources 段
    text = re.sub(r"\n+Sources\s*\n(?:\[[^\n]+\].*\n?)+\s*$", "", text, flags=re.I)

    # 3) 统一换行，并把字面量 \\n 转成真正换行
    text = text.replace("\r\n", "\n")
    if "\\n" in text:
        text = text.replace("\\n", "\n")

    return text.strip()

def _extract_stream_token_preserve(text_line: str) -> str | None:
    line = text_line
    if not line:
        return None

    # 处理 SSE 前缀
    s = line.strip()
    if s.lower() == "[done]":
        return None
    if s.startswith("data:"):
        s = s[5:].strip()

    # 尝试解析 JSON；如果不是 JSON，直接返回原行（不改动）
    try:
        obj = json.loads(s)
    except Exception:
        return line  # 非 JSON，原样返回（保留其中换行/空格）

    # 在常见路径中取字符串值（原样返回）
    def get_path(o, path):
        cur = o
        for p in path:
            if isinstance(p, int):
                if isinstance(cur, list) and len(cur) > p:
                    cur = cur[p]
                else:
                    return None
            else:
                if isinstance(cur, dict) and p in cur:
                    cur = cur[p]
                else:
                    return None
        return cur if isinstance(cur, str) else None

    candidates = [
        ("choices", 0, "delta", "content"),
        ("choices", 0, "text"),
        ("response",),
        ("content",),
        ("answer",),
        ("message", "content"),
        ("data", "content"),
    ]
    for path in candidates:
        val = get_path(obj, path)
        if isinstance(val, str):
            return val  # 原样返回，不做任何替换或去除

    # 没命中就把整行 JSON 丢弃（避免再把 JSON 文本回传前端）
    return None

def _extract_stream_piece(line: str) -> str:
    """
    从 SmartCare 流式每行里提取纯文本。
    兼容以下情况：
      - {"response": " ... "}
      - {"text": "..."} / {"content": "..."} / {"data": "..."}
      - 纯文本（直接返回）
      - SSE 风格的 'data: {...}'（可选兼容）
    """
    if not line:
        return ""

    # 1) 去掉可能的 SSE 前缀
    if line.startswith("data:"):
        line = line[len("data:"):].strip()

    # 2) 优先 JSON 解析
    try:
        obj = json.loads(line)
        if isinstance(obj, dict):
            for key in ("response", "text", "content", "data"):
                val = obj.get(key)
                if isinstance(val, str):
                    return val
        # 如果是数组或其他结构，这里不处理
    except Exception:
        pass

    # 3) 兼容非严格 JSON，简单用正则兜底（例如 {'response': '...'}）
    m = re.search(r"'response'\s*:\s*'(.*?)'", line)
    if m:
        return m.group(1)

    # 4) 如果 line 不是 JSON，就当作纯文本（少见但安全）
    #    但要排除纯粹的空白/心跳
    if line.strip():
        return line
    return ""

def _not_found_text(lang: str | None) -> str:
    if lang == "zh-Hant":
        return ("抱歉，我在目前納入的社會福利文件中沒有找到你這個問題的具體答案。"
                "你可以嘗試：\n"
                "• 換一種說法或補充更具體的名詞（例如津貼名稱、服務單位）\n"
                "• 指定文件或年份（例如 2024 年 LSG Subvention Manual）\n"
                "如果需要，我可以幫你重述查詢或列出相關章節供你查閱。")
    return ("Sorry, I couldn't find a specific answer to this question in the indexed documents.\n"
            "You can try:\n"
            "• Rephrasing or adding more specific terms (e.g., the exact allowance name/service unit)\n"
            "• Mentioning a document or year (e.g., 2024 LSG Subvention Manual)\n"
            "If you like, I can help refine your query or show nearby sections.")

from .llm_client import smartcare_chat, smartcare_chat_stream, smartcare_rewrite_query
@app.post("/chat", response_model=ChatAnswer)
async def chat(req: ChatRequest, _auth=Depends(require_bearer)):
    idx = get_index()
    emb = get_embedder()

    # 1) 取得 messages（前端现在会带最近若干轮）
    msgs = [m.dict() if hasattr(m, "dict") else m for m in req.messages]

    # 2) 改写为独立问题（有历史时更有效）
    #    如果 messages 很短（只有一条 user），就直接用它
    if len(msgs) >= 2:
        rewritten = await smartcare_rewrite_query(msgs)
        user_query = rewritten
    else:
        user_query = msgs[-1]["content"]

    # ★ 規則兜底：若仍含代詞，嘗試補上最近實體
    if _PRONOUN_PAT.search(user_query):
        ent = _guess_entity_from_history(msgs)
        if ent:
            # 不粗暴替换用户原句，只给检索信号加注释
            user_query = f"{user_query} (about {ent})"
    
    # 3) 用改写后的独立问题进行检索
    is_followup_pronoun = bool(_PRONOUN_PAT.search(msgs[-1]["content"]))
    contexts = hybrid_retrieve(user_query, idx, emb, settings.top_k, soft=is_followup_pronoun)
    
    # === 先判斷是否找得到來源 ===
    if len(contexts) < settings.min_sources_required:
        text = _not_found_text(req.language)
        if req.stream:
            async def event_stream():
                yield text
                yield "\nCITATIONS:[]\n"
            return StreamingResponse(event_stream(), media_type="text/plain")
        else:
            return ChatAnswer(answer=text, citations=[])

    # # 若命中不足，且疑似 CJK 查詢 → 翻譯成英文後重試一次
    # if len(contexts) < settings.min_sources_required and _looks_cjk(user_query):
    #     try:
    #         q_en = await smartcare_translate_to_en(user_query)
    #         contexts2 = hybrid_retrieve(q_en, idx, emb, settings.top_k, soft=True)
    #         if len(contexts2) >= len(contexts):
    #             user_query = q_en  # 記錄實際用來檢索的查詢
    #             contexts = contexts2
    #     except Exception:
    #         pass
    
    # === 再檢查是否過於籠統（但已有來源） ===
    if _should_clarify_smart(user_query):
        text = _clarify_question_smart(user_query, req.language, idx, emb)
        if req.stream:
            async def event_stream():
                yield text
                yield "\nCITATIONS:[]\n"
            return StreamingResponse(event_stream(), media_type="text/plain")
        else:
            return ChatAnswer(answer=text, citations=[])


    # 4) 正常拼 prompt（把原 messages 发给模型，这样它能“按上下文口吻”回答）
    prompt_msgs = build_prompt(msgs, contexts)
    if req.language == "zh-Hant":
        prompt_msgs.insert(0, {
            "role": "system",
            "content": "請使用繁體中文回答所有問題。"
        })
    elif req.language == "en":
        prompt_msgs.insert(0, {
            "role": "system",
            "content": "Please answer in English."
        })

    if req.stream:
        async def event_stream():
            buffer = ""
            async for line in smartcare_chat_stream(prompt_msgs):
                piece = _extract_stream_piece(line)
                if piece:
                    buffer += piece
                    yield piece
            clean_final = _sanitize_inline_citations(buffer, len(contexts))
            cites = _citations_by_usage(clean_final, contexts)
            trailer = "CITATIONS:" + json.dumps(cites, ensure_ascii=False)
            yield "\n" + trailer + "\n"
        return StreamingResponse(event_stream(), media_type="text/plain")

    data = await smartcare_chat(prompt_msgs)
    answer_text = _normalize_answer_text(_extract_answer_text(data))
    answer_text = _sanitize_inline_citations(answer_text, len(contexts))
    citations = _citations_by_usage(answer_text, contexts)
    return ChatAnswer(answer=answer_text, citations=citations)

# from fastapi.responses import PlainTextResponse
# @app.post("/chat/plain", response_class=PlainTextResponse)
# async def chat_plain(req: ChatRequest):
#     idx = get_index()
#     emb = get_embedder()

#     # ========= 新增：构造 msgs 并判断是否使用改写 =========
#     msgs = [m.dict() if hasattr(m, "dict") else m for m in req.messages]

#     if settings.enable_query_rewrite and len(msgs) >= 2:
#         rewritten = await smartcare_rewrite_query(msgs)
#         user_query = rewritten
#     else:
#         user_query = msgs[-1]["content"]

#     # ========= 然后再进行检索 =========
#     is_followup_pronoun = bool(_PRONOUN_PAT.search(msgs[-1]["content"]))
#     contexts = hybrid_retrieve(user_query, idx, emb, settings.top_k)
#     prompt_msgs = build_prompt([m.dict() for m in req.messages], contexts)

#     data = await smartcare_chat(prompt_msgs)
#     answer_text = _extract_answer_text(data)

#     # 规范化转义换行
#     if "\\n" in answer_text and "\n" not in answer_text:
#         answer_text = answer_text.replace("\\n", "\n")

#     # 把引用也附在文本末尾（逐行）
#     cites = format_citations(contexts)
#     lines = [answer_text, "", "References:"]
#     for i, c in enumerate(cites, 1):
#         page = f", page {c['page']}" if c.get("page") is not None else ""
#         lines.append(f"[Source {i}] {c['file']}{page}")

#     return "\n".join(lines)

if __name__ == "__main__":
    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=True)
