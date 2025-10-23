import httpx
from typing import List, AsyncGenerator
from .settings import settings

# stream & non-stream are split to avoid async generator return conflicts
async def smartcare_chat_stream(messages: List[dict]) -> AsyncGenerator[str, None]:
    payload = {
        "messages": messages,
        "temperature": settings.temperature,
        "max_tokens": settings.max_tokens,
        "stream": True,
    }
    async with httpx.AsyncClient(timeout=60) as client:
        async with client.stream("POST", settings.smartcare_base_url, json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if line:
                    yield line

async def smartcare_chat(messages: List[dict]) -> dict:
    payload = {
        "messages": messages,
        "temperature": settings.temperature,
        "max_tokens": settings.max_tokens,
        "stream": False,
    }
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(settings.smartcare_base_url, json=payload)
        r.raise_for_status()
        try:
            data = r.json()
        except Exception:
            text = r.text
            return {"choices": [{"message": {"role": "assistant", "content": text}}]}
        if isinstance(data, dict) and "choices" not in data:
            content = data.get("content") or data.get("answer") or data.get("message") or data.get("data")
            if isinstance(content, str):
                return {"choices": [{"message": {"role": "assistant", "content": content}}]}
        return data

async def smartcare_rewrite_query(messages: list[dict]) -> str:
    try:
        sys = {
            "role": "system",
            "content": (
                "You are a query rewriter. Rewrite ONLY the last user message into a standalone, context-complete question. "
                "Resolve pronouns (it/its/this/that/they/their; 它/其/這/該 等) to the most recent explicit entity mentioned "
                "in the conversation, especially policy/program names (e.g., Old Age Allowance, Old Age Living Allowance, LSG Subvention Manual).\n"
                "Keep the original language. Output the rewritten question text only.\n\n"
                "Examples:\n"
                "User history: What is Old Age Allowance?\n"
                "Last user: What are its eligibility requirements?\n"
                "Rewrite: What are the eligibility requirements for the Old Age Allowance?\n\n"
                "User history: 什麼是長者生活津貼？\n"
                "Last user: 申請要哪些文件？\n"
                "Rewrite: 長者生活津貼的申請需要哪些文件？"
            ),
        }
        payload_msgs = [sys] + messages
        data = await smartcare_chat(payload_msgs)
        text = data.get("choices", [{}])[0].get("message", {}).get("content") or ""
        return (text or "").strip() or messages[-1]["content"]
    except Exception:
        return messages[-1]["content"]

async def smartcare_translate_to_en(text: str) -> str:
    """
    将輸入翻譯為英文（保持原意、用於檢索），若失敗則回退原文。
    """
    try:
        sys = {
            "role": "system",
            "content": (
                "Translate the user's text into natural English suitable for information retrieval. "
                "Preserve the meaning. Output English only, no explanations."
            ),
        }
        data = await smartcare_chat([sys, {"role": "user", "content": text}])
        out = (data.get("choices", [{}])[0].get("message", {}) or {}).get("content", "") or ""
        return out.strip() or text
    except Exception:
        return text