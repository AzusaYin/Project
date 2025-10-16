import httpx
from typing import List, AsyncGenerator
from .settings import settings

async def smartcare_chat(messages: List[dict], stream: bool = False) -> AsyncGenerator[str, None] | str:
    payload = {
        "messages": messages,
        "temperature": settings.temperature,
        "max_tokens": settings.max_tokens,
        "stream": stream,
    }
    async with httpx.AsyncClient(timeout=60) as client:
        if stream:
            async with client.stream("POST", settings.smartcare_base_url, json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    yield line
        else:
            r = await client.post(settings.smartcare_base_url, json=payload)
            r.raise_for_status()
            data = r.json()
            # The API is OpenAI-style; adapt if needed
            return data