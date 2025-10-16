from pydantic import BaseModel
from typing import List, Optional, Literal

class IngestResponse(BaseModel):
    documents_indexed: int
    chunks_indexed: int

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    stream: bool = False
    language: Optional[Literal["en", "zh-Hant"]] = None  # optional hint

class Citation(BaseModel):
    file: str
    page: Optional[int] = None
    snippet: Optional[str] = None

class ChatAnswer(BaseModel):
    answer: str
    citations: List[Citation]