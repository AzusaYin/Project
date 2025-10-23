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
    language: Optional[Literal["en", "zh-Hant"]] = None

class Citation(BaseModel):
    file: str
    page: Optional[int] = None
    snippet: Optional[str] = None

class ChatAnswer(BaseModel):
    answer: str
    citations: List[Citation]

class FeedbackIn(BaseModel):
    threadId: str
    messageId: str
    label: Literal["up", "down"]
    userQuery: str = ""
    answer: str = ""
    language: Optional[Literal["en", "zh-Hant"]] = None
    citations: List[Citation] = []
    # 可选：一些检索与响应元数据，便于分析/自动调参
    meta: Optional[dict] = None
