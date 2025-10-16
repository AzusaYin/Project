from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import Optional

class Settings(BaseSettings):
    # Pydantic Settings v2 config
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,   # allow lower/upper env keys
        extra="ignore",         # ignore unknown keys instead of erroring
        env_ignore_empty=True,
    )

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True

    # Paths
    docs_dir: str = "data/docs"
    index_dir: str = "data/index"

    # RAG
    chunk_size: int = 900
    chunk_overlap: int = 200
    top_k: int = 5

    # Embeddings
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str = "cpu"        # 设置成 "cuda" 即可用 GPU
    embed_batch_size: int = 32           # GPU 可调 64~128；CPU 8~32
    encode_show_progress: bool = True

    # HKUST LLM API
    smartcare_base_url: str = "https://smartlab.cse.ust.hk/smartcare/dev/llm_chat/"
    temperature: float = 0.0
    max_tokens: int = 1024

    # Feature flags
    enable_bm25: bool = True

settings = Settings()
