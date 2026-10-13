"""OpenAI-compatible embedding model configuration."""

from langchain_openai import OpenAIEmbeddings
from config import (
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
)


def get_embeddings() -> OpenAIEmbeddings:
    """获取配置的 OpenAI 兼容嵌入模型实例。"""
    return OpenAIEmbeddings(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
        model=EMBEDDING_MODEL,
        check_embedding_ctx_length=False,
    )
