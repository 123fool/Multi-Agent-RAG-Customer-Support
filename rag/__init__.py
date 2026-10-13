"""
RAG (Retrieval-Augmented Generation) 模块
提供基于 Qdrant 向量数据库的知识检索服务
"""

from .knowledge_base import KnowledgeBase, get_knowledge_base
from .retriever import retrieve

__all__ = ["KnowledgeBase", "get_knowledge_base", "retrieve"]
