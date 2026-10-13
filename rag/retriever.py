"""
知识检索工具
提供便捷的检索接口供 Agent 调用
"""

from typing import List, Dict, Any

from .knowledge_base import get_knowledge_base


def retrieve(
    query: str,
    category: str,
    top_k: int = 5,
    score_threshold: float = 0.3,
) -> str:
    """
    检索相关知识并格式化为文本

    Args:
        query: 用户查询文本
        category: 知识分类 (products, tech_support, billing, complaints, general_service)
        top_k: 返回的最相关结果数量
        score_threshold: 最低相关度阈值

    Returns:
        格式化的知识文本，多条结果用换行分隔；无结果时返回空字符串
    """
    kb = get_knowledge_base()
    results = kb.search(
        query=query,
        category=category,
        top_k=top_k,
        score_threshold=score_threshold,
    )

    if not results:
        return ""

    formatted_parts = []
    for result in results:
        header = result["metadata"].get("header", "")
        header_prefix = f"【{header}】\n" if header else ""
        formatted_parts.append(f"{header_prefix}{result['text']}")

    return "\n\n---\n\n".join(formatted_parts)
