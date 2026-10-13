"""
知识库管理
使用 Qdrant 向量数据库（内存模式）存储和检索知识文档
"""

import os
from typing import List, Dict, Any, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, models

from .embeddings import get_embeddings, EMBEDDING_DIMENSION

CATEGORIES = [
    "products",
    "tech_support",
    "billing",
    "complaints",
    "general_service",
]

COLLECTION_PREFIX = "cs_"


class KnowledgeBase:
    """知识库管理类 - 管理多个分类的 Qdrant 集合"""

    def __init__(self, knowledge_dir: str = "knowledge"):
        self.knowledge_dir = knowledge_dir
        self.client: Optional[QdrantClient] = None
        self.embeddings = None
        self._initialized = False

    def initialize(self) -> None:
        """初始化 Qdrant 客户端（优先连接 Docker 服务器，回退到本地文件模式）"""
        from config import QDRANT_URL, QDRANT_API_KEY

        if QDRANT_URL:
            self.client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY or None)
            print(f"[Qdrant] Connected to server: {QDRANT_URL}")
        else:
            qdrant_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".qdrant_data")
            self.client = QdrantClient(path=qdrant_path)
            print("[Qdrant] Local file mode (no QDRANT_URL)")

        self.embeddings = get_embeddings()

        for category in CATEGORIES:
            collection_name = f"{COLLECTION_PREFIX}{category}"
            # 只在 collection 不存在时才创建
            existing = [c.name for c in self.client.get_collections().collections]
            if collection_name not in existing:
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=EMBEDDING_DIMENSION,
                        distance=Distance.COSINE,
                    ),
                )

        self._initialized = True

    def is_initialized(self) -> bool:
        return self._initialized

    def populate(self) -> Dict[str, int]:
        """从 knowledge/ 目录加载所有文档并写入 Qdrant（已加载过的跳过，直接用磁盘缓存）"""
        if not self._initialized:
            raise RuntimeError("KnowledgeBase not initialized. Call initialize() first.")

        results = {}
        for category in CATEGORIES:
            collection_name = f"{COLLECTION_PREFIX}{category}"

            # 检查 collection 是否已有数据，有则跳过
            try:
                count = self.client.count(collection_name=collection_name).count
                if count > 0:
                    results[category] = count
                    continue
            except Exception:
                pass

            filepath = os.path.join(self.knowledge_dir, f"{category}.md")
            if not os.path.exists(filepath):
                print(f"[Warning] Knowledge file not found: {filepath}")
                results[category] = 0
                continue

            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

            chunks = self._split_with_headers(content, category)

            collection_name = f"{COLLECTION_PREFIX}{category}"
            points = []
            for idx, chunk in enumerate(chunks):
                vector = self.embeddings.embed_query(chunk["text"])
                point = PointStruct(
                    id=idx,
                    vector=vector,
                    payload={
                        "text": chunk["text"],
                        "category": category,
                        "source": chunk["metadata"].get("source", f"{category}.md"),
                        "header": chunk["metadata"].get("header", ""),
                    },
                )
                points.append(point)

            if points:
                self.client.upsert(collection_name=collection_name, points=points)

            results[category] = len(points)
            print(f"Loaded {len(points)} chunks for category: {category}")

        return results

    def _split_with_headers(self, content: str, category: str) -> List[Dict[str, Any]]:
        """按 markdown 标题分割文档，保留标题作为元数据"""
        chunks = []
        lines = content.split("\n")
        current_header = ""
        current_lines = []

        for line in lines:
            if line.startswith("#"):
                if current_lines:
                    chunk_text = "\n".join(current_lines).strip()
                    if chunk_text:
                        chunks.append({
                            "text": chunk_text,
                            "metadata": {
                                "category": category,
                                "source": f"{category}.md",
                                "header": current_header,
                            },
                        })
                current_header = line.strip("# ").strip()
                current_lines = [line]
            else:
                current_lines.append(line)

        if current_lines:
            chunk_text = "\n".join(current_lines).strip()
            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "metadata": {
                        "category": category,
                        "source": f"{category}.md",
                        "header": current_header,
                    },
                })

        return chunks

    def search(
        self,
        query: str,
        category: str,
        top_k: int = 5,
        score_threshold: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """在指定分类中搜索相关知识"""
        if not self._initialized:
            raise RuntimeError("KnowledgeBase not initialized")

        collection_name = f"{COLLECTION_PREFIX}{category}"
        query_vector = self.embeddings.embed_query(query)

        results = self.client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=top_k,
            score_threshold=score_threshold,
        ).points

        return [
            {
                "text": point.payload["text"],
                "score": point.score,
                "metadata": {
                    "category": point.payload["category"],
                    "source": point.payload["source"],
                    "header": point.payload.get("header", ""),
                },
            }
            for point in results
        ]


_knowledge_base: Optional[KnowledgeBase] = None


def get_knowledge_base(knowledge_dir: str = "knowledge") -> KnowledgeBase:
    """获取全局知识库单例（初始化失败时自动重试）"""
    global _knowledge_base
    if _knowledge_base is None:
        _knowledge_base = KnowledgeBase(knowledge_dir=knowledge_dir)
    if not _knowledge_base.is_initialized():
        _knowledge_base.initialize()
    return _knowledge_base
