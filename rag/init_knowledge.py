"""
知识库初始化脚本
加载 knowledge/ 目录下的文档到 Qdrant 向量数据库
用法: python -m rag.init_knowledge
"""

import os
import sys


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from dotenv import load_dotenv
    load_dotenv(os.path.join(project_root, ".env"))

    from rag.knowledge_base import KnowledgeBase

    knowledge_dir = os.path.join(project_root, "knowledge")
    print(f"Knowledge directory: {knowledge_dir}")

    kb = KnowledgeBase(knowledge_dir=knowledge_dir)
    print("Initializing Qdrant (in-memory mode)...")
    kb.initialize()

    print("Populating knowledge base...")
    results = kb.populate()

    print("\n=== Knowledge Base Population Results ===")
    total = 0
    for category, count in results.items():
        print(f"  {category}: {count} chunks")
        total += count
    print(f"  TOTAL: {total} chunks")

    print("\n=== Search Test ===")
    test_queries = [
        ("手机有哪些型号推荐", "products"),
        ("电脑无法开机怎么办", "tech_support"),
        ("怎么申请退款", "billing"),
        ("我要投诉配送太慢", "complaints"),
        ("你们的营业时间是什么", "general_service"),
    ]
    for query, category in test_queries:
        results = kb.search(query, category, top_k=2)
        print(f"\nQuery: '{query}' (category: {category})")
        if results:
            for r in results:
                print(f"  Score={r['score']:.3f}: {r['text'][:80]}...")
        else:
            print("  No results found")


if __name__ == "__main__":
    main()
