#!/usr/bin/env python3
# Updated for 123fool: make report timestamps compatible with Python 3.11.
"""
RAG vs 关键词匹配 检索效果对比脚本

在同一批测试用例上，分别用两种方式检索知识，对比：
  1. 知识召回率（Recall）  — 检索结果是否包含期望关键词
  2. 关键词命中率         — 检索结果中期望关键词的命中比例

两种检索方式：
  · RAG:     rag.retrieve(query, category)         — Qdrant 向量检索
  · Keyword: 各 Agent 的 _match_xxx_info(query)    — 原有关键词匹配逻辑

无需启动 LangGraph 服务，全部在本地进程内运行。
"""

import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from typing import Dict, List

# Windows 控制台 GBK 兼容
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = os.path.dirname(__file__)
CASES_FILE = os.path.join(BASE_DIR, "test_cases.jsonl")
REPORT_FILE = os.path.join(BASE_DIR, "retrieval_comparison.md")


# 领域 → Agent 类名 + 关键词匹配方法名
DOMAIN_AGENT_MAP = {
    "products":        ("ProductAgent",   "_match_products"),
    "tech_support":    ("TechAgent",      "_match_tech_info"),
    "billing":         ("BillingAgent",   "_match_billing_info"),
    "complaints":      ("ComplaintAgent", "_match_complaint_info"),
    "general_service": ("GeneralAgent",   "_match_service_info"),
}


def load_cases() -> List[Dict]:
    cases = []
    with open(CASES_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            if not c.get("is_out_of_scope") and c.get("expected_keywords"):
                cases.append(c)
    return cases


def init_rag():
    """初始化 RAG 知识库。"""
    from rag.knowledge_base import get_knowledge_base
    kb = get_knowledge_base()
    if not kb.is_initialized():
        kb.initialize()
    kb.populate()
    print("[OK] RAG 知识库就绪")


def build_keyword_matchers():
    """实例化 5 个 Agent，返回 {domain: agent_instance}。"""
    from multi_agents import (
        ProductAgent, TechAgent, BillingAgent,
        ComplaintAgent, GeneralAgent,
    )
    cls_map = {
        "ProductAgent": ProductAgent,
        "TechAgent": TechAgent,
        "BillingAgent": BillingAgent,
        "ComplaintAgent": ComplaintAgent,
        "GeneralAgent": GeneralAgent,
    }
    matchers = {}
    for domain, (cls_name, method_name) in DOMAIN_AGENT_MAP.items():
        agent = cls_map[cls_name]()
        matchers[domain] = getattr(agent, method_name)
    print("[OK] 关键词匹配器就绪")
    return matchers


def check_keywords(text: str, keywords: List[str]) -> Dict:
    """检查 text 中包含哪些关键词。"""
    if not text:
        return {"hit_all": False, "hit_ratio": 0.0, "hit": [], "missed": list(keywords)}
    text_lower = text.lower()
    hit = [k for k in keywords if k.lower() in text_lower]
    missed = [k for k in keywords if k.lower() not in text_lower]
    return {
        "hit_all": len(missed) == 0,
        "hit_ratio": len(hit) / len(keywords) if keywords else 1.0,
        "hit": hit,
        "missed": missed,
    }


def main():
    cases = load_cases()
    print(f"[Compare] 加载 {len(cases)} 条业务用例（含 expected_keywords）\n")

    init_rag()
    matchers = build_keyword_matchers()

    from rag.retriever import retrieve as rag_retrieve

    # 统计
    stats = {
        "rag":      {"recall": 0, "hit_ratio_sum": 0.0, "total": 0},
        "keyword":  {"recall": 0, "hit_ratio_sum": 0.0, "total": 0},
    }
    domain_stats = defaultdict(lambda: {
        "rag": {"recall": 0, "total": 0},
        "keyword": {"recall": 0, "total": 0},
    })

    details = []

    for idx, case in enumerate(cases, 1):
        query = case["query"]
        domain = case["domain"]
        keywords = case["expected_keywords"]

        # RAG 检索
        try:
            rag_text = rag_retrieve(query=query, category=domain)
        except Exception as e:
            rag_text = f"[error]{e}"
        rag_res = check_keywords(rag_text, keywords)

        # 关键词匹配
        try:
            kw_text = matchers[domain](query)
        except Exception as e:
            kw_text = f"[error]{e}"
        kw_res = check_keywords(kw_text, keywords)

        # 累计
        stats["rag"]["total"] += 1
        stats["rag"]["recall"] += int(rag_res["hit_all"])
        stats["rag"]["hit_ratio_sum"] += rag_res["hit_ratio"]

        stats["keyword"]["total"] += 1
        stats["keyword"]["recall"] += int(kw_res["hit_all"])
        stats["keyword"]["hit_ratio_sum"] += kw_res["hit_ratio"]

        domain_stats[domain]["rag"]["total"] += 1
        domain_stats[domain]["rag"]["recall"] += int(rag_res["hit_all"])
        domain_stats[domain]["keyword"]["total"] += 1
        domain_stats[domain]["keyword"]["recall"] += int(kw_res["hit_all"])

        tag_rag = "HIT" if rag_res["hit_all"] else "MISS"
        tag_kw = "HIT" if kw_res["hit_all"] else "MISS"
        print(f"[{idx}/{len(cases)}] {case['id']}: RAG={tag_rag}  KW={tag_kw}")

        details.append({
            "id": case["id"],
            "domain": domain,
            "query": query,
            "keywords": keywords,
            "rag": {"hit_all": rag_res["hit_all"], "hit": rag_res["hit"], "missed": rag_res["missed"]},
            "keyword": {"hit_all": kw_res["hit_all"], "hit": kw_res["hit"], "missed": kw_res["missed"]},
        })

    # 计算汇总
    n = stats["rag"]["total"]
    rag_recall = stats["rag"]["recall"] / n * 100 if n else 0
    kw_recall = stats["keyword"]["recall"] / n * 100 if n else 0
    rag_avg_ratio = stats["rag"]["hit_ratio_sum"] / n * 100 if n else 0
    kw_avg_ratio = stats["keyword"]["hit_ratio_sum"] / n * 100 if n else 0

    # 写报告
    lines = [
        "# RAG vs 关键词匹配 检索效果对比报告\n",
        f"- 用例数：{n}",
        f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
        "## 总体对比\n",
        "| 指标 | 关键词匹配 | RAG 向量检索 | 提升幅度 |",
        "|------|-----------|-------------|---------|",
    ]

    def delta(new, old):
        d = new - old
        sign = "+" if d >= 0 else ""
        return f"{sign}{d:.1f}pp"

    lines.append(f"| 知识召回率（全命中） | {kw_recall:.1f}% ({stats['keyword']['recall']}/{n}) | {rag_recall:.1f}% ({stats['rag']['recall']}/{n}) | {delta(rag_recall, kw_recall)} |")
    lines.append(f"| 关键词平均命中率 | {kw_avg_ratio:.1f}% | {rag_avg_ratio:.1f}% | {delta(rag_avg_ratio, kw_avg_ratio)} |")

    # 分领域
    lines.append("\n## 分领域召回率对比\n")
    lines.append("| 领域 | 关键词匹配 | RAG 向量检索 | 提升 |")
    lines.append("|------|-----------|-------------|------|")
    for domain in DOMAIN_AGENT_MAP:
        ds = domain_stats[domain]
        t = ds["rag"]["total"]
        if t == 0:
            continue
        kr = ds["keyword"]["recall"] / t * 100
        rr = ds["rag"]["recall"] / t * 100
        lines.append(f"| {domain} | {ds['keyword']['recall']}/{t} ({kr:.1f}%) | {ds['rag']['recall']}/{t} ({rr:.1f}%) | {delta(rr, kr)} |")

    report = "\n".join(lines) + "\n"
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(report)

    # 明细
    details_file = os.path.join(BASE_DIR, "retrieval_comparison.jsonl")
    with open(details_file, "w", encoding="utf-8") as f:
        for d in details:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print(f"\n[Compare] 报告已写入 {REPORT_FILE}")
    print(f"[Compare] 明细已写入 {details_file}\n")
    print(report)


if __name__ == "__main__":
    main()
