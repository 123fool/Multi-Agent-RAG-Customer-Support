#!/usr/bin/env python3
"""
客服系统端到端评估脚本

评估四个维度：
  1. 分类准确率  — query_type 是否路由到正确的 Agent
  2. RAG 召回率  — 知识库是否检索到包含期望关键词的片段
  3. 回答质量    — 最终回答是否包含期望的关键信息（关键词命中 / LLM-as-Judge）
  4. 越界拒绝率  — 非客服类问题是否被正确拒绝

前置条件：
  · langgraph dev 已启动（默认 http://127.0.0.1:2024）
  · knowledge/*.md 已就绪（RAG 召回评估在本地进程内初始化 Qdrant）

用法：
  python -m evaluation.run_eval                       # 跑全部 230 条
  python -m evaluation.run_eval --limit 20            # 只跑前 20 条（快速验证）
  python -m evaluation.run_eval --domain products     # 只跑某个领域
  python -m evaluation.run_eval --judge               # 启用 LLM-as-Judge 打分
  python -m evaluation.run_eval --skip-rag            # 跳过 RAG 召回（省时间）
"""

import argparse
import json
import os
import sys
import time
import uuid
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
API_URL = os.getenv("LANGGRAPH_API_URL", "http://127.0.0.1:2024").rstrip("/")
GRAPH_NAME = os.getenv("LANGGRAPH_GRAPH_NAME", "customer_service")
HTTP_TIMEOUT = 120
POLL_INTERVAL = 0.5
MAX_WAIT = 180

BASE_DIR = os.path.dirname(__file__)
CASES_FILE = os.path.join(BASE_DIR, "test_cases.jsonl")
RESULTS_FILE = os.path.join(BASE_DIR, "results.jsonl")
REPORT_FILE = os.path.join(BASE_DIR, "report.md")

# 越界固定回复关键词（来自 multi_agent_customer_service.OUT_OF_SCOPE_REPLY）
OOS_MARKERS = ["仅处理", "智能客服", "具体业务诉求"]


# ---------------------------------------------------------------------------
# LangGraph API 调用
# ---------------------------------------------------------------------------

def ensure_assistant() -> str:
    """查找或创建助手，返回 assistant_id。"""
    resp = requests.post(
        f"{API_URL}/assistants/search",
        json={"graph_id": GRAPH_NAME, "limit": 1},
        timeout=10,
    )
    if resp.status_code == 200:
        items = resp.json()
        if items:
            return items[0]["assistant_id"]

    resp = requests.post(
        f"{API_URL}/assistants",
        json={"graph_id": GRAPH_NAME, "name": "Eval Assistant"},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()["assistant_id"]


def run_query(assistant_id: str, query: str) -> Dict[str, Any]:
    """
    对单条 query 走完整 LangGraph 流程，返回状态字段。
    每条用例使用独立线程，避免历史污染。
    """
    # 1. 新建线程
    resp = requests.post(f"{API_URL}/threads", json={}, timeout=10)
    resp.raise_for_status()
    thread_id = resp.json()["thread_id"]

    # 2. 提交运行（同步阻塞，input 内带 customer_query）
    resp = requests.post(
        f"{API_URL}/threads/{thread_id}/runs",
        json={
            "assistant_id": assistant_id,
            "input": {
                "messages": [{"role": "user", "content": query}],
                "customer_query": query,
                "session_id": thread_id,
            },
        },
        timeout=HTTP_TIMEOUT,
    )
    resp.raise_for_status()
    run_id = resp.json()["run_id"]

    # 3. 轮询直到完成
    start = time.time()
    while True:
        if time.time() - start > MAX_WAIT:
            return {"error": "timeout", "response": "", "query_type": "", "current_agent": ""}

        time.sleep(POLL_INTERVAL)
        s = requests.get(
            f"{API_URL}/threads/{thread_id}/runs/{run_id}",
            timeout=10,
        )
        if s.status_code != 200:
            continue
        status = s.json().get("status", "unknown")

        if status in ("completed", "success"):
            break
        if status in ("failed", "cancelled", "error"):
            return {"error": f"run_{status}", "response": "", "query_type": "", "current_agent": ""}

    # 4. 取线程状态
    s = requests.get(f"{API_URL}/threads/{thread_id}/state", timeout=10)
    s.raise_for_status()
    values = s.json().get("values", {})

    return {
        "error": None,
        "response": str(values.get("response", "") or ""),
        "query_type": str(values.get("query_type", "") or ""),
        "current_agent": str(values.get("current_agent", "") or ""),
        "thread_id": thread_id,
    }


# ---------------------------------------------------------------------------
# RAG 召回（本地进程内直接调用 retriever）
# ---------------------------------------------------------------------------

_rag_ready = False


def ensure_rag() -> bool:
    global _rag_ready
    if _rag_ready:
        return True
    try:
        from rag.knowledge_base import get_knowledge_base
        kb = get_knowledge_base()
        if not kb.is_initialized():
            kb.initialize()
        kb.populate()
        _rag_ready = True
        return True
    except Exception as e:
        print(f"[Warn] RAG 初始化失败: {e}")
        return False


def rag_retrieve(query: str, category: str, top_k: int = 5) -> str:
    """直接调用 retriever，返回拼接后的检索文本。"""
    from rag.retriever import retrieve
    return retrieve(query=query, category=category, top_k=top_k)


# ---------------------------------------------------------------------------
# LLM-as-Judge（可选）
# ---------------------------------------------------------------------------

_judge_llm = None


def get_judge():
    global _judge_llm
    if _judge_llm is None:
        from langchain_openai import ChatOpenAI
        from config import OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL
        _judge_llm = ChatOpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
            model=OPENAI_MODEL,
            timeout=60,
            max_retries=1,
        )
    return _judge_llm


def judge_answer(query: str, response: str, expected_keywords: List[str]) -> Tuple[int, str]:
    """LLM 对回答打 1-5 分，返回 (分数, 理由)。"""
    from langchain_core.messages import HumanMessage, SystemMessage
    judge_prompt = f"""你是客服回答质量评估专家。请对以下客服回答打分（1-5 分）：
5分：完全准确、专业、信息完整
4分：基本正确，信息较完整
3分：部分正确，有遗漏
2分：回答不准确或偏离问题
1分：完全错误或未回答

客户问题：{query}
客服回答：{response}
参考要点（回答应包含）：{', '.join(expected_keywords)}

只输出两行：第一行是分数（1-5），第二行是简短理由（不超过30字）。"""
    try:
        resp = get_judge().invoke([
            SystemMessage(content="你是严格的评估专家。"),
            HumanMessage(content=judge_prompt),
        ])
        text = (getattr(resp, "content", "") or "").strip()
        lines = text.split("\n")
        score = int(lines[0].strip().split()[0].replace("分", ""))
        reason = lines[1].strip() if len(lines) > 1 else ""
        return max(1, min(5, score)), reason
    except Exception as e:
        return 0, f"judge_error: {e}"


# ---------------------------------------------------------------------------
# 评估逻辑
# ---------------------------------------------------------------------------

def eval_classification(result: Dict, expected_qtype: str) -> bool:
    return result.get("query_type", "") == expected_qtype


def eval_rag_recall(query: str, domain: str, expected_keywords: List[str]) -> Dict:
    """返回 {hit, retrieved_text, hit_keywords, missed_keywords}。"""
    if not expected_keywords:
        return {"hit": True, "retrieved_text": "", "hit_keywords": [], "missed": []}
    try:
        text = rag_retrieve(query, domain)
    except Exception as e:
        return {"hit": False, "retrieved_text": f"[error]{e}", "hit_keywords": [], "missed": expected_keywords}
    text_lower = text.lower()
    hit = [k for k in expected_keywords if k.lower() in text_lower]
    missed = [k for k in expected_keywords if k.lower() not in text_lower]
    return {
        "hit": len(missed) == 0,
        "retrieved_text": text[:300],
        "hit_keywords": hit,
        "missed": missed,
    }


def eval_answer_quality(result: Dict, expected_answer_keywords: List[str]) -> Dict:
    """关键词命中率。"""
    response = result.get("response", "")
    if not expected_answer_keywords:
        return {"hit_all": True, "hit_ratio": 1.0, "hit_keywords": [], "missed": []}
    resp_lower = response.lower()
    hit = [k for k in expected_answer_keywords if k.lower() in resp_lower]
    missed = [k for k in expected_answer_keywords if k.lower() not in resp_lower]
    return {
        "hit_all": len(missed) == 0,
        "hit_ratio": len(hit) / len(expected_answer_keywords),
        "hit_keywords": hit,
        "missed": missed,
    }


def eval_oos_rejection(result: Dict) -> bool:
    """越界用例是否被正确拒绝。"""
    if result.get("query_type") == "out_of_scope":
        return True
    response = result.get("response", "")
    return any(m in response for m in OOS_MARKERS)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def load_cases(domain_filter: Optional[str] = None, limit: Optional[int] = None) -> List[Dict]:
    cases = []
    with open(CASES_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            if domain_filter and c["domain"] != domain_filter:
                continue
            cases.append(c)
    if limit:
        cases = cases[:limit]
    return cases


def run_evaluation(args):
    cases = load_cases(args.domain, args.limit)
    print(f"[Eval] 加载 {len(cases)} 条用例")
    if not cases:
        print("没有符合条件的用例，退出。")
        return

    # 探测服务
    try:
        assistant_id = ensure_assistant()
        print(f"[Eval] 助手就绪: {assistant_id}")
    except Exception as e:
        print(f"[Fatal] 无法连接 LangGraph 服务 ({API_URL})：{e}")
        print("请先启动 langgraph dev")
        return

    # 初始化 RAG（如未跳过）
    if not args.skip_rag:
        if ensure_rag():
            print("[Eval] RAG 知识库就绪")
        else:
            print("[Eval] RAG 不可用，召回率维度将跳过")

    # 逐条评估
    results = []
    metrics = defaultdict(lambda: {"total": 0, "pass": 0})
    domain_metrics = defaultdict(lambda: defaultdict(int))

    for idx, case in enumerate(cases, 1):
        cid = case["id"]
        query = case["query"]
        print(f"[{idx}/{len(cases)}] {cid}: {query[:30]}...", end=" ")

        # 调 API
        result = run_query(assistant_id, query)
        if result.get("error"):
            print(f"FAIL ({result['error']})")
            results.append({**case, "result": result, "eval": {"error": result["error"]}})
            continue

        eval_res: Dict[str, Any] = {}

        # 1. 分类准确率
        if not case["is_out_of_scope"]:
            cls_ok = eval_classification(result, case["expected_query_type"])
            eval_res["classification"] = cls_ok
            metrics["classification"]["total"] += 1
            if cls_ok:
                metrics["classification"]["pass"] += 1
            domain_metrics[case["domain"]]["cls_total"] += 1
            domain_metrics[case["domain"]]["cls_pass"] += int(cls_ok)

        # 2. RAG 召回率
        if not args.skip_rag and case.get("expected_keywords") and not case["is_out_of_scope"]:
            rag_res = eval_rag_recall(query, case["domain"], case["expected_keywords"])
            eval_res["rag_recall"] = rag_res
            metrics["rag_recall"]["total"] += 1
            if rag_res["hit"]:
                metrics["rag_recall"]["pass"] += 1

        # 3. 回答质量
        if case.get("expected_answer_keywords") and not case["is_out_of_scope"]:
            aq = eval_answer_quality(result, case["expected_answer_keywords"])
            eval_res["answer_quality"] = aq
            metrics["answer_keyword"]["total"] += 1
            if aq["hit_all"]:
                metrics["answer_keyword"]["pass"] += 1

        if args.judge and not case["is_out_of_scope"]:
            score, reason = judge_answer(query, result["response"], case["expected_answer_keywords"])
            eval_res["judge"] = {"score": score, "reason": reason}
            metrics["judge"]["total"] += 1
            metrics["judge"]["score_sum"] += score

        # 4. 越界拒绝率
        if case["is_out_of_scope"]:
            oos_ok = eval_oos_rejection(result)
            eval_res["oos_rejection"] = oos_ok
            metrics["oos_rejection"]["total"] += 1
            if oos_ok:
                metrics["oos_rejection"]["pass"] += 1

        tag = "PASS" if all(
            v if isinstance(v, bool) else v.get("hit", True)
            for v in eval_res.values()
            if isinstance(v, (bool, dict))
        ) else "CHECK"
        print(tag)

        results.append({"case": case, "result": result, "eval": eval_res})

    # 写明细
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n[Eval] 明细已写入 {RESULTS_FILE}")

    # 生成报告
    write_report(metrics, domain_metrics, len(cases), args)


def write_report(metrics, domain_metrics, total, args):
    lines = []
    lines.append("# 客服系统端到端评估报告\n")
    lines.append(f"- 评估时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- 用例总数：{total}")
    lines.append(f"- LLM-as-Judge：{'启用' if args.judge else '未启用'}\n")

    lines.append("## 总体指标\n")
    lines.append("| 维度 | 通过 / 总数 | 指标 |")
    lines.append("|------|-----------|------|")

    def pct(m):
        return f"{m['pass']}/{m['total']} ({m['pass']/m['total']*100:.1f}%)" if m["total"] else "N/A"

    if metrics["classification"]["total"]:
        lines.append(f"| 分类准确率 | {pct(metrics['classification'])} | Accuracy |")
    if metrics["rag_recall"]["total"]:
        lines.append(f"| RAG 召回率 | {pct(metrics['rag_recall'])} | Recall@5 |")
    if metrics["answer_keyword"]["total"]:
        lines.append(f"| 回答关键词命中（全命中） | {pct(metrics['answer_keyword'])} | Precision(KW) |")
    if metrics["judge"]["total"]:
        avg = metrics["judge"]["score_sum"] / metrics["judge"]["total"]
        lines.append(f"| LLM-as-Judge 平均分 | {avg:.2f} / 5.00 | - |")
    if metrics["oos_rejection"]["total"]:
        lines.append(f"| 越界拒绝率 | {pct(metrics['oos_rejection'])} | Rejection Rate |")

    # 分领域分类准确率
    if domain_metrics:
        lines.append("\n## 分领域分类准确率\n")
        lines.append("| 领域 | 通过 / 总数 | 准确率 |")
        lines.append("|------|-----------|------|")
        for domain, dm in sorted(domain_metrics.items()):
            if dm["cls_total"]:
                r = dm["cls_pass"] / dm["cls_total"] * 100
                lines.append(f"| {domain} | {dm['cls_pass']}/{dm['cls_total']} | {r:.1f}% |")

    report = "\n".join(lines) + "\n"
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[Eval] 报告已写入 {REPORT_FILE}\n")
    print(report)


def main():
    parser = argparse.ArgumentParser(description="客服系统端到端评估")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 条")
    parser.add_argument("--domain", default=None,
                        help="只跑某领域 (products/tech_support/billing/complaints/general_service)")
    parser.add_argument("--judge", action="store_true", help="启用 LLM-as-Judge")
    parser.add_argument("--skip-rag", action="store_true", help="跳过 RAG 召回评估")
    args = parser.parse_args()
    run_evaluation(args)


if __name__ == "__main__":
    main()
