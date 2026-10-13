#!/usr/bin/env python3
"""
幻觉诊断脚本：追踪一条 query 从检索到生成的全过程
"""
import sys, os
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

os.environ.pop("DISABLE_RAG", None)  # 确保 RAG 开启

from dotenv import load_dotenv
load_dotenv()

from multi_agents import ProductAgent
from multi_agent_customer_service import get_llm

# 1.5 初始化知识库（必须 populate 才有数据）
from rag.knowledge_base import get_knowledge_base
kb = get_knowledge_base()
if not kb.is_initialized():
    kb.initialize()
kb.populate()
print("[OK] 知识库已加载\n")

# 1. 初始化
llm = get_llm()
agent = ProductAgent()
agent.set_llm(llm)

query = "X1 Pro卖多少钱？"

# 2. 看检索到了什么
print("=" * 60)
print("[1] RAG 检索结果：")
print("=" * 60)
knowledge = agent.retrieve_knowledge(query)
print(knowledge[:500] if knowledge else "(空 — RAG 没检索到)")
print()

# 3. 看发给 LLM 的完整消息
print("=" * 60)
print("[2] 发给 LLM 的完整消息：")
print("=" * 60)

system_prompt = agent._enhance_system_prompt_with_context(
    f"你是{agent.name}，专门负责{agent.role}。"
)
messages = agent._build_messages(system_prompt, query, knowledge)
for i, msg in enumerate(messages):
    print(f"\n--- Message {i} [{type(msg).__name__}] ---")
    print(msg.content[:800])
print()

# 4. 看 LLM 回答
print("=" * 60)
print("[3] LLM 回答：")
print("=" * 60)
response = llm.invoke(messages)
print(response.content)
print()

# 5. 判断
print("=" * 60)
has_4999 = "4999" in response.content
print(f"回答中包含 '4999': {'是 ✓' if has_4999 else '否 ✗（幻觉）'}")
