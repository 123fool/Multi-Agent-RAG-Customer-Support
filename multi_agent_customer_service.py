"""
多智能体客服系统
使用LangGraph构建，包含多个专门的智能体来处理不同类型的客户查询
基于OpenAI兼容API提供LLM能力
支持多轮对话和会话管理
"""

import sys
sys.dont_write_bytecode = True  # 禁止生成 .pyc，避免 watchfiles 死循环

import os
import json
from typing import Dict, List, Any, Optional, TypedDict, Annotated
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.chat_history import BaseChatMessageHistory
from langgraph.graph import StateGraph
from langgraph.config import get_config
from langchain_core.tools import tool
from pydantic import BaseModel

# 加载环境变量
load_dotenv()

# 导入配置
from config import *

# 导入智能体和工具
from multi_agents import (
    ProductAgent, TechAgent, BillingAgent,
    ComplaintAgent, GeneralAgent
)
from tools import classify_query

# 导入会话管理器
from session_manager import LangChainSessionManager, default_session_manager

# 超出客服范围时的固定回复（护栏：不调用业务智能体）
OUT_OF_SCOPE_REPLY = (
    "抱歉，这里是智能客服，仅处理与产品、技术、账单、投诉及相关售后政策类问题；"
    "请用一句话说明您的具体业务诉求，我很乐意协助。"
)

# 定义状态类型
class AgentState(TypedDict):
    session_id: str
    messages: List[Any]
    current_agent: str
    customer_query: str
    query_type: str
    response: str
    tools_used: List[str]
    next_agent: str
    conversation_history: List[Any]
    memory: Optional[BaseChatMessageHistory]
    # 由图 checkpointer 持久化，跨 LangGraph 工作进程仍可续聊（内存 session_manager 无法做到）
    persisted_dialogue: List[Any]

# 全局会话管理器（使用 LangChain 标准接口）
session_manager = default_session_manager

# 延迟初始化LLM
_llm_instance = None

def initialize_llm_client():
    """初始化 LangChain ChatOpenAI 客户端"""
    if not OPENAI_API_KEY:
        raise ValueError("API密钥未设置")

    return ChatOpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
        model=OPENAI_MODEL,
        temperature=0,
        timeout=HTTP_TIMEOUT,
        max_retries=HTTP_MAX_RETRIES,
    )

def get_llm():
    """获取LLM实例，延迟初始化"""
    global _llm_instance
    if _llm_instance is None:
        try:
            if not OPENAI_API_KEY:
                print("❌ 错误: API密钥未设置，无法初始化LLM")
                _llm_instance = None
            else:
                _llm_instance = initialize_llm_client()
                print(f"✅ 成功初始化 ChatOpenAI 客户端 ({OPENAI_MODEL})")
        except Exception as e:
            print(f"❌ 初始化API客户端失败: {e}")
            print("将使用模拟响应模式")
            _llm_instance = None
    return _llm_instance

# 初始化智能体
def initialize_agents():
    """初始化所有智能体"""
    agents = {
        "product_agent": ProductAgent(),
        "tech_agent": TechAgent(),
        "billing_agent": BillingAgent(),
        "complaint_agent": ComplaintAgent(),
        "general_agent": GeneralAgent()
    }

    # 为每个智能体设置LLM和会话管理器
    for agent in agents.values():
        agent.set_llm(get_llm())  # 延迟获取LLM
        agent.set_session_manager(default_session_manager)

    return agents

# 定义查询分类节点
def classify_query_node(state: AgentState) -> AgentState:
    """Classify customer query"""
    try:
        cfg = get_config()
        tid = (cfg.get("configurable") or {}).get("thread_id")
        if tid:
            state["session_id"] = str(tid)
    except RuntimeError:
        pass

    # 初始化状态对象
    if "session_id" not in state or not state.get("session_id"):
        import uuid
        state["session_id"] = str(uuid.uuid4())

    if "tools_used" not in state:
        state["tools_used"] = []

    if "conversation_history" not in state:
        state["conversation_history"] = []

    if "persisted_dialogue" not in state or state.get("persisted_dialogue") is None:
        state["persisted_dialogue"] = []

    if "memory" not in state:
        state["memory"] = None

    if "next_agent" not in state:
        state["next_agent"] = ""

    if "messages" not in state:
        state["messages"] = []

    # 获取必需字段
    customer_query = state.get("customer_query", "")
    session_id = state["session_id"]

    if not customer_query:
        state["response"] = "Error: No customer query provided"
        state["query_type"] = "general_inquiry"
        return state

    # 使用分类工具
    try:
        llm_instance = get_llm()
        # 使用正确的工具调用方式
        try:
            result = classify_query.invoke({"query": customer_query, "llm": llm_instance})
            query_type = result
        except Exception as e:
            print(f"Error in tool invocation: {e}")
            # 回退到基础分类逻辑
            query_type = "general_inquiry"
    except Exception as e:
        print(f"Error in query classification: {e}")
        query_type = "general_inquiry"

    # 更新状态
    state["query_type"] = query_type
    state["tools_used"].append("query_classification")

    # 写入由 checkpointer 持久化的对话（用户轮次）
    pd = list(state.get("persisted_dialogue") or [])
    pd.append({
        "content": str(customer_query),
        "is_user": True,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })
    state["persisted_dialogue"] = pd

    # 同步到内存 session_manager（仅同进程有效；可选）
    try:
        session_manager.add_message(session_id, str(customer_query), is_user=True)
    except Exception as e:
        print(f"Error adding user message to session: {e}")

    # 护栏：超出范围直接固定回复，不进入业务智能体
    if state["query_type"] == "out_of_scope":
        state["response"] = OUT_OF_SCOPE_REPLY
        state["current_agent"] = "智能客服"
        pd_oos = list(state.get("persisted_dialogue") or [])
        pd_oos.append({
            "content": OUT_OF_SCOPE_REPLY,
            "is_user": False,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })
        state["persisted_dialogue"] = pd_oos
        try:
            session_manager.add_message(session_id, OUT_OF_SCOPE_REPLY, is_user=False)
        except Exception as e:
            print(f"Error adding out_of_scope refusal to session: {e}")
        state["tools_used"].append("out_of_scope_refusal")
        return state

    return state

# 定义智能体处理节点
def create_agent_node(agent_name: str):
    """创建智能体处理节点"""
    def agent_node(state: AgentState) -> AgentState:
        agents = initialize_agents()
        agent = agents.get(agent_name)
        if agent:
            # 获取会话上下文
            session_id = state["session_id"]
            state["conversation_history"] = list(state.get("persisted_dialogue") or [])

            # 处理查询
            result = agent.process(state)

            if not isinstance(result, dict):
                print(f"Agent {agent_name} returned non-dict result: {type(result)}")
                result = {"response": "Error: Agent processing failed", "current_agent": agent_name}

            if "response" not in result:
                print(f"Agent {agent_name} result missing 'response' field: {result}")
                result["response"] = "Error: No response from agent"

            # 助手轮次写入 checkpointer 状态
            pd = list(result.get("persisted_dialogue") or [])
            pd.append({
                "content": str(result["response"]),
                "is_user": False,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            })
            result["persisted_dialogue"] = pd

            # 同步到内存 session_manager（仅同进程有效；可选）
            try:
                session_manager.add_message(session_id, str(result["response"]), is_user=False)
            except Exception as e:
                print(f"Error adding AI message to session: {e}")

            return result
        else:
            state["response"] = f"Error: Agent {agent_name} not found"
            return state
    return agent_node

# 定义最终响应节点
def final_response_node(state: AgentState) -> AgentState:
    """Generate final response"""
    current_agent = state["current_agent"]
    response = state["response"]

    state["response"] = f"【{current_agent}'s Response】\n{response}"
    return state

# 图表入口点
# 使用方式：在langgraph.json文件中增加以下配置，声明构建图的方式，硬编码方式实现。
# "graphs": {
#     "customer_service": "./multi_agent_customer_service.py:make_graph"
# },
# 也可以在langgraph.json文件中使用workflow配置化的方式定义图的结构，但功能相对简单，无法实现复杂的逻辑
def make_graph():
    """构建LangGraph工作流图"""

    # 初始化知识库（RAG）
    # 用独立线程执行，避免 langgraph dev 的 ASGI 阻塞检测拦截 socket.connect / time.sleep
    try:
        import concurrent.futures

        def _init_kb():
            from rag.knowledge_base import get_knowledge_base
            kb = get_knowledge_base()
            if not kb.is_initialized():
                kb.initialize()
            return kb.populate()

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_init_kb)
            results = future.result(timeout=120)
        total = sum(results.values())
        print(f"RAG knowledge base initialized: {total} chunks loaded")
    except Exception as e:
        print(f"[Warning] Failed to initialize knowledge base: {e}")
        print("Agents will use keyword matching fallback")

    # 创建工作流图
    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("classify_query", classify_query_node)
    workflow.add_node("product_agent", create_agent_node("product_agent"))
    workflow.add_node("tech_agent", create_agent_node("tech_agent"))
    workflow.add_node("billing_agent", create_agent_node("billing_agent"))
    workflow.add_node("complaint_agent", create_agent_node("complaint_agent"))
    workflow.add_node("general_agent", create_agent_node("general_agent"))
    workflow.add_node("final_response", final_response_node)

    # 设置入口点
    workflow.set_entry_point("classify_query")

    # 添加条件边（根据查询类型路由到不同智能体）
    workflow.add_conditional_edges(
        "classify_query",
        lambda x: x.get("query_type", ""),  # 添加默认值，避免KeyError
        {
            "product_info": "product_agent",
            "technical_support": "tech_agent",
            "billing": "billing_agent",
            "complaint": "complaint_agent",
            "general_inquiry": "general_agent",
            "out_of_scope": "final_response",
        }
    )

    # 添加直接边（所有智能体都连接到最终响应）
    workflow.add_edge("product_agent", "final_response")
    workflow.add_edge("tech_agent", "final_response")
    workflow.add_edge("billing_agent", "final_response")
    workflow.add_edge("complaint_agent", "final_response")
    workflow.add_edge("general_agent", "final_response")

    # 设置结束点
    workflow.set_finish_point("final_response")

    # 编译工作流
    app = workflow.compile()

    print("✅ LangGraph工作流图构建完成")
    return app

# 创建默认工作流实例
if __name__ == "__main__":
    app = make_graph()
    print("🚀 多智能体客服系统启动成功！")
