"""
基础智能体类
所有专门智能体的基类
"""

from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod
import os
from session_manager import LangChainSessionManager


class BaseAgent(ABC):
    def __init__(self, name: str, role: str, expertise: List[str], rag_category: str = "", session_manager: LangChainSessionManager = None):
        self.name = name
        self.role = role
        self.expertise = expertise
        self.rag_category = rag_category
        self.llm = None  # 将在运行时注入
        self.session_manager = session_manager or LangChainSessionManager()

    def set_llm(self, llm):
        """设置LLM客户端"""
        self.llm = llm

    def set_session_manager(self, session_manager: LangChainSessionManager):
        """设置会话管理器"""
        self.session_manager = session_manager

    @abstractmethod
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """处理客户查询的抽象方法"""
        pass
#@abstractmethod:这是个装饰器,标记"这方法是抽象的 —— 父类只声明不实现,子类必须自己写"。
#pass:占位符,表示"这里啥也不做"(因为抽象方法本来就不该有实现)。

#方法名前下划线 _ = 约定"这是内部方法,只给类自己/子类用,外部别调"。
    def _get_conversation_context(
        self,
        session_id: str,
        state: Optional[Dict[str, Any]] = None,
        max_messages: int = 12,
    ) -> str:
        """优先使用图状态中持久化的对话（跨 LangGraph 进程/工作有效），否则回退到 session_manager。"""
        if state is not None:
            records = state.get("persisted_dialogue")
            if records:
                tail = records[-max_messages:]
                context_lines = []
                for msg in tail:
                    role = "用户" if msg.get("is_user", True) else "AI"
                    content = msg.get("content", "")
                    timestamp = msg.get("timestamp", "")
                    context_lines.append(f"[{timestamp}] {role}: {content}")
                return "\n".join(context_lines)
        try:
            conversation_context = self.session_manager.get_conversation_context(session_id, max_messages)

            if not conversation_context:
                return ""

            # 格式化对话历史
            context_lines = []
            for msg in conversation_context:
                role = "用户" if msg.get("is_user", True) else "AI"
                content = msg.get("content", "")
                timestamp = msg.get("timestamp", "")
                context_lines.append(f"[{timestamp}] {role}: {content}")

            return "\n".join(context_lines)
        except Exception as e:
            print(f"获取对话上下文时出错: {e}")
            return ""

    def _add_message_to_session(self, session_id: str, message: str, is_user: bool = True):
        """添加消息到会话历史"""
        try:
            self.session_manager.add_message(session_id, message, is_user)
        except Exception as e:
            print(f"添加消息到会话时出错: {e}")

    def _enhance_system_prompt_with_context(self, base_prompt: str) -> str:
        """增强系统提示，添加对话上下文说明"""
        context_instruction = """

【知识库使用规范（必须严格遵守）】
1. 当下方提供了"知识库信息 / 服务信息 / 政策信息 / 产品信息"等内容时，你的回答【必须且只能】基于这些信息，严禁编造、推测或补充任何未在其中出现的具体数字、价格、参数、型号、时限、电话号码。
2. 引用具体数据（价格、容量、时长、费率、电话号码等）时，必须与提供的知识库信息完全一致，逐字准确。
3. 如果知识库信息未覆盖客户问题的某个细节，请如实说明"该信息暂未在知识库中"，不要凭经验或常识臆测。

重要：请结合对话历史上下文，理解客户之前的问题和需求，提供连贯、个性化的回答。
如果这是多轮对话，请参考之前的对话内容，避免重复信息，并基于客户的新问题提供补充信息。
保持对话的连贯性和自然性，让客户感受到你理解他们的完整需求。"""

        return base_prompt + context_instruction

    def _build_messages(self, system_prompt: str, customer_query: str,
                        matched_info: str = "", conversation_context: str = "") -> list:
        """
        构建消息列表。将检索到的知识放入 SystemMessage（而非 HumanMessage），
        提高模型对知识的依从性，减少幻觉。

        消息结构：
          [SystemMessage] 对话历史（可选）
          [SystemMessage] 系统提示 + 知识库信息（合并，强约束）
          [HumanMessage]  用户纯问题
        """
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = []

        if conversation_context:
            messages.append(SystemMessage(
                content=f"对话历史上下文：\n{conversation_context}\n\n请基于以上对话历史和当前查询，提供连贯的解答。"
            ))

        if matched_info:
            full_system = (
                f"{system_prompt}\n\n"
                f"=== 知识库检索结果（你的回答【必须且只能】基于以下信息，"
                f"禁止编造任何未提及的数字、参数、价格、型号、时限） ===\n"
                f"{matched_info}\n"
                f"=== 知识库检索结果结束 ==="
            )
            messages.append(SystemMessage(content=full_system))
        else:
            messages.append(SystemMessage(content=system_prompt))

        messages.append(HumanMessage(content=customer_query))
        return messages

    def get_info(self) -> Dict[str, Any]:
        """获取智能体信息"""
        return {
            "name": self.name,
            "role": self.role,
            "expertise": self.expertise
        }

    def retrieve_knowledge(self, query: str) -> str:
        """
        使用 RAG 从知识库检索相关信息
        返回格式化的知识文本，无结果时返回空字符串。
        设置环境变量 DISABLE_RAG=1 可强制关闭 RAG（用于与关键词匹配方案做对比评估）。
        """
        if os.environ.get("DISABLE_RAG", "").lower() in ("1", "true", "yes"):
            print(f"[DEBUG] {self.name} RAG已禁用(DISABLE_RAG=1)")
            return ""
        try:
            from rag.retriever import retrieve
            result = retrieve(query=query, category=self.rag_category)
            print(f"[DEBUG] {self.name} RAG检索: query='{query[:30]}' → {'有结果('+str(len(result))+'字符)' if result else '空!'}")
            return result
        except Exception as e:
            print(f"RAG retrieval error in {self.name}: {e}")
            return ""
