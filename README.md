<h1 align="center">Multi-Agent RAG Customer Support</h1>

<p align="center">多智能体 RAG 客服系统</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-yellow.svg" alt="License Apache 2.0"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10%2B-blue.svg" alt="Python 3.10+"></a>
  <a href="https://flask.palletsprojects.com/"><img src="https://img.shields.io/badge/Flask-%E2%89%A52.3-339933.svg" alt="Flask"></a>
  <a href="https://python.langchain.com/"><img src="https://img.shields.io/badge/langchain--core-%E2%89%A51.3-green.svg" alt="langchain-core"></a>
  <a href="https://github.com/langchain-ai/langgraph"><img src="https://img.shields.io/badge/LangGraph-%E2%89%A51.0-purple.svg" alt="LangGraph"></a>
</p>

<p align="center"><em>模块化多智能体路由 · RAG 知识检索 · Flask Web 前台 · LangGraph 会话编排</em></p>


## 项目概述

这是一个基于LangGraph构建的多智能体客服系统，支持产品咨询、技术支持、账单处理、投诉处理等多种业务场景。系统采用模块化设计，每个智能体独立运行，通过配置文件定义工作流程。集成 RAG（检索增强生成）模块，基于 Qdrant 向量数据库和 BGE-M3 Embedding 实现知识检索，替代传统关键词匹配，显著提升回答准确性。

## 项目信息

- 维护者：范文哲（[123fool](https://github.com/123fool)）
- 仓库：[123fool/Multi-Agent-RAG-Customer-Support](https://github.com/123fool/Multi-Agent-RAG-Customer-Support)
- 许可证：Apache License 2.0，见 [LICENSE](LICENSE)。

公开版本已清理本地开发配置、原演示截图、临时调试文件及已有评测输出；源码保留当前工作区的 Embedding 接入实现。

## 项目结构

```
Multi-Agent-RAG-Customer-Support/
├── multi_agents/ # 智能体模块
│ ├── __init__.py # 智能体包初始化
│ ├── base_agent.py # 基础智能体类（含 RAG 检索方法）
│ ├── product_agent.py # 产品专家智能体
│ ├── tech_agent.py # 技术支持专家智能体
│ ├── billing_agent.py # 账单专家智能体
│ ├── complaint_agent.py # 投诉处理专家智能体
│ └── general_agent.py # 综合客服智能体
├── rag/ # RAG 检索增强生成模块
│ ├── __init__.py # RAG 模块初始化
│ ├── embeddings.py # BGE-M3 Embedding 封装（SiliconFlow API）
│ ├── knowledge_base.py # Qdrant 向量数据库管理（内存模式）
│ ├── retriever.py # 便捷检索函数
│ └── init_knowledge.py # 知识库初始化与测试脚本
├── knowledge/ # 知识文档（Markdown 格式）
│ ├── products.md # 产品知识库
│ ├── tech_support.md # 技术支持知识库
│ ├── billing.md # 账单知识库
│ ├── complaints.md # 投诉处理知识库
│ └── general_service.md # 综合服务知识库
├── tools/ # 工具函数模块
│ ├── __init__.py # 工具包初始化
│ └── query_tools.py # 查询分类工具
├── templates/ # Web界面模板
│ └── index.html # 主页面HTML模板
├── config.py # 基础配置文件
├── multi_agent_customer_service.py # 主程序文件（LangGraph工作流）
├── session_manager.py # 会话管理器（LangChain标准接口）
├── web_app.py # Web 入口（Flask 路由与 HTTP 会话）
├── chat_web_service.py # LangGraph 调用、会话状态解析等业务逻辑
├── langgraph.json # LangGraph工作流配置
├── requirements.txt # 项目依赖
├── .env # 环境变量配置
├── README.md # 项目说明文档
└── README_LangGraph_CLI.md # LangGraph CLI使用说明
```

## 主要特性

### 1. 模块化智能体设计
- 每个智能体独立实现，职责单一
- 基于抽象基类，便于扩展和维护
- 支持动态LLM注入

### 2. 配置驱动的工作流
- 工作流定义在`langgraph.json`中
- 支持条件路由和直接连接
- 无需硬编码流程逻辑

### 3. 专业的业务领域
- **产品专家**: 产品信息咨询和推荐
- **技术支持专家**: 技术问题诊断和解决
- **账单专家**: 财务和账单问题处理
- **投诉处理专家**: 客户投诉和建议处理
- **综合客服**: 一般咨询处理

### 4. RAG 知识检索
- **向量数据库**: 使用 Qdrant 内存模式存储知识向量，按业务领域分 Collection 管理
- **Embedding 模型**: 基于 SiliconFlow API 调用 BAAI/bge-m3（1024 维，中英文通用）
- **文档处理**: Markdown 知识文档按标题分割为语义片段，保留标题元数据
- **检索策略**: 各 Agent 按所属领域在对应 Collection 中检索 Top-K 相关知识，无结果时回退到关键词匹配
- **知识库管理**: 支持 `knowledge/` 目录下 Markdown 文档的批量加载与索引

### 5. 智能对话上下文管理
- **会话管理**: 支持多会话并发，每个会话独立管理
- **历史对话缓存**: 完整的对话历史记录，包含时间戳和角色标识
- **上下文感知**: 智能体基于历史对话生成连贯、个性化的回答
- **记忆功能**: 集成LangChain记忆组件，支持长期对话记忆
- **多轮对话**: 同一会话内支持连续对话，避免重复信息

## 安装和配置

### 1. 安装依赖
```bash
git clone https://github.com/123fool/Multi-Agent-RAG-Customer-Support.git
cd Multi-Agent-RAG-Customer-Support
pip install -r requirements.txt
pip install qdrant-client langgraph-cli
pip install -U "langgraph-cli[inmem]"
```
在win环境中，langgraph-cli下载后，需要将`langgraph.exe`路径加入PATH环境变量。或者使用时直接带全路径，例`<your site-packages>\bin\langgraph.exe`

### 2. 环境变量配置
复制 `env_example.txt` 为 `.env` 文件并填入自己的 API 密钥。配置示例中的密钥字段为空；Flask 会话密钥可通过 `FLASK_SECRET_KEY` 配置，未设置时使用进程内随机密钥。主要环境变量：

| 变量名 | 说明 | 默认值 |
|-------|------|-------|
| `OPENAI_API_KEY` | LLM API 密钥（兼容 OpenAI 接口） | 无 |
| `OPENAI_BASE_URL` | LLM API 地址 | 无 |
| `OPENAI_MODEL` | LLM 模型名称 | `Qwen/Qwen3-8B` |
| `EMBEDDING_MODEL` | Embedding 模型名称 | `BAAI/bge-m3` |
| `EMBEDDING_DIMENSION` | Embedding 向量维度 | `1024` |
| `RAG_TOP_K` | 检索返回的 Top-K 结果数 | `5` |
| `RAG_SCORE_THRESHOLD` | 检索相似度阈值 | `0.3` |
| `KNOWLEDGE_DIR` | 知识文档目录 | `knowledge` |
| `LANGSMITH_API_KEY` | LangSmith 追踪密钥 | 无 |
| `LANGSMITH_TRACING` | 是否启用 LangSmith 追踪 | `true` |
| `LANGSMITH_PROJECT` | LangSmith 项目名 | `Multi-Agent-RAG-Customer-Support` |

### 3. 图结构自检
```bash
python multi_agent_customer_service.py
```

## 🚀 运行说明

### 方式1：使用Studio UI访问langgraph服务
```
langgraph dev
```
启动后，会自动拉起LangSmith服务，包含LangStudio UI，默认2024端口。使用浏览器访问`https://smith.langchain.com/studio/thread?render=interact&baseUrl=http://127.0.0.1:2024`

### 方式2：使用Web服务调用langgraph api
```
## 终端1：启动LangGraph服务
langgraph dev

## 终端2：启动自定义Web服务
python ./web_app.py
```
使用浏览器访问`http://localhost:5000`，界面功能：
- 实时聊天: 输入问题，获得智能回复
- 智能体信息: 显示当前处理问题的专家和查询类型
- 历史管理: 查看、清除对话历史
- 数据导出: 导出对话记录用于分析

### 方式3：直接API调用
接口文档访问地址默认`http://127.0.0.1:2024/docs`，内嵌了js需要挂梯子。也可参考`https://langchain-ai.github.io/langgraph/cloud/reference/api/api_ref.html`

需要`langgraph dev`启动LangGraph服务。

## 工作流程

1. **会话创建**: 为每个客户创建唯一会话ID
2. **查询分类**: 系统自动分析客户查询类型
3. **知识检索**: 基于向量相似度从 Qdrant 知识库中检索相关知识片段
4. **上下文加载**: 加载历史对话上下文
5. **智能体路由**: 根据查询类型路由到相应的专业智能体
6. **专业处理**: 专业智能体结合检索知识和上下文处理客户查询
7. **响应生成**: 生成最终响应并更新会话历史
8. **状态保存**: 保存对话状态和记忆信息

### 工作流程图
```
客户查询 → 会话管理 → 查询分类 → 知识检索 → 上下文加载 → 智能体路由 → 专业处理 → 最终响应
    ↓         ↓         ↓         ↓          ↓          ↓          ↓         ↓
  输入    会话创建   类型识别   向量检索   历史加载    专家选择    专业解答    格式化输出
                ↓         ↓                              ↓
            状态保存   Qdrant+Embedding                 知识增强
```

## RAG 知识检索

### 架构
```
knowledge/*.md → 按标题分割 → BGE-M3 Embedding → Qdrant 内存向量库
                                                        ↓
Agent.process() → retrieve_knowledge(query, category) → Top-K 相关片段 → 增强 LLM Prompt
```

### 知识领域与 Collection 映射
| 业务领域 | 知识文件 | Qdrant Collection |
|---------|---------|------------------|
| 产品咨询 | `knowledge/products.md` | `cs_products` |
| 技术支持 | `knowledge/tech_support.md` | `cs_tech_support` |
| 账单处理 | `knowledge/billing.md` | `cs_billing` |
| 投诉处理 | `knowledge/complaints.md` | `cs_complaints` |
| 综合服务 | `knowledge/general_service.md` | `cs_general_service` |

### 检索流程
1. 知识文档（Markdown）按 `#` 标题自动分割为语义片段
2. 各片段通过 BGE-M3 模型生成 1024 维向量
3. 向量存入 Qdrant 对应领域的 Collection（余弦相似度）
4. Agent 处理用户查询时，先用 `retrieve_knowledge()` 做向量检索
5. 若检索结果低于阈值（默认 0.3），回退到关键词匹配

### 知识库初始化测试
```bash
python -m rag.init_knowledge
```

### 状态管理
系统使用 `AgentState` 来管理整个工作流的状态：
- `customer_query`: 客户查询内容
- `query_type`: 查询类型分类
- `current_agent`: 当前处理智能体
- `response`: 智能体回复
- `tools_used`: 使用的工具列表
- `session_id`: 会话唯一标识
- `conversation_history`: 对话历史记录
- `memory`: LangChain记忆组件

## 关于硅基流动API
系统中llm大模型使用的是硅基流动模型服务商，也可选其他，都是使用统一的openAI接口规范。

### 优势特点
- **国内服务**: 访问速度快，延迟低
- **价格实惠**: 相比其他API服务更经济
- **模型丰富**: 支持多种开源模型
- **API兼容**: 完全兼容OpenAI API格式
- **配置简单**: 只需设置API密钥和模型名称

### 推荐模型
- `qwen2.5-7b-instruct` - 性价比高，适合一般应用
- `qwen2.5-14b-instruct` - 性能更好，适合复杂任务
- `llama3.1-8b-instruct` - 通用性强，稳定性好
- `mistral-7b-instruct` - 推理能力强

## 扩展指南

### 添加新的智能体

1. 在 `multi_agents/` 目录下创建新的智能体文件
2. 继承`BaseAgent`类并实现`process`方法
3. 在`multi_agents/__init__.py`中导入新智能体
4. 在`langgraph.json`中添加节点和边配置

### 添加新的工具函数

1. 在`tools/`目录下创建新的工具文件
2. 使用`@tool`装饰器定义工具
3. 在`tools/__init__.py`中导入新工具

### 修改工作流程

1. 编辑`langgraph.json`文件
2. 修改节点、边和路由配置
3. 重启系统应用新配置

## 技术架构

- **LangGraph**: 工作流编排框架
- **LangChain Core**: LLM集成和消息处理
- **Qdrant**: 向量数据库（内存模式），知识存储与检索
- **BGE-M3 (BAAI)**: 中英文通用 Embedding 模型，通过 SiliconFlow API 调用
- **硅基流动API**: 大语言模型服务（OpenAI 兼容接口）
- **LangSmith**: 全链路追踪与调试（LLM 调用、Agent 路由、RAG 检索）
- **模块化设计**: 高内聚、低耦合的架构

## 相关文档
- [README_LangGraph_CLI.md](README_LangGraph_CLI.md) - LangGraph CLI使用指南
- [langgraph.json](langgraph.json) - 工作流配置文件
- [LangGraph API服务搭建](https://docs.langchain.com/langgraph-platform/cli#configuration-file)
- [LangGraph MCP适配器](https://github.com/langchain-ai/langchain-mcp-adapters)


## 示例数据与隐私

`knowledge/` 中的内容用于演示客服知识检索；联系方式已改为示例邮箱和占位名称，评测用例已同步更新。请勿将这些演示资料视为维护者的个人信息或真实业务承诺。

`.env`、开发工具本地配置、LangGraph 状态、Qdrant 数据、历史会话及生成的评测报告不提交到公开仓库。运行评测后可在本地生成新的结果。

## 回归测试

```bash
python -m unittest discover -s tests -v
```

测试覆盖 RAG 无结果时技术支持、账单与综合客服的关键词回退路径；不需要调用外部模型或向量数据库。

## 技术来源

- 项目参考：[zhouyan6/multi-agent-customer-service](https://github.com/zhouyan6/multi-agent-customer-service)。
- LangGraph、LangChain 与 Qdrant 等依赖的相关文档见上文。
