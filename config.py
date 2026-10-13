"""
配置文件
包含系统运行所需的各种配置参数
"""

import os
from dotenv import load_dotenv
#load_dotenv()可以把.env文件中的环境变量加载到os.environ中

# 加载环境变量
load_dotenv()

# OpenAI兼容API配置
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.siliconflow.cn/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "Qwen/Qwen3-8B")

# Embedding 模型配置
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "1024"))
#环境变量读出来永远是字符串 "1024"，需要转换为整数

# RAG 配置
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))
RAG_SCORE_THRESHOLD = float(os.getenv("RAG_SCORE_THRESHOLD", "0.3"))
KNOWLEDGE_DIR = os.getenv("KNOWLEDGE_DIR", "knowledge")

# Qdrant 向量数据库配置（为空时回退到本地文件模式）
QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")

# HTTP请求配置
HTTP_TIMEOUT = int(os.getenv("HTTP_TIMEOUT", "30"))
HTTP_MAX_RETRIES = int(os.getenv("HTTP_MAX_RETRIES", "3"))
HTTP_HEADERS = {
    "Content-Type": "application/json",
    "User-Agent": "MultiAgentCustomerService/1.0.0"
}

# 系统配置
SYSTEM_NAME = "多智能体RAG客服系统"
VERSION = "1.0.0"

# 日志配置
LOG_CONFIG = {
    "level": os.getenv("LOG_LEVEL", "INFO"),
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
}
