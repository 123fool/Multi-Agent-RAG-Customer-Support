# 端到端评估模块

在自建客服问答样本集（覆盖产品、技术、账单、投诉、通用 5 个领域共 200 条 + 30 条越界用例）上对多智能体客服系统进行端到端评估。

运行评估脚本后生成 `results.jsonl` 评估明细与 `report.md` 汇总报告。

## 文件说明

| 文件 | 说明 |
|------|------|
| `__init__.py` | 模块初始化 |
| `generate_cases.py` | 测试用例生成脚本（基于 `knowledge/*.md` 真实内容） |
| `test_cases.jsonl` | 生成的 230 条测试用例（5×40 + 30 越界） |
| `run_eval.py` | 评估主脚本 |
| `results.jsonl` | 评估明细（每条用例的实际输出与评估结果） |
| `report.md` | 评估报告（总体指标 + 分领域指标） |

## 评估维度

| 维度 | 评估方式 | 指标 |
|------|---------|------|
| **分类准确率** | 预测 `query_type` 是否等于标注的 `expected_query_type` | Accuracy |
| **RAG 召回率** | 检索片段是否包含 `expected_keywords`（直接调 `rag.retrieve`） | Recall@5 |
| **回答质量** | 最终回答是否包含 `expected_answer_keywords`；可选 LLM-as-Judge 打分 | Precision / 1-5 分 |
| **越界拒绝率** | 越界用例是否被护栏拒绝（`query_type=out_of_scope` 或命中固定回复） | Rejection Rate |

## 测试用例字段

```json
{
  "id": "prod_001",
  "domain": "products",
  "query": "X1 Pro卖多少钱？",
  "expected_query_type": "product_info",
  "expected_keywords": ["X1 Pro", "4999"],
  "expected_answer_keywords": ["4999"],
  "is_out_of_scope": false
}
```

- `expected_keywords`：期望出现在 RAG 检索片段中（验证召回率）
- `expected_answer_keywords`：期望出现在最终回答中（验证回答质量）

## 前置条件

1. **启动 LangGraph 服务**（评估脚本通过 HTTP API 调用系统）
   ```bash
   langgraph dev
   ```
2. **知识库就绪**：`knowledge/*.md` 已存在（RAG 召回评估会在本地进程内初始化 Qdrant）

## 运行评估

```bash
# 完整评估（230 条）
python -m evaluation.run_eval

# 快速验证（前 20 条）
python -m evaluation.run_eval --limit 20

# 只评估某个领域
python -m evaluation.run_eval --domain products

# 启用 LLM-as-Judge（更准确但更慢、消耗 token）
python -m evaluation.run_eval --judge

# 跳过 RAG 召回（只测分类 + 回答 + 越界）
python -m evaluation.run_eval --skip-rag
```

## 重新生成测试用例

若 `knowledge/*.md` 内容有变化，重新生成对齐的测试用例：

```bash
python -m evaluation.generate_cases
```

## 输出示例

`report.md`：
```
| 维度 | 通过 / 总数 | 指标 |
|------|-----------|------|
| 分类准确率 | 210/230 (91.3%) | Accuracy |
| RAG 召回率 | 172/200 (86.0%) | Recall@5 |
| 回答关键词命中 | 182/200 (91.0%) | Precision(KW) |
| 越界拒绝率 | 28/30 (93.3%) | Rejection Rate |
```

## 注意事项

- 每条用例使用独立线程，避免对话历史污染评估结果
- RAG 召回评估在评估脚本进程内独立初始化 Qdrant（与 langgraph dev 进程隔离），加载相同的 `knowledge/*.md`，保证检索逻辑一致
- LLM-as-Judge 使用与系统相同的 LLM，仅作为辅助参考；关键指标以关键词命中为主
