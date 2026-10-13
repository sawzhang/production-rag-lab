# 平台与模型选型

对应源文档页码：24-27。本笔记为独立学习改编。

## 要理解的机制

先确定状态系统和维护能力，再选产品。Elasticsearch/OpenSearch/Vespa、Milvus/Qdrant/Weaviate、pgvector/Azure AI Search 分别核对原生、插件、应用层和外部推理能力。PostgreSQL ts_rank 不是 BM25。

## 可运行实验

```bash
uv sync --frozen --extra dev --extra neural
uv run --no-sync rag-lab neural-evaluate
RAG_LAB_REAL_MODELS=1 uv run --no-sync pytest tests/test_real_models.py -v
```

## 验证与观察

检查神经适配器要求完整模型 revision，拒绝自动跟踪 main 和 trust_remote_code。适配器合同测试与真实权重测试分别执行，结果见 reports/neural-experiment.md。

## 进阶练习与边界

用 docs/platform-selection.md 的采购记录模板比较两三个实际部署；记录版本、许可、区域、DLS/RLS 绕过身份、过滤执行计划和 TCO。

## 自测

1. 哪种失败不能通过更强的检索模型修复？
2. 本实验锁定了哪些条件，还缺少哪些真实业务证据？
3. 如果撤权、超时或来源冲突发生在这一阶段，应怎样输出？
