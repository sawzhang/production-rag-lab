# 验证记录 · 2026-10-13

## 本次实际执行

- macOS ARM64，Python 3.12.13，uv 锁定依赖；原始 PyPI 下载超时后用镜像完成安装，最终 lock 恢复官方 PyPI URL。
- `ruff check .` 与 `ruff format --check .` 通过。
- `pytest --cov=rag_lab --cov-fail-under=85`：**88 passed**，覆盖率 **98.07%**。包含 48 个附录 A 合同回归、安全竞态、恢复账本、指标算术和 CLI。
- `uv build`：sdist 与 wheel 构建成功。
- GitHub Actions 首次远端验证通过：[run 38392513316](https://github.com/sawzhang/production-rag-lab/actions/runs/38392513316)，Linux Python 3.11/3.12/3.13 三个任务全部成功（lint、88 项测试、覆盖率门槛与消融）。
- 实际 CLI 报价：PHONE-A-256 / S01 / 到店 / 会员为 450000 分 CNY；数据完全合成，不代表真实售价。
- 五路消融：BM25、LSA dense、RRF hybrid、weighted、overlap rerank，dev 16 题、holdout 8 题。开发集预算实验 6 组。
- 压测：200 次 / 8 worker，**0 错误**；本机 P50 4.19 ms，P95 32.98 ms，P99 40.29 ms；约 617.4 请求/秒。

逐题候选和指标见 `evaluation.json`，逐配置开发实验见 `sweep.json`；机器、请求量和实测统计见 `benchmark.json`。压测包含一半只读合成报价、一半文档搜索，无网络/GPU/真实数据库，不能外推生产容量。

## 怎样解释成绩

保留集 8 条上五种配置的 Recall@4、MRR@4、nDCG@4 均为 1.0。小语料和高词面重叠让该集缺少区分能力，**没有证据证明 hybrid 或 reranker 必须使用或普遍更好**。这是能够复现的实验结果，不是对真实企业效果的承诺。开发集与保留集不是业务专家独立审核的盲测；近似问题族仍有相关性。

88 个测试验证定义好的行为合同，48 个 Q 用例不是声称通用自然语言系统可回答附录的任意问法。覆盖率表示代码执行覆盖，不等于事实正确率、模型质量或安全保证。神经适配器的 100% 覆盖来自 fake model 合同测试；**未下载运行真实模型权重**。

## 未实测范围

真实 SentenceTransformer/CrossEncoder、ANN/数据库、外部生成模型、食品或交易业务接入、OCR、CDC、分布式事务、会话/共享答案缓存、DLP、GraphRAG/learned sparse/late interaction/视觉检索、线上 SSO、影子/canary/A/B 和生产 SLO 均未实测。具体接入与验收见 `docs/production-readiness.md`。CI 多 Python 版本结果以 GitHub Actions 实际状态为准。
