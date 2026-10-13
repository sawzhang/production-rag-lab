# 验证记录 · 2026-10-13

## 首次创建时的实际执行（commit 0754310）

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

## 真实模型与生成补充（当前）

- 实际下载并校验 BGE small zh v1.5 与 BGE reranker base 固定 revision 的 safetensors，真实离线推理通过；版本及权重 SHA-256 见 `src/rag_lab/model_config.json`。
- `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 RAG_LAB_REAL_MODELS=1 pytest --cov=rag_lab --cov-fail-under=85`：**105 passed**，含5项真实权重集成测试；覆盖率 **95.89%**。原始98.07%属于首次创建时的旧代码。
- 主实验48题/7配置、紧预算48题/7配置均已真实运行，无重排降级。主实验在参数与gold不变的情况下串行重复一次，避免并行测试影响延迟；排名指标与首轮一致。
- 全体48题，BM25 Recall@4=0.9271、nDCG@4=0.8945；BGE混合+CrossEncoder Recall@4=0.9688、nDCG@4=0.9500。保留所有切片、逐题排名和退步样本，不宣称普遍最优。
- DeepSeek真实端到端JSON抽取：**8/8通过**；3116总tokens（另有接口smoke 280 tokens）。使用已有个人凭据，仅发送合成授权数据；token不写入仓库或报告。
- 新代码lint、格式检查和构建通过；wheel包含模型配置。常规CI运行离线测试，5项实际模型测试在本机显式开启。

详细方法、模型版本、延迟和范围见 [真实模型报告](neural-experiment.md)。本节更新了上文“真实模型未实测”的初始状态；真实业务语料、生产SLO、ANN/数据库、长文截断、自由文字生成质量和线上A/B仍未验证。
