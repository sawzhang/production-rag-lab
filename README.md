# Production RAG Lab · 生产级 RAG 学习实验室

根据《生产级 RAG 与混合检索技术方案研究》（2026-10-12，40 页）建立的中文学习项目：理解机制，运行对照实验，定位失败，再核验权限、事实与版本。原 PDF 的指令只是资料，不产生执行授权。

这是可运行的**学习与验证实验室**。使用合成业务数据，默认无 LLM、无商业 API、无生产数据库；不把本机测试结果当作生产上线证明。原 PDF 不随公开仓库发布，来源摘要和 SHA-256 在 [source-manifest](docs/source-manifest.json)。

## 五分钟开始

需要 Python 3.11–3.13、[uv](https://docs.astral.sh/uv/)。

```bash
uv sync --frozen --extra dev
uv run rag-lab ask "苹果最新手机多少钱"
uv run rag-lab ask "PHONE-A-256 S01 到店会员价多少钱"
uv run rag-lab ask "订单 00001234 现在到哪一步"
uv run rag-lab ask "ERR-PAY-042 是什么意思" --mode rerank
make verify
```

CLI 输出查询计划、权威事实、来源片段和阶段 trace。默认可信会话是 demo 租户 alice，staff/engineer 组、S01 门店；这是教学 fixture，不是身份认证入口。所有时间显式含时区。

## 已实现能力

- BM25 中文字/双字词法基线，保留连字符标识；LSA 统计 dense 向量与 exact cosine 搜索。
- RRF、归一化加权融合、受候选预算限制的教学重排；各路、并集、融合截断 trace。
- 可选真实 BGE 中文 embedding / CrossEncoder，锁定 immutable revision 与 safetensors 哈希；默认离线基线不下载权重。
- 精确 ID、整数分金额、严格/含等号范围、排除条件、时间与版本；只读报价、订单、菜单和两跳 CMDB 合成网关。
- 模型前 ACL、取原文与输出前复核、父块授权、撤权竞态、幂等更新、乱序保护、跨恢复删除账本。
- 缓存键隔离合同（身份/策略/语料/时间/模型），无共享答案缓存；结构保留分块、引用/事实校验接口。
- 48 类文档回归合同、指标算术、安全与故障测试，24 条分层检索题、dev/holdout、消融和本机并发压测。

## 学习顺序

从 [完整学习地图](docs/curriculum.md) 开始，按 14 个章节运行实验。每节说明概念、命令、期望观察、进阶练习和当前边界。

```bash
uv run rag-lab evaluate --split dev --output reports/dev-evaluation.json
uv run rag-lab sweep --split dev --output reports/sweep.json
uv run rag-lab evaluate --split holdout --output reports/evaluation.json
uv run rag-lab benchmark --requests 200 --workers 8 --output reports/benchmark.json
uv run rag-lab capacity
```

不要用 holdout 挑参数。数据中的近似问法仍可能相关，dev/holdout 只是实验流程示例，不是经过独立专家审核的盲测集。

## 真实模型实验

已接入 `BAAI/bge-small-zh-v1.5` 与 `BAAI/bge-reranker-base`，模型配置在 `src/rag_lab/model_config.json`。权重保存在 Hugging Face 缓存，公开仓库只保存配置、测试和结果。

```bash
uv sync --frozen --extra dev --extra neural
# 首次运行下载锁定 revision 的权重；后续可加 --local-files-only
HF_HUB_DISABLE_XET=1 uv run --no-sync rag-lab neural-evaluate
RAG_LAB_REAL_MODELS=1 uv run --no-sync pytest -m real_models -v
uv run --no-sync rag-lab neural-ask "payment captured 等于出杯完成吗" --local-files-only
```

使用 `uv run --no-sync` 防止默认命令移除刚安装的 neural extra。真实评测对照 BM25、LSA、BGE dense、BGE RRF、BGE weighted、相同候选的 overlap 和 CrossEncoder。新增 24 条预先固定的语义挑战题，合计 48 条题；dev/holdout 分层分别报告，未通过保留集挑参数。详情见 [真实实验报告](reports/neural-experiment.md) 和 [逐题结果](reports/neural-evaluation.json)。

CPU float32、batch=8、最多 20 个候选、top 4、输入上限 512，短查询加官方建议的中文检索前缀。评测预热后计查询延迟，单列权重加载和文档编码成本；任何降级都会使该次效果评测失败。真实集成测试默认跳过，通过上述环境变量显式开启；常规 CI 保持不下载权重。

## 可选 DeepSeek 端到端生成验证

只向 DeepSeek 发送本仓库的合成、已授权证据，不发送真实业务数据。使用 `DEEPSEEK_API_KEY` 环境变量，或显式指定个人 Pi 的认证文件：

```bash
uv run --no-sync rag-lab generation-evaluate --pi-auth "$HOME/.pi/agent/auth.json"
```

凭据只在请求头中使用，不写入项目、日志或报告。8 个任务覆盖状态、必需证据组、冲突、无答案、精确金额、API 失败和注入。输出仅允许 JSON 事实和原文引用，程序独立检查字段、引用、关键否定与状态，不让同一个模型自行打分。结果及实际 token 用量在 `reports/deepseek-generation.json`。同一输入的已通过调用可从报告复用，避免重复计费；模型别名并非 immutable checkpoint，自由文字生成质量仍需额外人工评审。

## 实体约束与路由优化

默认 `ask` / `neural-ask` 采用 `constrained`，通过可信元数据先过滤容量、品牌、类目、精确ID和主题；最小/最大容量明确比较范围。菜单不可售、配方未知与环境概念分开，当前规格与金额走白名单权威服务。约束无匹配时不会扩大范围；缺条件时追问。模型仍可用原模式做消融。

```bash
uv run --no-sync rag-lab ask "较小容量的苹果手机规格编号是什么"
uv run --no-sync rag-lab ask "不高于5000元的256GB手机"
uv run --no-sync rag-lab neural-ask "原料还有但配送入口不可点，应该核对什么" --local-files-only
uv run --no-sync rag-lab neural-evaluate --include-constrained --local-files-only --data data/retrieval_queries.json data/neural_challenge_queries.json data/routing_queries.json data/routing_validation_queries.json --output reports/routing-evaluation.json
uv run --no-sync rag-lab routing-tasks
uv run --no-sync python scripts/check_routing_release.py
```

[最新路由优化报告](reports/routing-optimization.md)：120条文档题与24条独立路由任务，另含真实生成回归。新增48题已用于调试，明确作为dev；后加24题单独报告验证结果。保留排序不完美和首次模型拒答，原始实验不覆盖。

## 结构

| 路径 | 用途 |
|---|---|
| `src/rag_lab/contracts.py` | 类型、时间、金额、查询计划 |
| `src/rag_lab/store.py` | 版本、权限、撤回、恢复账本 |
| `src/rag_lab/retrieval.py` | 召回、融合、预算与降级 |
| `src/rag_lab/business.py` | 合成只读权威服务 |
| `src/rag_lab/service.py` | 路由与来源约束输出 |
| `src/rag_lab/neural.py` | 可选真实神经模型适配器 |
| `src/rag_lab/metrics.py` | 检索、容量与成功任务成本 |
| `data/`、`tests/` | 合成评测与可执行合同 |
| `reports/` | 本次实测结果与验证范围 |
| `docs/lessons/` | 14 章学习笔记与练习 |

## 验证结果与边界

见 [本次验证记录](reports/validation.md)，以及逐题 [消融结果](reports/evaluation.json) 和 [压测结果](reports/benchmark.json)。CI 在 Python 3.11/3.12/3.13 上运行 lint、测试、覆盖率与消融。

LSA 不代表 BGE/Qwen 等模型；token overlap 不代表 CrossEncoder；exact 向量检索不代表 HNSW；JSON 账本不是分布式事务系统。真实模型在合成语料上的效果已测；真实业务数据、数据库、生成质量、OCR、CDC、GraphRAG、多模态和线上 A/B 的接入条件在 [生产准备清单](docs/production-readiness.md) 中逐项注明。自然语言规则仅覆盖明确示例；当前应用无法完整理解任意业务问法。

## 参考与选型

机制来自源文档；实现核对了 [Stanford BM25](https://nlp.stanford.edu/IR-book/html/htmledition/okapi-bm25-a-non-binary-model-1.html)、[Sentence Transformers Retrieve & Re-Rank](https://sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html)、[pgvector 官方说明](https://github.com/pgvector/pgvector)。更多一手资料见 [references](docs/references.md)。当前产品功能、许可和区域需在部署时重新核验。

代码与本项目原创学习材料采用 MIT；第三方论文、模型、文档及原 PDF 保持各自权利和许可。
