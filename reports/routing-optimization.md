# 实体约束与检索路由优化 · 2026-10-13

本轮固定同一BGE embedding/reranker revision、CPU float32、batch 8、4线程和语料正文，比较无约束混合重排与实体/主题约束策略。优化来自确定性边界与证据契约，未训练或更换模型。历史实验保存在 neural-experiment.md 和原 JSON，不用不同分母的旧均值直接宣称提升。

## 改动与失效原因

- 容量：新增可信category/brand/capacity_gb/style/entities字段；最小/最大在当前授权有效比较范围内计算，GB数值、严格/含等号区间、规格ID、品牌与排除条件在模型前过滤。缺元数据不被当作满足约束，零结果不放宽范围；相对范围在编码期间发生更新时失败关闭。
- 交易路径：只读catalogue工具返回精确规格及selection比较范围、已匹配数量、极值和谓词。报价额外核验用户指定容量、币种、品牌、款式，冲突时不返回金额。
- 菜单：availability与recipe_safety分开；主题元数据由来源登记赋予，原料/配送不可点不会将配方未知或注入文档送入模型。跨主题问题保留相关主题并集，发布问题同时保留价格和菜单契约。
- 环境：同名生产/沙箱的概念解释走规则文档；真实消费者查询必须有受控服务身份和明确环境，不再默认查menu。两个环境同时指定时追问。
- 召回扩展仅追加受控同义词，保留原始ID/数字/否定；CrossEncoder始终接收原始问题。首轮发现扩展词导致菜单排序偏向发布契约，已通过召回/重排输入分离修复。
- ACL仍在筛选之前和输出时执行，主题过滤不改变可信Identity。策略没有引用评测题编号或gold文档ID。

## 评测分层与边界

- 原48题是历史回放，已知N06/N10/N20用于定位问题，不再把它们视为本轮盲测。
- 新48题C01-C48包含容量、菜单、配方、CMDB与反例控制，已经用于发现回归，split明确改为dev，题目和gold未按模型结果改写。
- 另24题V01-V24在第二轮修复前固定，单独报告验证结果；随后没有依据这些结果调整排序规则。同一作者、同一合成小语料，近似问法与文档家族仍相关，不能视为独立专家盲审或统计显著证据。
- 120条有合法gold的文档题参与检索指标；24条catalogue/报价/追问/无答案路由任务独立验收，不混入Recall分母。
- 输入文件、可信元数据corpus及路由源码有SHA-256；主实验20/20/top4与紧预算5/8/top4均真实运行，无重排降级。

## 同一120题上的对照

|策略|Recall@4|MRR@4|nDCG@4|必需组覆盖|CPU P50 ms|CPU P95 ms|
|---|---:|---:|---:|---:|---:|---:|
|原混合+CrossEncoder|0.9792|0.9250|0.9168|0.9792|299.05|387.31|
|约束+原问题CrossEncoder|1.0000|1.0000|0.9864|1.0000|10.76|284.51|

|分层|样本数|原nDCG@4|约束nDCG@4|约束Recall@4|
|---|---:|---:|---:|---:|
|neural_challenge_queries/holdout|24|0.9001|0.9831|1.0000|
|retrieval_queries/dev|16|1.0000|1.0000|1.0000|
|retrieval_queries/holdout|8|1.0000|1.0000|1.0000|
|routing_queries/dev|48|0.8951|0.9831|1.0000|
|routing_validation_queries/holdout|24|0.8938|0.9831|1.0000|

主实验逐题nDCG未观察到低于对应无约束基线的结果；这不是未来零回归保证。延迟是已预热的本机请求，不含权重加载/下载；缩小合法候选池减少CPU推理，不能据此外推生产吞吐。

|紧预算策略|Recall@4|nDCG@4|覆盖率|
|---|---:|---:|---:|
|neural_hybrid_cross_encoder|0.9792|0.9168|0.9792|
|constrained_cross_encoder|1.0000|0.9864|1.0000|

## 原退步样本与剩余错误

|样本|原排名/分数|约束排名/分数|
|---|---|---|
|N06 原料还有但配送入口不可点，应该核对什么|['unknown', 'menu-contract', 'injection', 'order-timeout'] / 0.1738|['menu-contract', 'menu-channel'] / 0.7967|
|N10 同名服务在沙箱和线上可以当成同一个对象吗|['price-rule', 'incident1', 'conflict-a', 'menu-contract'] / 0.0000|['graph-rule'] / 1.0000|
|N12 缺少消息处理日志就表示消息丢了吗|['order-timeout', 'incident2', 'injection', 'graph-rule'] / 0.7967|['order-timeout', 'incident2'] / 0.7967|
|N20 较小容量的苹果手机规格编号是什么|['phone256', 'phone128', 'case', 'numeric'] / 0.6309|['phone128'] / 1.0000|

N06和N12均已覆盖全部必需证据，但不同相关性等级的顺序仍非最佳（nDCG约0.7967）。保留该限制，不把全部命中写成完美排序。容量API的范围是授权合成catalogue；文档极值的范围是当前可见且有效的规格文档，两者都显式携带scope，不能推断全球最小容量。

## 任务、真实生成与发布检查

- 24条路由合同全部通过：精确容量/价格边界、Pro排除、币种、未知规格、规格矛盾、历史能力缺失、环境身份与概念解释。
- DeepSeek优化路径最初11/12通过。G09把合法catalogue结果误拒为insufficient，因为输入仅给单个规格，没有比较谓词和范围；校验器阻断了该草稿。失败原文与usage保存在 deepseek-routing-first-pass.json。
- 在权威fact中补充真实程序计算的selection（比较范围、匹配数量、极值、谓词），同一题再次调用后12/12通过；没有放宽校验或修改预期状态。Service本身能直接返回确定性fact，不必依赖LLM重写该事实。
- 8个已有请求输入完全一致时复用以前的真实API记录；另外4个新场景和G09改变证据后的再次请求是真实调用。最终12条记录合计 4649 tokens，另外保留首次G09失败的 319 tokens；不是声称这些累计记录全部在本轮重新计费。
- 输出仍为原文引用和原样fact，独立校验JSON、整数类型、引用、必需来源、否定和预期状态，不使用LLM自评。凭据不进入文件或日志。
- 全套 **154项测试通过**，含6项实际权重测试，覆盖率 **95.44%**；检查撤权、删除、未知元数据、元数据更新竞态、零权限、未知品牌、报价冲突及快照恢复。lint、格式检查与wheel构建通过。
- scripts/check_routing_release.py核验记录中的输入/源码指纹、无降级、逐题无退步、关键失效修复、证据覆盖、24路由任务和12生成合同，所有检查通过。该脚本检查已有实验记录，不会重新运行模型。

## 复现

```bash
uv sync --frozen --extra dev --extra neural
uv run --no-sync rag-lab neural-evaluate --include-constrained --local-files-only --data data/retrieval_queries.json data/neural_challenge_queries.json data/routing_queries.json data/routing_validation_queries.json --output reports/routing-evaluation.json
uv run --no-sync rag-lab neural-evaluate --include-constrained --local-files-only --profile src/rag_lab/model_config_tight.json --data data/retrieval_queries.json data/neural_challenge_queries.json data/routing_queries.json data/routing_validation_queries.json --output reports/routing-budget-evaluation.json
uv run --no-sync rag-lab routing-tasks
RAG_LAB_REAL_MODELS=1 uv run --no-sync pytest --cov=rag_lab --cov-fail-under=85
uv run --no-sync rag-lab generation-evaluate --optimized --pi-auth "$HOME/.pi/agent/auth.json" --output reports/deepseek-routing-generation.json
uv run --no-sync python scripts/check_routing_release.py
```

这些结果仅适用于20条合法文档与5个合成商品变体的小型实验。品牌字典、主题与表达范围有限；生产仍需来源owner审核元数据、完整对象ACL、可靠同步、真实业务评测和峰值负载验证。
