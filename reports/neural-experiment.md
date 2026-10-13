# 真实模型实验报告 · 2026-10-13

本篇保留首次真实模型基线。后续实体约束、容量与菜单路由改动及扩充评测见 [最新优化报告](routing-optimization.md)。

## 本次实际运行的模型与协议

- Embedding：`BAAI/bge-small-zh-v1.5@7999e1d3359715c523056ef9478215996d62a620`；中文 query prefix，512 输入上限，归一化 cosine。
- Reranker：`BAAI/bge-reranker-base@2cfc18c9415c912f9d8155881c133215df768a70`；真实 CrossEncoder 双文本输入，512 输入上限。
- 两份 safetensors SHA-256 与官方固定 revision 的 LFS 元信息一致；已实际读取并验证，字节数和哈希保存在结果文件。权重在个人 HF 缓存中，不提交 Git。
- 固定 CPU / float32 / batch 8 / torch 4 线程 / seed 42；RRF c=60，weighted alpha=0.5。重排等待预算 30 秒，任何降级使效果实验失败。
- 原有 24 题（dev 16、holdout 8）加事先固定的 24 条语义挑战题；均为合成中文小库。未用这些结果选模型、改 gold 或调融合参数。
- 主实验 window=20 / candidate budget=20 / top 4；紧预算对照 window=5 / candidate budget=8 / top 4。相同融合候选用于 overlap 和 CrossEncoder。当前合法文档只有20条，所以主实验近似全合法库精排；紧预算实验另外观察召回漏失。

## 检索与重排效果

下表主效果采用24条语义挑战题，延迟采用全部48条请求。原有 dev/holdout 的分层成绩与每题排名均在 JSON 中保留。

|配置|挑战 Recall@4|挑战 MRR@4|挑战 nDCG@4|全体 Recall@1|P50 ms|P95 ms|
|---|---:|---:|---:|---:|---:|---:|
|bm25|0.8542|0.7778|0.7890|0.7812|1.19|1.54|
|lsa_dense|0.8542|0.8125|0.8116|0.7917|2.57|3.36|
|neural_dense|0.9167|0.8264|0.8426|0.8125|6.96|8.06|
|neural_hybrid_rrf|0.9375|0.8576|0.8627|0.8333|7.38|8.23|
|neural_hybrid_weighted|0.8958|0.8229|0.8238|0.8021|7.34|8.21|
|neural_hybrid_overlap|0.8958|0.7708|0.7962|0.7708|7.56|8.83|
|neural_hybrid_cross_encoder|0.9375|0.9167|0.9001|0.8646|259.58|309.78|

CPU推理延迟不包含下载、加载和预热。加载 5.11 秒；预热及合法文档编码 0.45 秒。神经文档向量使用内存缓存，query仍实际编码；LSA基线仍逐题fit。该差异须纳入延迟解释，不能直接外推平台吞吐或成本。

## 固定候选的成对比较

|左侧相对右侧|平均 nDCG 差值（48题）|胜/平/负|
|---|---:|---|
|neural_dense vs bm25|+0.0268|5/39/4|
|neural_hybrid_rrf vs neural_dense|+0.0101|4/42/2|
|neural_hybrid_cross_encoder vs neural_hybrid_rrf|+0.0187|4/43/1|
|neural_hybrid_cross_encoder vs neural_hybrid_overlap|+0.0519|6/40/2|

所有神经重排请求的降级数为 0。不同模型和融合方案不保证逐题胜出，不能只展示平均值或成功案例。

## 紧候选预算

|配置|挑战 Recall@4|挑战 nDCG@4|候选 Recall 宏均值（48题）|
|---|---:|---:|---:|
|bm25|0.8542|0.7890|0.9583|
|lsa_dense|0.8542|0.8116|0.9583|
|neural_dense|0.9167|0.8426|0.9688|
|neural_hybrid_rrf|0.9375|0.8413|0.9792|
|neural_hybrid_weighted|0.9167|0.8471|0.9792|
|neural_hybrid_overlap|0.8958|0.7962|0.9792|
|neural_hybrid_cross_encoder|0.9375|0.9001|0.9792|

## 失败与退步样本

- N06「原料还有但配送入口不可点，应该核对什么」：gold `['menu-channel', 'menu-contract']`，CrossEncoder最终 `['unknown', 'menu-contract', 'injection', 'order-timeout']`；nDCG=0.1738，BM25=0.4131。
- N10「同名服务在沙箱和线上可以当成同一个对象吗」：gold `['graph-rule']`，CrossEncoder最终 `['price-rule', 'incident1', 'conflict-a', 'menu-contract']`；nDCG=0.0000，BM25=0.0000。
- N12「缺少消息处理日志就表示消息丢了吗」：gold `['incident2', 'order-timeout']`，CrossEncoder最终 `['order-timeout', 'incident2', 'injection', 'graph-rule']`；nDCG=0.7967，BM25=0.7967。
- N20「较小容量的苹果手机规格编号是什么」：gold `['phone128']`，CrossEncoder最终 `['phone256', 'phone128', 'case', 'numeric']`；nDCG=0.6309，BM25=1.0000。

对容量、数字、ID及状态问题，应继续走权威字段和精确约束，不能让语义评分代替逻辑判断。多gold题中一个片段有用并不代表所有必需证据都已覆盖。

## DeepSeek 端到端生成

真实 `deepseek-flash` API：temperature 0、thinking disabled、最多768输出 tokens。仅发送合成授权片段和权威fact；不用LLM自评，而由独立程序校验JSON schema、原样fact（包括整数类型）、引用原文、必需来源组、关键否定和预期状态。输出为抽取式JSON，不产生未检查的自由文字主张。

|任务|预期|实际|合同检查|
|---|---|---|---|
|G01 钱付了就代表饮料做好了吗|answered|answered|通过|
|G02 退款受理资金到账|answered|answered|通过|
|G03 发布前检查价格菜单接口契约|answered|answered|通过|
|G04 优惠规则会员折扣叠加|conflict|conflict|通过|
|G05 ERR-PAY-999 怎么修|insufficient|insufficient|通过|
|G06 PHONE-A-256 S01 到店会员价多少钱|answered|answered|通过|
|G07 PHONE-A-256 S01 到店多少钱|insufficient|insufficient|通过|
|G08 当前订单是否已完成|insufficient|insufficient|通过|

完成 8/8，任务通过 8/8；实际使用 2804 输入、312 输出、3116 总 tokens（不含此前接口 smoke 的280 tokens）。凭据不进入日志、提示或报告。

G08对抗证据直接注入生成边界，其余任务经过真实BGE/权威网关再调用API。检查原文引用不自动证明来源真实性；本文使用可控合成真值。DeepSeek服务模型是可变别名，并非可下载的immutable权重；温度0也不保证外部服务逐字确定。该8题实验不能衡量自由文字答案的完整语义groundedness或真实业务正确率。

## 测试、复现与限制

- 最终全套测试覆盖率 95.89%，测试数量和真实权重集成结果见 validation.md。
- 原模型revision、输入文件hash、corpus hash、运行环境版本、逐题分数、候选ID和延迟在 `neural-evaluation.json`；紧预算在 `neural-budget-evaluation.json`。
- 权重已完成真正离线加载；真实集成测试检查向量维度/范数、正负排序、ACL、零权限、精确报价、撤权/删除、缓存更新与重复推理。
- 未测试真实业务语料、ANN、数据库、生产并发、长文截断、线上A/B或全量模型选型。各切片样本很小，近似题/文档家族仍相关，gold不是独立专家盲审，不能把这些点估计解释为统计显著或生产SLO。

```bash
uv sync --frozen --extra dev --extra neural
uv run --no-sync rag-lab neural-evaluate --local-files-only
uv run --no-sync rag-lab neural-evaluate --profile src/rag_lab/model_config_tight.json --output reports/neural-budget-evaluation.json --local-files-only
RAG_LAB_REAL_MODELS=1 uv run --no-sync pytest --cov=rag_lab --cov-fail-under=85
uv run --no-sync rag-lab generation-evaluate --pi-auth "$HOME/.pi/agent/auth.json"
```
