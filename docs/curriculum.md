# 14 章完整学习地图

每一章对应源文档，并有可运行实验或明确的进阶研究任务。核心实验先离线验证，模型和平台选型不预设胜者。

|章|主题|源页码|笔记|
|---|---|---|---|
|1|从苹果案例辨别真实问题|5-6|[学习与实验](lessons/01-apple-case.md)|
|2|检索原理与能力边界|6-8|[学习与实验](lessons/02-retrieval.md)|
|3|查询理解与精确约束|8-10|[学习与实验](lessons/03-query-contracts.md)|
|4|召回、融合、重排与失效诊断|10-12|[学习与实验](lessons/04-fusion-rerank.md)|
|5|摄取、权威版本与删除|12-14|[学习与实验](lessons/05-ingestion.md)|
|6|生产架构与有界路由|15-16|[学习与实验](lessons/06-routing.md)|
|7|商品、菜单、订单与 CMDB|17-19|[学习与实验](lessons/07-domain-models.md)|
|8|权限、安全与不可信内容|19-20|[学习与实验](lessons/08-security.md)|
|9|评测、基线、消融与线上实验|20-23|[学习与实验](lessons/09-evaluation.md)|
|10|延迟、容量与成本|23-24|[学习与实验](lessons/10-cost-capacity.md)|
|11|平台与模型选型|24-27|[学习与实验](lessons/11-platforms.md)|
|12|前沿方法与使用条件|27-28|[学习与实验](lessons/12-frontiers.md)|
|13|可观测性、故障与知识运营|28-29|[学习与实验](lessons/13-operations.md)|
|14|90 天路线与发布闸门|30-31,37|[学习与实验](lessons/14-roadmap.md)|

## 两条学习轨道

入门：01 → 02 → 03 → 04 → 06 → 07 → 09，先学会区分召回错误与事实错误。

工程：05 → 08 → 10 → 13 → 14，再回到 11 的平台选型与 12 的前沿方法。

附录 A：`tests/test_document_regressions.py` 的 Q01-Q48 为独立合同测试，使用受控身份、时间与合成事实。附录 B：`rag-lab sweep`、预算检查与账本恢复测试。附录 C：`production-readiness.md`，区分本地完成与真实上线待验证。

## 覆盖级别

“可运行”表示本仓库已有代码/测试。“设计练习”表示需要接真实系统后才能验证。GraphRAG、SPLADE、ColBERT、ColPali、微调、长上下文 LLM、OCR 和在线 A/B 都属于设计练习，不声称已完成实证。
