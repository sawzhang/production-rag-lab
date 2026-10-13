# 平台与模型实验记录模板

本项目不采购平台，不对滚动产品功能做未经实测的排名。八个选项分别是 Elasticsearch、OpenSearch、Vespa、Milvus、Qdrant、Weaviate、PostgreSQL+pgvector、Azure AI Search。已有搜索或 SQL 系统优先测原位扩展，避免过早引入双写状态。

|项目|实际记录|
|---|---|
|发行版本/API/区域/套餐|待填写 immutable 版本与实际部署|
|BM25 / dense / fusion / rerank|逐项说明 native、plugin、application 或 external inference|
|过滤与权限|前/后/迭代过滤、DLS/RLS、对象/字段权限、绕过角色|
|模型|ID、revision、许可证、模板、长度、截断、精度、batch、硬件|
|更新与删除|CDC、水位、tombstone、各派生物回执、恢复账本|
|质量|同一合法 gold、同一预算、同一保留集，分层指标|
|性能与成本|相同复制、更新、硬件预算、Recall 目标与错误率|
|迁移|新索引、新模型配对切换，删除/权限账本保持前进|
|停止条件|严重越权/金额错阻断，其他门槛由业务与平台批准|

PostgreSQL 全文 ts_rank 并不是 BM25；RRF 是融合，不是 CrossEncoder；tenant 字段不是可信身份；preview 功能不能按 GA 验收；模型名称不是效果或许可保证。以上提醒应通过实际发行版和官方材料核对。

真实模型实验使用 `uv sync --extra dev --extra neural`，然后显式构造 `NeuralDense(model_id, full_commit_sha)` 和 `NeuralReranker(model_id, full_commit_sha)`。模型权重缓存放仓库外，不提交。外部模型运行时需遵守来源数据允许的传输边界。
