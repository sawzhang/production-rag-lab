# 一手资料与研究入口

原文目录覆盖 61 项资料，本项目按学习主题提供研究入口。源 PDF 完整文献表是进一步阅读依据，不把本文链接当当前采购承诺。

- [RAG 原论文](https://arxiv.org/abs/2005.11401)、[DPR](https://arxiv.org/abs/2004.04906)、[HNSW](https://arxiv.org/abs/1603.09320)
- [BM25](https://nlp.stanford.edu/IR-book/html/htmledition/okapi-bm25-a-non-binary-model-1.html)、[RRF 论文](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)
- [SPLADE](https://arxiv.org/abs/2107.05720)、[ColBERT](https://arxiv.org/abs/2004.12832)、[ColBERTv2](https://arxiv.org/abs/2112.01488)、[ColPali](https://arxiv.org/abs/2407.01449)
- [BEIR](https://arxiv.org/abs/2104.08663)、[Lost in the Middle](https://arxiv.org/abs/2307.03172)、[HyDE](https://arxiv.org/abs/2212.10496)、[Query2doc](https://arxiv.org/abs/2303.07678)
- [Sentence Transformers retrieve/rerank](https://sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html)
- [Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)、[Microsoft GraphRAG](https://microsoft.github.io/graphrag/)
- [OWASP RAG Security](https://cheatsheetseries.owasp.org/cheatsheets/RAG_Security_Cheat_Sheet.html)、[Prompt Injection Prevention](https://cheatsheetseries.owasp.org/cheatsheets/LLM_Prompt_Injection_Prevention_Cheat_Sheet.html)
- [Elasticsearch](https://www.elastic.co/docs)、[OpenSearch](https://docs.opensearch.org/latest/)、[Vespa](https://docs.vespa.ai/)
- [Milvus](https://milvus.io/docs)、[Qdrant](https://qdrant.tech/documentation/)、[Weaviate](https://docs.weaviate.io/)、[pgvector](https://github.com/pgvector/pgvector)、[Azure AI Search](https://learn.microsoft.com/en-us/azure/search/)

运行时接口核对：2026-10-13 阅读 Stanford BM25、Sentence Transformers retrieve/rerank、pgvector 官方说明。产品比较仅作为核对任务，不宣布其他产品功能已在本仓库部署验证。

真实模型接口与 query 前缀依据：[BGE 中文 embedding 模型卡](https://huggingface.co/BAAI/bge-small-zh-v1.5)、[BGE reranker 模型卡](https://huggingface.co/BAAI/bge-reranker-base)。两者固定 SHA 在模型配置中；归一化 cosine 和重排分数仅用于排序，不作为正确概率。

DeepSeek JSON 生成实验使用 [官方 OpenAI 兼容接口](https://api-docs.deepseek.com/)，模型别名 `deepseek-flash`、thinking disabled；记录服务实际返回的模型名和 token 用量。
