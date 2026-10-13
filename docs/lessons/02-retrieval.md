# 检索原理与能力边界

对应源文档页码：6-8。本笔记为独立学习改编。

## 要理解的机制

BM25 的 tf 饱和与长度归一化影响词法评分；本项目使用正值 IDF 变体。LSA 是 TF-IDF 的低秩表示，形成真实 dense 向量，但不能代表神经 embedding 的中文能力。精确 cosine 搜索没有 ANN 误差。learned sparse、ColBERT 和 CrossEncoder 是不同机制。

## 可运行实验

```bash
uv run rag-lab evaluate --split dev
```

## 验证与观察

查看 retrieval.py 的 BM25、SVD、余弦和 RRF，手算两条候选的分数；测试集合为空和只剩一条时的行为。

## 进阶练习与边界

用 neural.py 接入锁定 revision 的 bi-encoder 和 CrossEncoder；固定语料和候选预算。进一步引入 ANN 时，用 exact 子集测交集召回。

## 自测

1. 哪种失败不能通过更强的检索模型修复？
2. 本实验锁定了哪些条件，还缺少哪些真实业务证据？
3. 如果撤权、超时或来源冲突发生在这一阶段，应怎样输出？
