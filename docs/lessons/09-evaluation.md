# 评测、基线、消融与线上实验

对应源文档页码：20-23。本笔记为独立学习改编。

## 要理解的机制

Recall、HitRate、MRR、nDCG 与必需证据组覆盖回答不同问题。无合法 gold 记 N/A；API 题不进入文档召回分母。nDCG 使用 2^rel-1 和 log2(rank+1)。24 条合成检索题与 48 个合同测试不足以估计真实业务成绩。

## 可运行实验

```bash
uv run rag-lab evaluate --split holdout --output reports/evaluation.json
```

## 验证与观察

看 macro 和 slice 样本量，检查多 gold 的 Recall 与 HitRate 差异。union 和 fused 截断都在逐题结果中。

## 进阶练习与边界

扩充 300-500 条经过领域双人审核的真实题；按源文档/事件族切分，先影子，再 canary，再按用户稳定分桶 A/B。

## 自测

1. 哪种失败不能通过更强的检索模型修复？
2. 本实验锁定了哪些条件，还缺少哪些真实业务证据？
3. 如果撤权、超时或来源冲突发生在这一阶段，应怎样输出？
