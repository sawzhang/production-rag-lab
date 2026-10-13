# 摄取、权威版本与删除

对应源文档页码：12-14。本笔记为独立学习改编。

## 要理解的机制

来源 ID、稳定 chunk ID、revision、标题、ACL、生效区间和 lineage 是最小可解释契约。版本事件需要幂等且拒绝乱序覆盖。块的边界保留表头、单位、否定和例外。删除先阻断可见，再清理派生物。

## 可运行实验

```bash
uv run pytest tests/test_invariants.py -k "ledger or idempotency or chunks"
```

## 验证与观察

保存旧快照后删除文档，恢复到新进程时先装删除和撤权账本，确保旧索引不能复活内容。

## 进阶练习与边界

教学存储使用内存和 JSON。生产应换成事务存储、持久 tombstone/CDC/outbox，加入 parser/chunker/embedding revision、源水位和清理回执。

## 自测

1. 哪种失败不能通过更强的检索模型修复？
2. 本实验锁定了哪些条件，还缺少哪些真实业务证据？
3. 如果撤权、超时或来源冲突发生在这一阶段，应怎样输出？
