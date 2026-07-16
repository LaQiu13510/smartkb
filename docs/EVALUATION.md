# SmartKB 测试与评测

## 离线测试

```bash
python test_imports.py
python test_e2e.py
python test_api.py
```

覆盖范围：

- 配置和核心模块导入。
- Redis 查询缓存的读写、命中和失效。
- Embedding 语义边界分块与词汇降级。
- BM25 索引和混合检索主链路。
- 上下文去重、字符与 Token 预算、来源标注和敏感信息脱敏。
- FastAPI 请求校验、文档上传、缓存命中和 SSE Token 流。

## 检索评测

```bash
python eval/retrieval_eval.py
```

离线评测使用固定文档集和查询集，对向量检索、BM25 和混合检索分别计算：

| 指标 | 含义 |
| --- | --- |
| `Top-1 Accuracy` | 第一条结果是否属于相关文档 |
| `Recall@3` | 前三条结果覆盖了多少相关文档 |
| `MRR@3` | 第一条相关文档出现得越靠前，分数越高 |
| `nDCG@3` | 同时衡量相关结果是否被召回以及排序位置 |
| 平均检索延迟 | 离线数据集上的单次检索耗时，仅用于策略对比 |

评测脚本输出完整 JSON，包括数据集规模、各策略汇总指标和逐问题排序明细，便于在修改检索策略后进行回归对比。

## Agent 路由评测

```bash
python eval/agent_eval.py
```

该脚本验证 SmartKB 的 `chat`、`list` 和 `retrieve` 路由是否符合预期。

## 真实服务检查

```bash
python test_imports.py --live
python test_e2e.py --live
```

真实检查会访问配置的 LLM、Embedding、Milvus 和 PostgreSQL，不应在未配置密钥的公共 CI 环境中执行。
