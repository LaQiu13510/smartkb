# SmartKB 部署与运行

## 运行要求

- Python 3.11 或兼容版本
- DeepSeek 兼容聊天模型 API
- 至少一个 Embedding 服务或本地 Embedding 模型
- PostgreSQL
- Milvus
- Redis，可选；未配置时自动使用进程内缓存

## 本地运行

```bash
python -m venv .venv
python -m pip install -r requirements.txt
cp .env.example .env
uvicorn app:app --host 0.0.0.0 --port 8501
```

启动后访问：

- Web 页面：`http://127.0.0.1:8501`
- OpenAPI 文档：`http://127.0.0.1:8501/docs`
- 健康检查：`http://127.0.0.1:8501/api/health`

## Docker Compose

```bash
docker compose config
docker compose up --build
```

Compose 包含 FastAPI、PostgreSQL、Redis、Milvus、etcd 和 MinIO。数据库和向量库使用命名卷持久化，应用通过健康检查等待依赖服务就绪。

`.env` 为可选加载文件；仓库可以在没有真实密钥的情况下完成 Compose 配置解析，但调用模型和 Embedding 前仍需填写对应配置。

## 上线前检查

```bash
python test_imports.py
python test_e2e.py
python test_api.py
python eval/agent_eval.py
python eval/retrieval_eval.py
```

配置真实服务后再执行：

```bash
python test_imports.py --live
python test_e2e.py --live
```

## 常见问题

| 现象 | 排查方向 |
| --- | --- |
| Embedding 调用失败 | 检查服务密钥、网络和模型名称；分块会自动降级，但向量入库仍需要可用 Embedding |
| Milvus 维度不一致 | 确认 `VECTOR_DIM` 与 Embedding 输出维度一致，必要时重建 Collection |
| PostgreSQL 写入失败 | 检查 `DB_URL`、账号权限和网络连通性 |
| 查询没有上下文 | 确认文档已入库、Milvus 有向量、BM25 索引已刷新 |
| SSE 被代理缓冲 | 关闭代理缓冲，并保留 `X-Accel-Buffering: no` 响应头 |
| Redis 不可用 | 检查 `REDIS_URL`；系统会降级到进程内缓存，但多实例之间不共享缓存 |
