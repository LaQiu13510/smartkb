"""
Milvus 向量存储
===============
基于 PyMilvus 封装向量存储操作。
提供 Collection 创建、向量插入、向量检索等核心能力。
"""

import time
from typing import List, Optional

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)

from config import MILVUS_HOST, MILVUS_PORT, COLLECTION_NAME, VECTOR_DIM


class MilvusStore:
    """Milvus 向量存储封装"""

    def __init__(self):
        self._connected = False
        self._collection: Collection | None = None

    # ---- 连接管理 ----

    def connect(self) -> bool:
        """连接到 Milvus 服务"""
        if self._connected:
            return True
        try:
            connections.connect(
                alias="default",
                host=MILVUS_HOST,
                port=MILVUS_PORT,
                timeout=10,
            )
            self._connected = True
            print(f"[Milvus] 连接成功: {MILVUS_HOST}:{MILVUS_PORT}")
            return True
        except Exception as e:
            print(f"[Milvus] 连接失败: {e}")
            return False

    def _require_connection(self):
        """确保已连接 Milvus，否则抛出清晰的运行时错误。"""
        if self._connected:
            return
        if not self.connect():
            raise RuntimeError(
                f"Milvus 未连接，请确认服务已启动: {MILVUS_HOST}:{MILVUS_PORT}"
            )

    def disconnect(self):
        """断开 Milvus 连接"""
        if self._connected:
            connections.disconnect("default")
            self._connected = False

    # ---- Collection 管理 ----

    def collection_exists(self, name: str | None = None) -> bool:
        """检查 Collection 是否存在"""
        self._require_connection()
        target = name or COLLECTION_NAME
        return utility.has_collection(target)

    def create_collection(
        self,
        name: str | None = None,
        drop_if_exists: bool = False,
    ) -> Collection:
        """创建 Milvus Collection"""
        self._require_connection()
        target = name or COLLECTION_NAME

        if drop_if_exists and self.collection_exists(target):
            utility.drop_collection(target)
            print(f"[Milvus] 已删除旧 Collection: {target}")

        if self.collection_exists(target):
            self._collection = Collection(target)
            self._collection.load()
            print(f"[Milvus] 复用已有 Collection: {target}")
            return self._collection

        # 定义 Schema
        fields = [
            FieldSchema(
                name="id",
                dtype=DataType.VARCHAR,
                max_length=128,
                is_primary=True,
                auto_id=False,
            ),
            FieldSchema(
                name="content",
                dtype=DataType.VARCHAR,
                max_length=65535,
            ),
            FieldSchema(
                name="embedding",
                dtype=DataType.FLOAT_VECTOR,
                dim=VECTOR_DIM,
            ),
            FieldSchema(
                name="file_name",
                dtype=DataType.VARCHAR,
                max_length=512,
            ),
            FieldSchema(
                name="chunk_index",
                dtype=DataType.INT64,
            ),
            FieldSchema(
                name="source_page",
                dtype=DataType.INT64,
            ),
        ]

        schema = CollectionSchema(
            fields=fields,
            description="SmartKB 知识库向量存储",
            enable_dynamic_field=False,
        )

        self._collection = Collection(name=target, schema=schema)

        # 创建索引 (IVF_FLAT: 平衡准确率和速度)
        index_params = {
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024},
        }
        self._collection.create_index(
            field_name="embedding",
            index_params=index_params,
        )
        self._collection.load()
        print(f"[Milvus] 创建新 Collection: {target}")
        return self._collection

    def get_collection(self) -> Collection:
        """获取当前 Collection"""
        self._require_connection()
        if self._collection is None:
            self.create_collection()
        return self._collection

    # ---- 数据操作 ----

    def insert(
        self,
        ids: List[str],
        contents: List[str],
        embeddings: List[List[float]],
        file_names: List[str],
        chunk_indices: List[int],
        source_pages: List[int] | None = None,
    ) -> int:
        """批量插入向量数据，返回插入条数"""
        if source_pages is None:
            source_pages = [0] * len(ids)

        collection = self.get_collection()

        entities = [
            ids,
            contents,
            embeddings,
            file_names,
            chunk_indices,
            source_pages,
        ]

        insert_result = collection.insert(entities)
        collection.flush()
        return insert_result.insert_count

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        expr: str | None = None,
    ) -> List[dict]:
        """
        向量相似度检索

        Args:
            query_embedding: 查询向量
            top_k: 返回结果数
            expr: 过滤表达式，如 'file_name == "xxx.pdf"'

        Returns:
            检索结果列表，每项包含 id, content, score 等字段
        """
        collection = self.get_collection()

        search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": 16},
        }

        results = collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=expr,
            output_fields=[
                "id",
                "content",
                "file_name",
                "chunk_index",
                "source_page",
            ],
        )

        # 格式化结果
        formatted = []
        for hits in results:
            for hit in hits:
                formatted.append({
                    "chunk_id": hit.id,
                    "content": hit.entity.get("content", ""),
                    "file_name": hit.entity.get("file_name", ""),
                    "chunk_index": hit.entity.get("chunk_index", 0),
                    "source_page": hit.entity.get("source_page", 0),
                    "score": hit.score,
                })

        return formatted

    def delete_by_file(self, file_name: str) -> int:
        """根据文件名删除向量数据，返回删除条数"""
        collection = self.get_collection()
        escaped = file_name.replace("\\", "\\\\").replace('"', '\\"')
        expr = f'file_name == "{escaped}"'
        result = collection.delete(expr)
        collection.flush()
        return result.delete_count if hasattr(result, 'delete_count') else 0

    def count(self) -> int:
        """获取 Collection 中的向量总数"""
        collection = self.get_collection()
        return collection.num_entities

    def drop_collection(self, name: str | None = None):
        """删除整个 Collection"""
        self._require_connection()
        target = name or COLLECTION_NAME
        if self.collection_exists(target):
            utility.drop_collection(target)
            self._collection = None
            print(f"[Milvus] 已删除 Collection: {target}")


# 全局单例
_store_instance: MilvusStore | None = None


def get_milvus_store() -> MilvusStore:
    """获取全局 MilvusStore 实例"""
    global _store_instance
    if _store_instance is None:
        _store_instance = MilvusStore()
        _store_instance.connect()
    return _store_instance
