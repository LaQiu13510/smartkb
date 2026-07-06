"""
PostgreSQL 元数据存储
=====================
使用 SQLAlchemy 管理文档元数据、对话历史等结构化数据。
"""

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from config import DB_URL


# ---- ORM 基类 ----

class Base(DeclarativeBase):
    pass


# ---- 数据模型 ----

class Document(Base):
    """文档元数据表"""
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    file_name = Column(String(512), nullable=False, unique=True)
    file_type = Column(String(32), nullable=False, default="txt")
    file_size = Column(Integer, default=0)
    chunk_count = Column(Integer, default=0)
    total_chars = Column(Integer, default=0)
    status = Column(String(32), default="uploaded")  # uploaded | chunked | indexed
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ChatHistory(Base):
    """对话历史表"""
    __tablename__ = "chat_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), nullable=False, index=True)
    role = Column(String(32), nullable=False)  # user | assistant | system
    content = Column(Text, nullable=False)
    sources = Column(Text, nullable=True)  # JSON string of source documents
    latency_ms = Column(Float, default=0.0)  # 响应延迟 (毫秒)
    created_at = Column(DateTime, default=datetime.utcnow)


class EvaluationRecord(Base):
    """RAG 评测记录表"""
    __tablename__ = "evaluation_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    question = Column(Text, nullable=False)
    expected_answer = Column(Text, nullable=True)
    generated_answer = Column(Text, nullable=True)
    hit_rate = Column(Float, default=0.0)     # 检索命中率
    mrr = Column(Float, default=0.0)          # 平均倒数排名
    latency_ms = Column(Float, default=0.0)   # 响应延迟
    created_at = Column(DateTime, default=datetime.utcnow)


# ---- 数据库管理 ----

class PostgresStore:
    """PostgreSQL 存储封装"""

    def __init__(self, database_url: str = DB_URL):
        self._engine = create_engine(
            database_url,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
        )
        self._session_factory = sessionmaker(bind=self._engine)

    # ---- 连接 ----

    def init_tables(self):
        """初始化所有数据表"""
        Base.metadata.create_all(self._engine)
        print("[Postgres] 数据表初始化完成")

    def get_session(self) -> Session:
        """获取新的数据库会话"""
        return self._session_factory()

    # ---- 文档操作 ----

    def add_document(
        self,
        file_name: str,
        file_type: str,
        file_size: int = 0,
        chunk_count: int = 0,
        total_chars: int = 0,
    ) -> Document:
        """添加文档记录"""
        with self.get_session() as session:
            # 检查是否已存在
            existing = (
                session.query(Document)
                .filter_by(file_name=file_name)
                .first()
            )
            if existing:
                # 更新已有记录
                existing.file_size = file_size
                existing.chunk_count = chunk_count
                existing.total_chars = total_chars
                existing.status = "indexed"
                existing.updated_at = datetime.utcnow()
                session.commit()
                return existing

            doc = Document(
                file_name=file_name,
                file_type=file_type,
                file_size=file_size,
                chunk_count=chunk_count,
                total_chars=total_chars,
                status="indexed",
            )
            session.add(doc)
            session.commit()
            session.refresh(doc)
            return doc

    def get_all_documents(self) -> list[Document]:
        """获取所有文档记录"""
        with self.get_session() as session:
            return session.query(Document).order_by(
                Document.created_at.desc()
            ).all()

    def get_document_by_name(self, file_name: str) -> Document | None:
        """根据文件名获取文档"""
        with self.get_session() as session:
            return (
                session.query(Document)
                .filter_by(file_name=file_name)
                .first()
            )

    def delete_document(self, file_name: str):
        """删除文档记录"""
        with self.get_session() as session:
            session.query(Document).filter_by(file_name=file_name).delete()
            session.commit()

    # ---- 对话历史 ----

    def add_chat(
        self,
        session_id: str,
        role: str,
        content: str,
        sources: str = "",
        latency_ms: float = 0.0,
    ):
        """添加对话记录"""
        with self.get_session() as sess:
            record = ChatHistory(
                session_id=session_id,
                role=role,
                content=content,
                sources=sources,
                latency_ms=latency_ms,
            )
            sess.add(record)
            sess.commit()

    def get_chat_history(
        self,
        session_id: str,
        limit: int = 20,
    ) -> list[ChatHistory]:
        """获取对话历史"""
        with self.get_session() as session:
            return (
                session.query(ChatHistory)
                .filter_by(session_id=session_id)
                .order_by(ChatHistory.created_at.desc())
                .limit(limit)
                .all()
            )

    # ---- 评测记录 ----

    def add_evaluation(
        self,
        question: str,
        expected_answer: str = "",
        generated_answer: str = "",
        hit_rate: float = 0.0,
        mrr: float = 0.0,
        latency_ms: float = 0.0,
    ):
        """添加评测记录"""
        with self.get_session() as session:
            record = EvaluationRecord(
                question=question,
                expected_answer=expected_answer,
                generated_answer=generated_answer,
                hit_rate=hit_rate,
                mrr=mrr,
                latency_ms=latency_ms,
            )
            session.add(record)
            session.commit()

    def get_evaluation_stats(self) -> dict:
        """获取评测统计数据"""
        with self.get_session() as session:
            records = session.query(EvaluationRecord).all()
            if not records:
                return {
                    "total": 0,
                    "avg_hit_rate": 0,
                    "avg_mrr": 0,
                    "avg_latency_ms": 0,
                }
            total = len(records)
            avg_hit = sum(r.hit_rate for r in records) / total
            avg_mrr = sum(r.mrr for r in records) / total
            avg_lat = sum(r.latency_ms for r in records) / total
            return {
                "total": total,
                "avg_hit_rate": round(avg_hit, 4),
                "avg_mrr": round(avg_mrr, 4),
                "avg_latency_ms": round(avg_lat, 1),
            }


# 全局单例
_pg_instance: PostgresStore | None = None


def get_postgres_store() -> PostgresStore:
    """获取全局 PostgresStore 实例"""
    global _pg_instance
    if _pg_instance is None:
        _pg_instance = PostgresStore()
        _pg_instance.init_tables()
    return _pg_instance
