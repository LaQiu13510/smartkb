"""
文本分割器
==========
实现递归字符分割与语义感知分块。
默认使用语义感知分块：优先按标题、段落和相邻片段相似度切分，并保留递归分割作为 fallback。

分块策略演进 (记录在项目报告中):
  版本 1.0: 固定大小分块 (Fixed-size chunking) → chunk overlap 无法保证语义完整
  版本 2.0: 递归字符分割 (Recursive split) → 按段落、句子、词逐级切分
  版本 3.0: 语义感知分块 (Semantic-aware chunking) → 标题/段落边界 + 相邻片段相似度检测 ← 当前方案
"""

import math
import re
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter

from models.embedding import get_embedding_model
from rag.loader import Document


class TextSplitter:
    """递归语义文本分割器"""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: list[str] | None = None,
        strategy: str = "semantic",
        semantic_threshold: float = 0.55,
        semantic_embeddings: bool = True,
        lexical_fallback_threshold: float = 0.12,
    ):
        """
        Args:
            chunk_size: 每个分块的最大字符数
            chunk_overlap: 相邻分块之间的重叠字符数
            separators: 分隔符优先级列表
            strategy: semantic 或 recursive
            semantic_threshold: 相邻片段 Embedding 余弦相似度低于该值时倾向切分
            semantic_embeddings: 是否启用 Embedding 语义边界检测
            lexical_fallback_threshold: Embedding 不可用时的词汇相似度阈值
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = strategy
        self.semantic_threshold = semantic_threshold
        self.semantic_embeddings = semantic_embeddings
        self.lexical_fallback_threshold = lexical_fallback_threshold
        self._semantic_backend = "not_used"

        if separators is None:
            # 中文友好的分隔符：段落 → 句子 → 短句 → 词 → 字符
            separators = [
                "\n\n",    # 段落
                "\n",      # 换行
                "。",      # 中文句号
                "！",      # 中文感叹号
                "？",      # 中文问号
                "；",      # 中文分号
                "，",      # 中文逗号
                ".",       # 英文句号
                "!",       # 英文感叹号
                "?",       # 英文问号
                ";",       # 英文分号
                ",",       # 英文逗号
                " ",       # 空格 (英文分词边界)
                "",        # 字符级分割 (最后手段)
            ]

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=len,
            is_separator_regex=False,
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """将文档列表切分为文本块列表"""
        all_chunks = []
        for doc in documents:
            chunks = self.split_single_document(doc)
            all_chunks.extend(chunks)
        return all_chunks

    def split_single_document(self, document: Document) -> List[Document]:
        """切分单个文档"""
        if self.strategy == "semantic":
            texts = self._semantic_split_text(document.page_content)
        else:
            texts = self._splitter.split_text(document.page_content)

        chunks = []
        for i, text in enumerate(texts):
            chunk_meta = {
                **document.metadata,
                "chunk_index": i,
                "chunk_count": len(texts),
                "chunk_id": f"{document.metadata.get('file_hash', 'unk')}_chunk_{i}",
            }
            chunks.append(Document(content=text, metadata=chunk_meta))

        return chunks

    @property
    def config(self) -> dict:
        """返回分块配置信息"""
        return {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "splitter_type": "SemanticAwareTextSplitter" if self.strategy == "semantic" else "RecursiveCharacterTextSplitter",
            "semantic_threshold": self.semantic_threshold,
            "semantic_embeddings": self.semantic_embeddings,
            "semantic_backend": self._semantic_backend,
        }

    def _semantic_split_text(self, text: str) -> List[str]:
        """语义感知分块：标题/段落边界 + 相邻片段相似度检测。"""
        units = self._split_semantic_units(text)
        if not units:
            return []
        if len(units) == 1 and len(units[0]) > self.chunk_size:
            return self._splitter.split_text(units[0])

        chunks = []
        current = ""
        previous_unit = ""
        similarities = self._adjacent_similarities(units)

        for index, unit in enumerate(units):
            candidate = f"{current}\n\n{unit}".strip() if current else unit
            similarity = similarities[index - 1] if index > 0 else None
            boundary = self._should_start_new_chunk(
                current,
                previous_unit,
                unit,
                similarity,
            )
            if current and (len(candidate) > self.chunk_size or boundary):
                chunks.extend(self._finalize_chunk(current))
                current = unit
            else:
                current = candidate
            previous_unit = unit

        if current:
            chunks.extend(self._finalize_chunk(current))
        return chunks

    def _split_semantic_units(self, text: str) -> List[str]:
        blocks = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
        if len(blocks) > 1:
            return blocks
        sentence_units = re.split(r"(?<=[。！？.!?])\s*", text)
        return [unit.strip() for unit in sentence_units if unit.strip()]

    def _should_start_new_chunk(
        self,
        current: str,
        previous_unit: str,
        unit: str,
        similarity: float | None = None,
    ) -> bool:
        if not current:
            return False
        if self._looks_like_heading(unit):
            return True
        if len(current) < self.chunk_size * 0.45:
            return False
        if similarity is not None:
            return similarity < self.semantic_threshold
        return self._jaccard(previous_unit, unit) < self.lexical_fallback_threshold

    def _adjacent_similarities(self, units: List[str]) -> List[float | None]:
        if len(units) < 2:
            return []
        if self.semantic_embeddings:
            try:
                model = get_embedding_model()
                embeddings = model.embed_documents(units)
                if len(embeddings) == len(units):
                    self._semantic_backend = getattr(model, "model_name", "embedding")
                    return [
                        self._cosine(embeddings[index], embeddings[index + 1])
                        for index in range(len(embeddings) - 1)
                    ]
            except Exception:
                self._semantic_backend = "lexical_fallback"

        self._semantic_backend = "lexical_fallback"
        return [None] * (len(units) - 1)

    def _cosine(self, left: List[float], right: List[float]) -> float:
        dot = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if not left_norm or not right_norm:
            return 0.0
        return dot / (left_norm * right_norm)

    def _looks_like_heading(self, text: str) -> bool:
        stripped = text.strip()
        return stripped.startswith("#") or bool(re.match(r"^(\d+[\.\)]|[一二三四五六七八九十]+[、.])\s*\S+", stripped))

    def _jaccard(self, left: str, right: str) -> float:
        left_tokens = self._tokens(left)
        right_tokens = self._tokens(right)
        if not left_tokens or not right_tokens:
            return 0.0
        return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)

    def _tokens(self, text: str) -> set[str]:
        return {
            token.lower()
            for token in re.findall(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+", text)
            if token.strip()
        }

    def _finalize_chunk(self, text: str) -> List[str]:
        if len(text) <= self.chunk_size:
            return [text]
        return self._splitter.split_text(text)


# 工厂函数
def create_splitter(
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    strategy: str = "semantic",
    semantic_threshold: float = 0.55,
    semantic_embeddings: bool = True,
) -> TextSplitter:
    return TextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        strategy=strategy,
        semantic_threshold=semantic_threshold,
        semantic_embeddings=semantic_embeddings,
    )
