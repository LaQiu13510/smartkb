"""
文本分割器
==========
实现递归字符分割 (RecursiveCharacterTextSplitter)。
这是 RAG 系统中最常用的分块策略，按自然语义边界分割文本。

分块策略演进 (记录在项目报告中):
  版本 1.0: 固定大小分块 (Fixed-size chunking) → chunk overlap 无法保证语义完整
  版本 2.0: 递归字符分割 (Recursive split) → 按段落、句子、词逐级切分 ← 当前方案
  版本 3.0: (TODO) 语义分块 (Semantic chunking) → 基于 Embedding 相似度边界检测
"""

from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.loader import Document


class TextSplitter:
    """递归语义文本分割器"""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: list[str] | None = None,
    ):
        """
        Args:
            chunk_size: 每个分块的最大字符数
            chunk_overlap: 相邻分块之间的重叠字符数
            separators: 分隔符优先级列表
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

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
        # 使用 LangChain splitter 切分文本
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
            "splitter_type": "RecursiveCharacterTextSplitter",
        }


# 工厂函数
def create_splitter(
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> TextSplitter:
    return TextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
