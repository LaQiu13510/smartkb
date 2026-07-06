"""RAG context management for SmartKB."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List


@dataclass
class RAGContextItem:
    content: str
    file_name: str
    source_page: int = 0
    score: float = 0.0


class RAGContextManager:
    """Build deduplicated, source-aware context for generation."""

    SECRET_PATTERNS = [
        re.compile(r"sk-[A-Za-z0-9_\-]{8,}"),
        re.compile(r"AIza[A-Za-z0-9_\-]{8,}"),
        re.compile(r"postgresql://[^\s]+:[^\s]+@"),
    ]

    def __init__(self, max_chars: int = 4000, dedupe_prefix: int = 100):
        self.max_chars = max_chars
        self.dedupe_prefix = dedupe_prefix

    def build(self, results: List[dict], max_chars: int | None = None) -> str:
        budget = max_chars or self.max_chars
        context_parts = []
        total_chars = 0
        seen = set()

        for index, result in enumerate(results, start=1):
            item = self._to_item(result)
            if not item.content:
                continue

            content_key = item.content[: self.dedupe_prefix]
            if content_key in seen:
                continue
            seen.add(content_key)

            content = self._redact(item.content)
            if total_chars + len(content) > budget:
                remaining = budget - total_chars
                if remaining > 100:
                    content = content[:remaining] + "..."
                else:
                    break

            page_info = f" (第{item.source_page}页)" if item.source_page > 0 else ""
            context_parts.append(
                f"[文档片段 {index}]\n"
                f"来源: {item.file_name}{page_info}\n"
                f"相关度: {item.score:.4f}\n"
                f"内容:\n{content}\n"
            )
            total_chars += len(content)

        return "\n---\n".join(context_parts)

    def _redact(self, text: str) -> str:
        redacted = text
        for pattern in self.SECRET_PATTERNS:
            redacted = pattern.sub("[REDACTED_SECRET]", redacted)
        return redacted

    def _to_item(self, result: dict) -> RAGContextItem:
        return RAGContextItem(
            content=result.get("content", "").strip(),
            file_name=result.get("file_name", "未知文件"),
            source_page=result.get("source_page", 0),
            score=result.get("hybrid_score", result.get("score", 0.0)),
        )


_context_manager: RAGContextManager | None = None


def get_rag_context_manager() -> RAGContextManager:
    global _context_manager
    if _context_manager is None:
        _context_manager = RAGContextManager()
    return _context_manager
