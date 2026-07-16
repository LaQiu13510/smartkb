"""SmartKB 的 RAG 上下文治理。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from config import RAG_CONTEXT_MAX_CHARS, RAG_CONTEXT_MAX_TOKENS


@dataclass
class RAGContextItem:
    content: str
    file_name: str
    source_page: int = 0
    score: float = 0.0


class RAGContextManager:
    """按来源、重复度、字符预算和 Token 预算组织检索证据。"""

    SECRET_PATTERNS = [
        re.compile(r"sk-[A-Za-z0-9_\-]{8,}"),
        re.compile(r"AIza[A-Za-z0-9_\-]{8,}"),
        re.compile(r"postgresql://[^\s]+:[^\s]+@"),
    ]

    def __init__(
        self,
        max_chars: int = RAG_CONTEXT_MAX_CHARS,
        max_tokens: int = RAG_CONTEXT_MAX_TOKENS,
        dedupe_prefix: int = 100,
        near_duplicate_threshold: float = 0.88,
    ):
        self.max_chars = max_chars
        self.max_tokens = max_tokens
        self.dedupe_prefix = dedupe_prefix
        self.near_duplicate_threshold = near_duplicate_threshold
        self._encoding = self._load_encoding()

    def build(
        self,
        results: List[dict],
        max_chars: int | None = None,
        max_tokens: int | None = None,
    ) -> str:
        char_budget = self.max_chars if max_chars is None else max_chars
        token_budget = self.max_tokens if max_tokens is None else max_tokens
        context_parts: list[str] = []
        total_chars = 0
        total_tokens = 0
        exact_keys: set[str] = set()
        token_sets: list[set[str]] = []

        for result in results:
            item = self._to_item(result)
            if not item.content:
                continue

            normalized = self._normalize(item.content)
            exact_key = normalized[: self.dedupe_prefix]
            item_tokens = self._dedupe_tokens(normalized)
            if exact_key in exact_keys or self._is_near_duplicate(item_tokens, token_sets):
                continue

            page_info = f" (第{item.source_page}页)" if item.source_page > 0 else ""
            header = (
                f"来源: {item.file_name}{page_info}\n"
                f"相关度: {item.score:.4f}\n"
                "内容:\n"
            )
            header_chars = len(header) + 16
            header_tokens = self._count_tokens(header) + 8
            remaining_chars = char_budget - total_chars - header_chars
            remaining_tokens = token_budget - total_tokens - header_tokens
            if remaining_chars <= 80 or remaining_tokens <= 32:
                break

            content = self._redact(item.content)
            content = self._truncate(content, remaining_chars, remaining_tokens)
            if not content.strip():
                break

            block = (
                f"[文档片段 {len(context_parts) + 1}]\n"
                f"{header}{content}\n"
            )
            context_parts.append(block)
            total_chars += len(block)
            total_tokens += self._count_tokens(block)
            exact_keys.add(exact_key)
            token_sets.append(item_tokens)

        return "\n---\n".join(context_parts)

    def stats(self, context: str) -> dict:
        return {
            "chars": len(context),
            "tokens": self._count_tokens(context),
            "char_budget": self.max_chars,
            "token_budget": self.max_tokens,
        }

    def _truncate(self, text: str, max_chars: int, max_tokens: int) -> str:
        value = text[:max_chars]
        if self._count_tokens(value) <= max_tokens:
            return value
        if self._encoding is not None:
            encoded = self._encoding.encode(value)
            return self._encoding.decode(encoded[:max_tokens]).rstrip() + "..."

        estimate_chars = max(1, max_tokens * 2)
        return value[:estimate_chars].rstrip() + "..."

    def _count_tokens(self, text: str) -> int:
        if self._encoding is not None:
            return len(self._encoding.encode(text))
        return max(1, len(text) // 2)

    def _load_encoding(self):
        try:
            import tiktoken

            return tiktoken.get_encoding("cl100k_base")
        except Exception:
            return None

    def _is_near_duplicate(
        self,
        candidate: set[str],
        existing: list[set[str]],
    ) -> bool:
        if not candidate:
            return False
        for other in existing:
            union = candidate | other
            if union and len(candidate & other) / len(union) >= self.near_duplicate_threshold:
                return True
        return False

    def _dedupe_tokens(self, text: str) -> set[str]:
        english = set(re.findall(r"[a-z0-9_]{2,}", text.lower()))
        cjk = re.findall(r"[\u4e00-\u9fff]", text)
        bigrams = {
            "".join(cjk[index : index + 2])
            for index in range(max(0, len(cjk) - 1))
        }
        return english | bigrams

    def _normalize(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip().lower()

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
            score=result.get("rerank_score", result.get("hybrid_score", result.get("score", 0.0))),
        )


_context_manager: RAGContextManager | None = None


def get_rag_context_manager() -> RAGContextManager:
    global _context_manager
    if _context_manager is None:
        _context_manager = RAGContextManager()
    return _context_manager
