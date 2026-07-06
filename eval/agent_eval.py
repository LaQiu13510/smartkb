"""Offline agent routing evaluation for SmartKB."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

from langchain_core.messages import HumanMessage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.graph import router_node


CASES = [
    {"task": "你好，你能做什么？", "expected_route": "chat"},
    {"task": "知识库里有哪些文档？", "expected_route": "list"},
    {"task": "what documents are uploaded?", "expected_route": "list"},
    {"task": "RAG 的核心组件有哪些？", "expected_route": "retrieve"},
    {"task": "混合检索和 RRF 是什么？", "expected_route": "retrieve"},
]


def run_eval() -> dict:
    details = []
    for case in CASES:
        state = {"messages": [HumanMessage(content=case["task"])]}
        result = router_node(state)
        actual = result.get("route")
        details.append(
            {
                "task": case["task"],
                "expected_route": case["expected_route"],
                "actual_route": actual,
                "passed": actual == case["expected_route"],
            }
        )

    passed = sum(1 for item in details if item["passed"])
    total = len(details)
    return {
        "project": "smartkb-rag",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "total": total,
        "passed": passed,
        "route_accuracy": round(passed / total, 4) if total else 0,
        "details": details,
    }


def main():
    report = run_eval()
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
