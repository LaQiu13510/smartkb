"""SmartKB chat、list、retrieve 路由离线评测。"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from langchain_core.messages import HumanMessage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.graph import router_node


CASES = [
    {"task": "你好", "route": "chat"},
    {"task": "您好！", "route": "chat"},
    {"task": "谢谢", "route": "chat"},
    {"task": "再见", "route": "chat"},
    {"task": "hello", "route": "chat"},
    {"task": "你是谁？", "route": "chat"},
    {"task": "你能做什么？", "route": "chat"},
    {"task": "介绍一下你自己", "route": "chat"},
    {"task": "知识库里有哪些文档？", "route": "list"},
    {"task": "请列出所有文件", "route": "list"},
    {"task": "展示文档列表", "route": "list"},
    {"task": "目前有多少文档？", "route": "list"},
    {"task": "上传了什么文件？", "route": "list"},
    {"task": "库里有什么资料文件？", "route": "list"},
    {"task": "what documents are uploaded?", "route": "list"},
    {"task": "list the files", "route": "list"},
    {"task": "show documents", "route": "list"},
    {"task": "系统里都存了啥？", "route": "list"},
    {"task": "RAG 的核心组件有哪些？", "route": "retrieve"},
    {"task": "混合检索和 RRF 是什么？", "route": "retrieve"},
    {"task": "帮我检索 PostgreSQL 的部署要求", "route": "retrieve"},
    {"task": "帮助我查找 Redis 缓存失效策略", "route": "retrieve"},
    {"task": "help me find the Milvus configuration", "route": "retrieve"},
    {"task": "文档里如何介绍上下文管理？", "route": "retrieve"},
    {"task": "FastAPI 流式接口如何实现？", "route": "retrieve"},
    {"task": "公司的报销制度是什么？", "route": "retrieve"},
    {"task": "总结项目开发规范", "route": "retrieve"},
    {"task": "查询 BM25 的计算方式", "route": "retrieve"},
    {"task": "怎么部署这个知识库？", "route": "retrieve"},
    {"task": "敏感信息应该如何脱敏？", "route": "retrieve"},
]


def macro_f1(rows: list[dict]) -> tuple[float, dict[str, dict[str, float]]]:
    labels = sorted({row["expected_route"] for row in rows})
    metrics = {}
    for label in labels:
        true_positive = sum(row["expected_route"] == label and row["actual_route"] == label for row in rows)
        false_positive = sum(row["expected_route"] != label and row["actual_route"] == label for row in rows)
        false_negative = sum(row["expected_route"] == label and row["actual_route"] != label for row in rows)
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
        metrics[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }
    return round(sum(item["f1"] for item in metrics.values()) / len(metrics), 4), metrics


def run_eval() -> dict:
    details = []
    confusion = defaultdict(lambda: defaultdict(int))
    for case in CASES:
        result = router_node({"messages": [HumanMessage(content=case["task"])]})
        actual = result.get("route", "")
        details.append({
            "task": case["task"],
            "expected_route": case["route"],
            "actual_route": actual,
            "passed": actual == case["route"],
        })
        confusion[case["route"]][actual] += 1

    passed = sum(row["passed"] for row in details)
    f1, per_route = macro_f1(details)
    return {
        "project": "smartkb-rag",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "dataset": {"cases": len(details), "routes": 3},
        "metrics": {
            "route_accuracy": round(passed / len(details), 4),
            "route_macro_f1": f1,
        },
        "per_route": per_route,
        "confusion_matrix": {
            expected: dict(predicted)
            for expected, predicted in confusion.items()
        },
        "details": details,
    }


def main():
    print(json.dumps(run_eval(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
