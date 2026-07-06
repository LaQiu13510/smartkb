"""
评测数据集
==========
用于 RAG 系统评测的测试问题集。
这些问题用于验证检索、生成和评测流程是否稳定。
"""

# 通用评测问题 (当知识库为空时用作 base 测试)
BASE_QUESTIONS = [
    {
        "question": "什么是RAG技术？",
        "keywords": ["检索", "生成", "增强", "retrieval"],
    },
    {
        "question": "向量数据库的作用是什么？",
        "keywords": ["向量", "嵌入", "相似度", "检索"],
    },
]

# 知识库评测问题模板 (需要根据上传的实际文档定制)
# 当用户上传文档后，可以根据文档内容构造此类问题
CUSTOM_QUESTION_TEMPLATES = [
    {
        "question": "根据文档，{topic}的主要特点有哪些？",
        "keywords": [],
        "description": "需要替换 {topic} 为实际文档主题",
    },
    {
        "question": "文档中提到了哪些关于{keyword}的内容？",
        "keywords": [],
        "description": "需要替换 {keyword} 为文档中的关键术语",
    },
    {
        "question": "请总结文档的核心观点",
        "keywords": [],
        "description": "测试概括能力",
    },
]


def generate_test_questions(
    doc_topics: list[str],
    doc_keywords: list[str],
) -> list[dict]:
    """
    根据文档内容自动生成测试问题

    Args:
        doc_topics: 文档主题列表
        doc_keywords: 文档关键词列表

    Returns:
        定制化的测试问题列表
    """
    questions = []

    for topic in doc_topics:
        questions.append({
            "question": f"请详细介绍{topic}",
            "keywords": [topic],
        })
        questions.append({
            "question": f"{topic}有哪些关键特性或要点？",
            "keywords": [topic],
        })

    if doc_keywords:
        kw_str = "、".join(doc_keywords[:5])
        questions.append({
            "question": f"关于{kw_str}，文档中是怎么说的？",
            "keywords": doc_keywords[:5],
        })

    questions.append({
        "question": "请用几句话总结文档的主要内容",
        "keywords": [],
    })

    return questions

