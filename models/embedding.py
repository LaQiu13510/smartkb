"""
Embedding 模型封装
==================
按优先级自动选择最优 Embedding 后端:
  1. 智谱 AI (ZhipuAI) — embedding-2, 1024维, 国内直连
  2. 阿里云百炼 (Dashscope) — text-embedding-v2, 1536维
  3. HuggingFace 本地 — all-MiniLM-L6-v2, 384维

初始化时自动探测，用户无需关心后端选择。
"""

import os
from typing import List

from config import GEMINI_API_KEY


class EmbeddingModel:
    """Embedding 模型封装，自动选择最优后端"""

    def __init__(self):
        self._model = None
        self._backend = None
        self.dimension = 0
        self.model_name = "unknown"
        self._init_model()

    # ================================================================
    # 后端探测
    # ================================================================

    def _init_model(self):
        """依次尝试各后端，直到成功"""

        # ---- 1. 智谱 AI (ZhipuAI) ----
        if self._try_zhipu():
            return

        # ---- 2. 阿里云百炼 (Dashscope) ----
        if self._try_dashscope():
            return

        # ---- 3. HuggingFace 本地 ----
        if self._try_huggingface():
            return

        # ---- 4. Google (境外, 需要 VPN) ----
        if self._try_google():
            return

        raise RuntimeError(
            "无法初始化任何 Embedding 模型。请检查:\n"
            "  1. ZHIPUAI_API_KEY (智谱AI) 是否配置\n"
            "  2. DASHSCOPE_API_KEY (阿里云百炼) 是否配置\n"
            "  3. 或: pip install langchain-huggingface sentence-transformers"
        )

    def _try_zhipu(self) -> bool:
        """尝试智谱 AI Embedding"""
        api_key = os.getenv("ZHIPUAI_API_KEY", "")
        if not api_key:
            return False

        try:
            from langchain_openai import OpenAIEmbeddings

            model = OpenAIEmbeddings(
                model="embedding-2",
                api_key=api_key,
                base_url="https://open.bigmodel.cn/api/paas/v4/",
                dimensions=1024,
                timeout=15,
            )
            test_vec = model.embed_query("测试连接")
            if test_vec and len(test_vec) > 0:
                self._model = model
                self._backend = "zhipu"
                self.model_name = "embedding-2"
                self.dimension = len(test_vec)
                print(
                    f"[Embedding] 智谱AI 'embedding-2' 初始化成功, "
                    f"维度={self.dimension}"
                )
                return True
        except ImportError:
            print("[Embedding] langchain-openai 未安装")
        except Exception as e:
            print(f"[Embedding] 智谱AI 不可用: {str(e)[:150]}")

        return False

    def _try_dashscope(self) -> bool:
        """尝试阿里云百炼 Embedding"""
        api_key = os.getenv("DASHSCOPE_API_KEY", "")
        if not api_key:
            return False

        try:
            from langchain_community.embeddings import DashScopeEmbeddings

            model = DashScopeEmbeddings(
                model="text-embedding-v2",
                dashscope_api_key=api_key,
            )
            test_vec = model.embed_query("测试连接")
            if test_vec and len(test_vec) > 0:
                self._model = model
                self._backend = "dashscope"
                self.model_name = "text-embedding-v2"
                self.dimension = len(test_vec)
                print(
                    f"[Embedding] 阿里云百炼 'text-embedding-v2' 初始化成功, "
                    f"维度={self.dimension}"
                )
                return True
        except ImportError:
            print("[Embedding] langchain-community 未安装 (需要 DashScope)")
        except Exception as e:
            print(f"[Embedding] 阿里云百炼 不可用: {str(e)[:150]}")

        return False

    def _try_huggingface(self) -> bool:
        """尝试 HuggingFace 本地模型"""
        try:
            from langchain_huggingface import HuggingFaceEmbeddings

            model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
            test_vec = model.embed_query("test")
            if test_vec and len(test_vec) > 0:
                self._model = model
                self._backend = "huggingface"
                self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
                self.dimension = len(test_vec)
                print(
                    f"[Embedding] HuggingFace 'all-MiniLM-L6-v2' 初始化成功, "
                    f"维度={self.dimension}"
                )
                return True
        except ImportError:
            print("[Embedding] langchain-huggingface 未安装")
        except Exception as e:
            print(f"[Embedding] HuggingFace 不可用: {str(e)[:150]}")

        return False

    def _try_google(self) -> bool:
        """尝试 Google Embedding (需要 VPN)"""
        if not GEMINI_API_KEY or not GEMINI_API_KEY.startswith("AIza"):
            return False

        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings

            for model_name in ["models/text-embedding-004", "text-embedding-004"]:
                try:
                    model = GoogleGenerativeAIEmbeddings(
                        model=model_name,
                        google_api_key=GEMINI_API_KEY,
                        request_timeout=10,
                    )
                    test_vec = model.embed_query("test")
                    if test_vec and len(test_vec) > 0:
                        self._model = model
                        self._backend = "google"
                        self.model_name = model_name
                        self.dimension = len(test_vec)
                        print(
                            f"[Embedding] Google '{model_name}' 初始化成功, "
                            f"维度={self.dimension}"
                        )
                        return True
                except Exception as e:
                    print(f"[Embedding] Google '{model_name}': {str(e)[:120]}")
                    continue
        except ImportError:
            pass

        return False

    # ================================================================
    # 公共接口
    # ================================================================

    @property
    def backend(self) -> str:
        return self._backend or "unknown"

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._model.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._model.embed_query(text)

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        return await self._model.aembed_documents(texts)

    async def aembed_query(self, text: str) -> List[float]:
        return await self._model.aembed_query(text)

    def test_connection(self) -> tuple[bool, str]:
        try:
            embedding = self.embed_query("Hello, test")
            return True, (
                f"后端={self._backend}, 模型={self.model_name}, "
                f"维度={len(embedding)}"
            )
        except Exception as e:
            return False, str(e)[:200]


# 全局单例
_embedding_model: EmbeddingModel | None = None


def get_embedding_model() -> EmbeddingModel:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = EmbeddingModel()
    return _embedding_model
