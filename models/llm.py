"""
LLM 模型封装
============
使用 DeepSeek API (OpenAI 兼容接口) 提供对话和生成能力。
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL


class DeepSeekLLM:
    """DeepSeek LLM 封装，提供统一调用接口"""

    def __init__(self, temperature: float = 0.1, max_tokens: int = 2048):
        self._llm = ChatOpenAI(
            model=DEEPSEEK_MODEL,
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    @property
    def llm(self) -> ChatOpenAI:
        return self._llm

    def chat(
        self,
        messages: list[BaseMessage],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """同步对话"""
        # 支持动态覆盖 temperature 和 max_tokens
        if temperature is not None or max_tokens is not None:
            llm = ChatOpenAI(
                model=DEEPSEEK_MODEL,
                api_key=DEEPSEEK_API_KEY,
                base_url=DEEPSEEK_BASE_URL,
                temperature=temperature if temperature is not None else 0.1,
                max_tokens=max_tokens if max_tokens is not None else 2048,
            )
            response = llm.invoke(messages)
        else:
            response = self._llm.invoke(messages)
        return response.content

    async def achat(
        self,
        messages: list[BaseMessage],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """异步对话"""
        if temperature is not None or max_tokens is not None:
            llm = ChatOpenAI(
                model=DEEPSEEK_MODEL,
                api_key=DEEPSEEK_API_KEY,
                base_url=DEEPSEEK_BASE_URL,
                temperature=temperature if temperature is not None else 0.1,
                max_tokens=max_tokens if max_tokens is not None else 2048,
            )
            response = await llm.ainvoke(messages)
        else:
            response = await self._llm.ainvoke(messages)
        return response.content

    def generate_with_template(
        self,
        system_prompt: str,
        user_message: str,
        **kwargs,
    ) -> str:
        """使用系统提示词和用户消息模板生成回复"""
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message),
        ]
        return self.chat(messages, **kwargs)

    def test_connection(self) -> tuple[bool, str]:
        """测试 DeepSeek API 连通性"""
        try:
            response = self.chat([HumanMessage(content="Hi")])
            if response:
                return True, f"连接成功，模型: {DEEPSEEK_MODEL}"
            return False, "返回为空"
        except Exception as e:
            return False, f"连接失败: {str(e)}"


# 全局单例
_llm_instance: DeepSeekLLM | None = None


def get_llm(
    temperature: float = 0.1,
    max_tokens: int = 2048,
) -> DeepSeekLLM:
    """获取全局 LLM 实例"""
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = DeepSeekLLM(
            temperature=temperature,
            max_tokens=max_tokens,
        )
    return _llm_instance
