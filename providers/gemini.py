import json
from typing import List, Dict

from openai import AsyncOpenAI

from core.config import Config
from providers.base import LLMResponse, LLMToolCall
from providers.default import DefaultLLM
from providers.utils.chat import LLMChat


class GeminiLLM(DefaultLLM):


    client = AsyncOpenAI(
        api_key=Config.GEMINI_API_KEY,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

    async def generate(self, chat: LLMChat, model_name: str | None = None, temperature: float | None = None,
                       timeout: float | None = None, tools: List[Dict] | None = None) -> LLMResponse:

        model_name = model_name or Config.GEMINI_MODEL

        completion = await self.client.chat.completions.create(
            model=model_name,
            messages=chat.history,
            temperature=temperature,
            # tools=tools
        )

        message = completion.choices[0].message

        tool_calls = []
        if getattr(message, "tool_calls", None):
            tool_calls = [
                LLMToolCall(id=t.id, name=t.function.name, arguments=json.loads(t.function.arguments))
                for t in message.tool_calls
            ]

        return LLMResponse(message.content, tool_calls)
