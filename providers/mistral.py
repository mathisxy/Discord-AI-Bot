import asyncio
import json
from typing import List, Dict, Any
from mistralai import Mistral

from core.config import Config
from providers.default import DefaultLLM, LLMResponse, LLMToolCall
from providers.utils.chat import LLMChat

client = Mistral(api_key=Config.MISTRAL_API_KEY)

class MistralLLM(DefaultLLM):


    async def generate(self, chat: LLMChat, model_name: str | None = None, temperature: float | None = None,
                       timeout: float | None = None, tools: List[Dict] | None = None) -> LLMResponse:

        model_name = model_name if model_name else Config.MISTRAL_MODEL

        # async with RunContext(model=Config.MISTRAL_MODEL) as run_ctx:
        #     run_results = await client.beta.conversations.run_async(
        #         run_ctx=run_ctx,
        #         inputs=chat.history[1:],
        #         #description="Manuel, ein Discord Bot",
        #         instructions=chat.system_entry
        #     )

        response = await client.chat.complete_async(
            model=model_name,
            messages=chat.history,
            temperature=temperature,
            tools=tools,
        )

        message = response.choices[0].message

        tool_calls = []
        if message.tool_calls:
            tool_calls = [LLMToolCall(name=t.function.name, arguments=json.loads(t.function.arguments)) for t in message.tool_calls] if message.tool_calls else []

        return LLMResponse(message.content, tool_calls)


# async def call_ai(history: List[Dict], instructions: str) -> str:
#     mcp_client = MCPClientSSE(sse_params=SSEServerParams(url=mcp_server_url, timeout=100))
#
#     async with RunContext(model=model) as run_ctx:
#         await run_ctx.register_mcp_client(mcp_client=mcp_client)
#         run_results = await client.beta.conversations.run_async(
#             run_ctx=run_ctx,
#             inputs=history,
#             description="Manuel, ein Discord Bot",
#             instructions=instructions
#         )
#         return run_results.output_as_text

    @staticmethod
    def construct_tool_call_message(tool_calls: List[LLMToolCall]) -> Dict[str, Any]:

        return {"role": "system", "tool_calls": [
            {"id": t.name, "arguments": t.arguments} for t in tool_calls
        ]}

    @staticmethod
    def construct_tool_call_results(name: str, content: str) -> Dict[str, str]:

        return {
            "role": "tool",
            "tool_call_id": name,
            "content": f"#{content}"
        }