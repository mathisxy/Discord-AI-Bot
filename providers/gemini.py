import base64
import logging
from typing import List, Dict, Any

from google import genai
from google.genai import types

from core.chat_history import ChatHistoryFileSaved, ChatHistoryMessage, ChatHistoryFile, ChatHistoryFileText, \
    ChatHistoryController
from core.config import Config
from providers.base import LLMResponse, LLMToolCall
from providers.default import DefaultLLM


class GeminiLLM(DefaultLLM):


    client = genai.Client(api_key=Config.GEMINI_API_KEY)


    async def generate(self, chat: ChatHistoryController, model_name: str | None = None, temperature: float | None = None,
                       timeout: float | None = None, tools: List[Dict] | None = None) -> LLMResponse:

        model_name = model_name or Config.GEMINI_MODEL
        messages = [self.format_history_entry(msg) for msg in chat.history]
        system_instruction = self.format_history_entry(chat.system_entry) if chat.history else None
        if system_instruction:
            messages = messages[1:]


        config = types.GenerateContentConfig(
            tools=tools,
            **({"system_instruction": system_instruction} if system_instruction is not None else {}),
            **({"temperature": temperature} if temperature is not None else {}),
        )

        logging.info(config)

        response = await self.client.aio.models.generate_content(
            model=model_name,
            contents=messages,
            config=config,
        )

        message = response.text

        tool_calls = []

        if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:

            for part in response.candidates[0].content.parts:

                if call := part.function_call:
                    tool_calls.append(
                        LLMToolCall(id="", name=call.name, arguments=call.args)
                    )

        return LLMResponse(message, tool_calls)


    @classmethod
    def format_history_entry(cls, entry: ChatHistoryMessage) -> Dict[str, Any]:

        parts = []

        if entry.content:
            parts.append(types.Part.from_text(
                text= entry.content
            ))

        for file in entry.files:
            if isinstance(file, ChatHistoryFile):
                if isinstance(file, ChatHistoryFileText):
                    parts.append(types.Part.from_text(
                        text= f"<#File name=\"{file.name}\">{file.text_content}</File>"
                    ))
                elif isinstance(file, ChatHistoryFileSaved) and file.mime_type in Config.GEMINI_VISION_MODEL_TYPES:
                    logging.info(f"Using vision for {file}")
                    with open(file.full_path, "rb") as f:
                        # b64 = base64.b64encode(f.read()).decode("utf-8")
                        data = f.read()

                    parts.append(types.Part.from_bytes(
                        data=data,
                        mime_type=file.mime_type,
                    ))
                else:
                    parts.append(types.Part.from_text(
                        text= f"<#File name=\"{file.name}\">",
                    ))

        for tool_call in entry.tool_calls:
            parts.append(types.Part.from_function_call(
                name=tool_call.name,
                args=tool_call.arguments,
            ))

        if entry.tool_response:
            tool_call, response = entry.tool_response
            parts.append(types.Part.from_function_response(
                name=tool_call.name,
                response={
                    "result": response,
                }
            ))

        role = entry.role
        if role == "assistant":
            role = "model"

        formatted_entry = {
            "role": role,
            "parts": parts,
        }

        logging.info(formatted_entry)

        return formatted_entry # TODO ROLES
