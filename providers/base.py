import asyncio
import importlib
import logging
import pkgutil
from abc import ABC, abstractmethod
from typing import Dict, List, TYPE_CHECKING, Type, Any, Tuple

from core.chat_history import ChatHistoryMessage, LLMToolCall, LLMResponse, ChatHistoryController
from core.config import Config
from core.discord_messages import DiscordMessage
from providers.utils import mcp_client_integrations

if TYPE_CHECKING:
    from providers.utils.mcp_client_integrations.base import MCPIntegration


class BaseLLM(ABC):

    def __init__(self):
        self.chats: Dict[str, ChatHistoryController] = {}
        self.mcp_client_integration_module: Type[MCPIntegration] = self.load_mcp_integration_class()


    @abstractmethod
    async def call(self, history: List[ChatHistoryMessage], instructions: ChatHistoryMessage, queue: asyncio.Queue[DiscordMessage | None], channel: str, use_help_bot=False):
        pass


    @abstractmethod
    async def generate(self, chat: ChatHistoryController, model_name: str | None = None, temperature: float | None = None, timeout: float | None = None, tools: List[Dict] | None = None) -> LLMResponse:
        pass


    @classmethod
    def load_mcp_integration_class(cls):

        class_name = Config.MCP_INTEGRATION_CLASS

        for _, module_name, _ in pkgutil.iter_modules(mcp_client_integrations.__path__):
            logging.debug(module_name)
            module = importlib.import_module(f"providers.utils.mcp_client_integrations.{module_name}")
            logging.debug(module)
            if hasattr(module, class_name):
                integration_cls = getattr(module, class_name)
                return integration_cls

        from providers.utils.mcp_client_integrations.base import MCPIntegration
        return MCPIntegration


    @classmethod
    @abstractmethod
    def format_history_entry(cls, entry: ChatHistoryMessage) -> Dict[str, Any]:
        pass

    @classmethod
    @abstractmethod
    def add_assistant_message(cls, chat: ChatHistoryController, message: str) -> None:
        pass

    @classmethod
    @abstractmethod
    def add_error_message(cls, chat: ChatHistoryController, message: str) -> None:
        pass

    @classmethod
    @abstractmethod
    def add_tool_call_message(cls, chat: ChatHistoryController, tool_calls: List[LLMToolCall]) -> None:
        pass

    @classmethod
    @abstractmethod
    def add_tool_call_results_message(cls, chat: ChatHistoryController, tool_responses: List[Tuple[LLMToolCall, str]]) -> None:
        pass

    @classmethod
    @abstractmethod
    def extract_custom_tool_call(cls, text: str) -> LLMToolCall:
        pass