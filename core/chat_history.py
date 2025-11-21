from dataclasses import dataclass, field
from typing import Literal, Tuple, Dict, List
from pathlib import Path

@dataclass(kw_only=True)
class LLMToolCall:
    """Supports only calls of type function"""
    id: str
    name: str
    arguments: Dict

@dataclass
class LLMResponse:
    text: str
    tool_calls: List[LLMToolCall] = field(default_factory=list)

@dataclass
class ChatHistoryFile:

    name: str
    mime_type: str

@dataclass
class ChatHistoryFileSaved(ChatHistoryFile):

    save_path: Path

@dataclass
class ChatHistoryFileText(ChatHistoryFile):

    text_content: str

@dataclass(kw_only=True)
class ChatHistoryMessage:

    role: Literal["system", "user", "assistant", "tool"]
    content: str|None = None
    files: [ChatHistoryFile] = field(default_factory=list)
    tool_calls: [LLMToolCall] = field(default_factory=list)
    tool_response: Tuple[LLMToolCall, str] = field(default_factory=list)

    is_temporary: bool = False
