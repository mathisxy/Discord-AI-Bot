import logging
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

    full_path: Path
    """Full save path including the filename"""

    def __post_init__(self):
        self.validate_path()

    def validate_path(self):
        """Checks if save_path is a File"""

        if not self.full_path.parts:
            raise ValueError(f"Invalid Filepath: '{self.full_path}'")

        if not self.full_path.name:
            raise ValueError(f"Missing Filename in Filepath: '{self.full_path}'")


    async def save(self, file_bytes) -> None:

        with open(self.full_path, "wb") as f:
            f.write(file_bytes)

    def __del__(self):
        if self.full_path.exists():
            try:
                self.full_path.unlink(missing_ok=True)
                logging.info(f"Deleted '{self.full_path}'")
            except Exception as e:
                logging.exception(f"Deletion of '{self.full_path}' failed: {e}")


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
