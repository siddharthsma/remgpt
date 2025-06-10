"""
Base abstract class for LLM Context blocks.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
import logging
from ..core.types import Message
from .token_counter import TokenCounter


class BaseBlock(ABC):
    """Abstract base class for all context blocks."""
    
    def __init__(self, name: str, is_read_only: bool = False, logger: Optional[logging.Logger] = None):
        """Initialize block with a name and read-only flag."""
        self.name = name
        self.is_read_only = is_read_only
        self._token_counter: Optional[TokenCounter] = None
        self.logger = logger or logging.getLogger(f"{__name__}.{name}")
    
    def set_token_counter(self, token_counter: TokenCounter):
        """Set the token counter for this block."""
        self._token_counter = token_counter
    
    @abstractmethod
    def to_messages(self) -> List[Message]:
        """Convert this block to a list of messages."""
        pass
    
    def get_token_count(self) -> int:
        """Get the token count for this block."""
        if not self._token_counter:
            raise ValueError("Token counter not set. Call set_token_counter first.")
        return self._token_counter.count_messages_tokens(self.to_messages()) 