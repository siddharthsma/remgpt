"""
FIFO queue block for LLM Context management.
"""

from typing import List, Optional
from collections import deque
import logging
from ..base_block import BaseBlock
from ...core.types import Message


class FIFOQueueBlock(BaseBlock):
    """Block containing a FIFO queue of messages."""
    
    def __init__(self, max_size: Optional[int] = None, name: str = "fifo_queue", logger: Optional[logging.Logger] = None):
        """Initialize FIFO queue with optional max size."""
        super().__init__(name, is_read_only=False, logger=logger)
        self.messages = deque(maxlen=max_size)
        self.max_size = max_size
    
    def to_messages(self) -> List[Message]:
        """Convert queue to list of messages."""
        return list(self.messages)
    
    def get_messages(self) -> List[Message]:
        """Get all messages as a list (alias for to_messages for compatibility)."""
        return list(self.messages)
    
    def add_message(self, message: Message):
        """Add a message to the queue."""
        self.messages.append(message)
    
    def add_messages(self, messages: List[Message]):
        """Add multiple messages to the queue."""
        for message in messages:
            self.messages.append(message)
    
    def clear(self):
        """Clear all messages from the queue."""
        self.messages.clear()
    
    def clear_messages(self):
        """Clear all messages from the queue (alias for clear for compatibility)."""
        self.messages.clear()
    
    def get_size(self) -> int:
        """Get current number of messages in queue."""
        return len(self.messages) 