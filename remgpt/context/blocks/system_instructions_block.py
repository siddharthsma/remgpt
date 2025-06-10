"""
System instructions block for context management.
"""

from typing import List, Optional
import logging
from ..base_block import BaseBlock
from ...core.types import Message, SystemMessage


class SystemInstructionsBlock(BaseBlock):
    """Block containing system instructions (read-only)."""
    
    def __init__(self, instructions: str, name: str = "system_instructions", logger: Optional[logging.Logger] = None):
        """Initialize with system instructions."""
        super().__init__(name, is_read_only=True, logger=logger)
        self.instructions = instructions
    
    def to_messages(self) -> List[Message]:
        """Convert to system message."""
        if not self.instructions.strip():
            return []
        return [SystemMessage(content=self.instructions)] 