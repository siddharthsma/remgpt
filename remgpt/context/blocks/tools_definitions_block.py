"""
Tools definitions block for LLM Context management.
"""

from typing import List, Dict, Any, Optional
import logging
import json
from ..base_block import BaseBlock
from ...types import Message, SystemMessage


class ToolsDefinitionsBlock(BaseBlock):
    """Block containing tool definitions (read-only)."""
    
    def __init__(self, tools: List[Dict[str, Any]], name: str = "tools_definitions", logger: Optional[logging.Logger] = None):
        """Initialize with tools definitions."""
        super().__init__(name, is_read_only=True, logger=logger)
        self.tools = tools or []
    
    def to_messages(self) -> List[Message]:
        """Convert to system message with tool definitions."""
        if not self.tools:
            return []
        
        tools_content = f"Available tools:\n{json.dumps(self.tools, indent=2)}"
        return [SystemMessage(content=tools_content)] 