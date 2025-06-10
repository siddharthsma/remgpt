"""
Context management package for RemGPT.
"""

from .token_counter import TokenCounter
from .base_block import BaseBlock
from .llm_context import LLMContext
from .llm_context_manager import LLMContextManager
from .factory import create_context_manager, create_context_with_config
from .context_tools import ContextManagementToolFactory
from .blocks import (
    SystemInstructionsBlock,
    MemoryInstructionsBlock,
    ToolsDefinitionsBlock,
    WorkingContextBlock,
    FIFOQueueBlock
)

__all__ = [
    "TokenCounter",
    "BaseBlock",
    "LLMContext",
    "LLMContextManager",
    "create_context_manager",
    "create_context_with_config",
    "ContextManagementToolFactory",
    "SystemInstructionsBlock",
    "MemoryInstructionsBlock",
    "ToolsDefinitionsBlock", 
    "WorkingContextBlock",
    "FIFOQueueBlock"
] 