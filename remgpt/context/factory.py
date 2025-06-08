"""
Factory functions for creating context managers.
"""

from typing import Optional, Dict, Any, List
import logging
from .llm_context_manager import LLMContextManager
from .llm_context import LLMContext
from .blocks import (
    SystemInstructionsBlock,
    MemoryInstructionsBlock,
    ToolsDefinitionsBlock,
    WorkingContextBlock,
    FIFOQueueBlock
)


def create_context_manager(
    max_tokens: int,
    system_instructions: str = "",
    memory_content: str = "",
    tools: Optional[List[Dict[str, Any]]] = None,
    model: str = "gpt-4",
    logger: Optional[logging.Logger] = None
) -> LLMContextManager:
    """
    Create an LLMContextManager with initial read-only content.
    
    Since SystemInstructionsBlock, MemoryInstructionsBlock, and ToolsDefinitionsBlock 
    are read-only, this factory function allows setting their initial content.
    
    Args:
        max_tokens: Maximum token limit for monitoring
        system_instructions: Initial system instructions
        memory_content: Initial memory content
        tools: Initial tool definitions list
        model: Model name for token counting
        logger: Optional logger instance
        
    Returns:
        Configured LLMContextManager
    """
    manager = LLMContextManager(max_tokens=max_tokens, model=model, logger=logger)
    
    # Replace the default read-only blocks with ones containing initial content
    if system_instructions:
        manager.context.system_instructions = SystemInstructionsBlock(system_instructions, logger=logger)
        manager.context.system_instructions.set_token_counter(manager.token_counter)
        manager.context.blocks["system_instructions"] = manager.context.system_instructions
    
    if memory_content:
        manager.context.memory_instructions = MemoryInstructionsBlock(memory_content, logger=logger)
        manager.context.memory_instructions.set_token_counter(manager.token_counter)
        manager.context.blocks["memory_instructions"] = manager.context.memory_instructions
    
    if tools:
        manager.context.tools_definitions = ToolsDefinitionsBlock(tools, logger=logger)
        manager.context.tools_definitions.set_token_counter(manager.token_counter)
        manager.context.blocks["tools_definitions"] = manager.context.tools_definitions
    
    return manager


def create_context_with_config(
    config: Dict[str, Any]
) -> LLMContextManager:
    """
    Create an LLMContextManager from a configuration dictionary.
    
    Args:
        config: Configuration dictionary with keys:
            - max_tokens: int
            - system_instructions: str (optional)
            - memory_content: str (optional)
            - tools: List[Dict[str, Any]] (optional)
            - model: str (optional, defaults to "gpt-4")
            - logger: logging.Logger (optional)
            
    Returns:
        Configured LLMContextManager
    """
    max_tokens = config.get("max_tokens")
    if not max_tokens:
        raise ValueError("max_tokens is required in config")
    
    return create_context_manager(
        max_tokens=max_tokens,
        system_instructions=config.get("system_instructions", ""),
        memory_content=config.get("memory_content", ""),
        tools=config.get("tools"),
        model=config.get("model", "gpt-4"),
        logger=config.get("logger")
    ) 