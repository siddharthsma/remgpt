"""
LLM Context class for managing multiple blocks.
"""

from typing import List, Dict, Any, Optional
import logging
from .base_block import BaseBlock
from .token_counter import TokenCounter
from .blocks import (
    SystemInstructionsBlock,
    MemoryInstructionsBlock,
    ToolsDefinitionsBlock,
    WorkingContextBlock,
    FIFOQueueBlock
)
from ..core.types import Message


class LLMContext:
    """Main context container that manages multiple blocks."""
    
    def __init__(self, token_counter: Optional[TokenCounter] = None, logger: Optional[logging.Logger] = None):
        """Initialize LLM Context."""
        self.token_counter = token_counter or TokenCounter()
        self.logger = logger or logging.getLogger(__name__)
        self.blocks: Dict[str, BaseBlock] = {}
        
        # Initialize default blocks with logger
        self.system_instructions = SystemInstructionsBlock("", logger=self.logger)
        self.memory_instructions = MemoryInstructionsBlock("", logger=self.logger)
        self.tools_definitions = ToolsDefinitionsBlock([], logger=self.logger)
        self.working_context = WorkingContextBlock({}, logger=self.logger)
        self.fifo_queue = FIFOQueueBlock(logger=self.logger)
        
        # Set token counters for all blocks
        self._update_all_token_counters()
        
        # Add default blocks to collection
        self.blocks = {
            "system_instructions": self.system_instructions,
            "memory_instructions": self.memory_instructions,
            "tools_definitions": self.tools_definitions,
            "working_context": self.working_context,
            "fifo_queue": self.fifo_queue
        }
    
    def _update_all_token_counters(self):
        """Update token counters for all blocks."""
        for block in [self.system_instructions, self.memory_instructions, 
                     self.tools_definitions, self.working_context, self.fifo_queue]:
            block.set_token_counter(self.token_counter)
    
    def to_messages(self, block_order: Optional[List[str]] = None) -> List[Message]:
        """
        Convert entire context to list of messages.
        
        NEW ALGORITHM:
        System messages are constructed from:
        - SystemInstructionsBlock
        - MemoryInstructionsBlock  
        - ToolsDefinitionsBlock
        
        Conversation messages come from:
        - WorkingContextBlock (saved topics)
        - FIFOQueueBlock (current conversation)
        
        Args:
            block_order: Optional list specifying the order of blocks.
                        If None, uses default order.
        """
        if block_order is None:
            # Default order: System blocks first, then conversation blocks
            block_order = [
                "system_instructions",
                "memory_instructions", 
                "tools_definitions",
                "working_context",
                "fifo_queue"
            ]
        
        all_messages = []
        
        # First, collect system messages from system blocks
        system_blocks = ["system_instructions", "memory_instructions", "tools_definitions"]
        system_content_parts = []
        
        for block_name in system_blocks:
            if block_name in self.blocks and block_name in block_order:
                block_messages = self.blocks[block_name].to_messages()
                for msg in block_messages:
                    # Extract content from system messages
                    if hasattr(msg, 'content') and msg.content.strip():
                        system_content_parts.append(msg.content)
        
        # Create a single comprehensive system message
        if system_content_parts:
            from ..core.types import SystemMessage
            combined_system_content = "\n\n".join(system_content_parts)
            system_message = SystemMessage(content=combined_system_content)
            all_messages.append(system_message)
        
        # Then add conversation messages (working context + fifo queue)
        conversation_blocks = ["working_context", "fifo_queue"]
        for block_name in conversation_blocks:
            if block_name in self.blocks and block_name in block_order:
                block_messages = self.blocks[block_name].to_messages()
                all_messages.extend(block_messages)
        
        return all_messages
    
    def get_total_tokens(self) -> int:
        """Get total token count across all blocks."""
        return sum(block.get_token_count() for block in self.blocks.values())
    
    def get_block_token_counts(self) -> Dict[str, int]:
        """Get token count for each block."""
        return {name: block.get_token_count() for name, block in self.blocks.items()} 