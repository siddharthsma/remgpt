"""
Memory instructions block for LLM Context management.
"""

from typing import List, Optional
import logging
from ..base_block import BaseBlock
from ...types import Message, SystemMessage


class MemoryInstructionsBlock(BaseBlock):
    """Block containing memory instructions (read-only)."""
    
    def __init__(self, memory_content: str, name: str = "memory_instructions", logger: Optional[logging.Logger] = None):
        """Initialize with memory content."""
        super().__init__(name, is_read_only=True, logger=logger)
        
        # If no custom memory content is provided, use default context management instructions
        if not memory_content.strip():
            memory_content = self._get_default_context_instructions()
        
        self.memory_content = memory_content
    
    def _get_default_context_instructions(self) -> str:
        """Get default context management instructions for the LLM."""
        return """CONTEXT MANAGEMENT INSTRUCTIONS:

You have access to special context management functions that help you manage your memory efficiently. Here's when and how to use them:

1. TOPIC DRIFT DETECTED:
   - When you receive a message stating "TOPIC DRIFT DETECTED", it means the conversation has shifted to a new topic
   - You should call the 'save_current_topic' function to summarize and save the previous conversation topic
   - Provide a clear, concise summary and key facts from the conversation
   - This will clear the conversation history and add the topic to your working memory

2. APPROACHING TOKEN LIMIT:
   - When you receive a message stating "APPROACHING TOKEN LIMIT", your context is getting full
   - You should call the 'evict_oldest_topic' function to remove the oldest saved topic
   - This will free up space for new conversation content
   - Only do this when explicitly warned about the token limit

3. FUNCTION CALLING GUIDELINES:
   - save_current_topic(topic_summary, topic_key_facts): Use when topic drift is detected
     * topic_summary: A concise summary of the conversation topic (1-2 sentences)
     * topic_key_facts: List of important facts or decisions made (optional)
   
   - evict_oldest_topic(): Use when approaching token limit
     * No parameters needed, automatically removes the oldest topic

4. IMPORTANT NOTES:
   - These functions are for context management only - use them when prompted by the system
   - Regular conversations do not require these functions
   - Your working memory contains saved topics that provide context for ongoing conversations
   - Always prioritize the current conversation while being aware of saved topics for context"""
    
    def to_messages(self) -> List[Message]:
        """Convert to system message."""
        if not self.memory_content.strip():
            return []
        return [SystemMessage(content=f"Memory Instructions:\n{self.memory_content}")]
    
    def update_memory_content(self, new_content: str):
        """
        Update memory content (for dynamic instruction updates).
        
        Args:
            new_content: New memory instructions
        """
        self.memory_content = new_content
        self.logger.info("Memory instructions updated") 