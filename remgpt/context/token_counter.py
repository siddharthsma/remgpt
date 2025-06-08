"""
Token counting utility for LLM Context management system.
"""

from typing import List
import tiktoken
from ..types import Message


class TokenCounter:
    """Utility class for counting tokens in messages."""
    
    def __init__(self, model: str = "gpt-4"):
        """Initialize with specific model encoding."""
        try:
            self.encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            # Fallback to cl100k_base if model not found
            self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def count_message_tokens(self, message: Message) -> int:
        """Count tokens in a single message."""
        # Base tokens for message structure
        tokens = 3  # Basic message overhead
        
        # Add role tokens
        tokens += len(self.encoding.encode(message.role.value))
        
        # Add content tokens
        if isinstance(message.content, str):
            tokens += len(self.encoding.encode(message.content))
        elif isinstance(message.content, list):
            # Handle multimodal content
            for content_block in message.content:
                if hasattr(content_block, 'text'):
                    tokens += len(self.encoding.encode(content_block.text))
                # Note: Image tokens would need special handling
        
        # Add name tokens if present
        if message.name:
            tokens += len(self.encoding.encode(message.name))
            
        return tokens
    
    def count_messages_tokens(self, messages: List[Message]) -> int:
        """Count total tokens in a list of messages."""
        return sum(self.count_message_tokens(msg) for msg in messages) 