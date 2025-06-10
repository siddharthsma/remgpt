"""
Token counting utilities for LLM context management.
"""

import logging
from typing import List, Union, Optional
from ..core.types import Message

class TokenCounter:
    """
    Utility class for counting tokens in messages and text.
    
    Provides methods to estimate token counts for different content types
    and helps manage context size limits.
    """
    
    def __init__(self, chars_per_token: float = 4.0, logger: Optional[logging.Logger] = None):
        """
        Initialize the token counter.
        
        Args:
            chars_per_token: Average characters per token (default: 4.0 for most models)
            logger: Optional logger instance
        """
        self.chars_per_token = chars_per_token
        self.logger = logger or logging.getLogger(__name__)
        self.model_used = None
    
    def update_from_llm_client(self, llm_client):
        """
        Update token counting parameters from LLM client.
        
        Args:
            llm_client: The LLM client to sync with
        """
        if hasattr(llm_client, 'model_name'):
            self.model_used = llm_client.model_name
            
            # Adjust chars_per_token based on model if needed
            # Different models may have different tokenization patterns
            if 'gpt' in llm_client.model_name.lower():
                self.chars_per_token = 4.0  # Good default for GPT models
            elif 'claude' in llm_client.model_name.lower():
                self.chars_per_token = 3.8  # Claude tends to be slightly different
            
            self.logger.info(f"Token counter synced with model: {self.model_used}")
        else:
            self.logger.warning("LLM client does not have model_name attribute")
    
    def count_message_tokens(self, message: Message) -> int:
        """
        Count tokens in a single message.
        
        Args:
            message: The message to count tokens for
            
        Returns:
            Estimated token count
        """
        if not message or not message.content:
            return 0
        
        # Handle different content types
        if isinstance(message.content, str):
            return self._count_text_tokens(message.content)
        elif isinstance(message.content, list):
            # Multi-modal content (text + images)
            total_tokens = 0
            for content_block in message.content:
                if hasattr(content_block, 'text'):
                    total_tokens += self._count_text_tokens(content_block.text)
                elif hasattr(content_block, 'type') and content_block.type == 'image':
                    # Rough estimate for image tokens (varies by model)
                    total_tokens += 85  # Conservative estimate for image processing
            return total_tokens
        else:
            return self._count_text_tokens(str(message.content))
    
    def count_messages_tokens(self, messages: List[Message]) -> int:
        """
        Count tokens across multiple messages.
        
        Args:
            messages: List of messages to count tokens for
            
        Returns:
            Total estimated token count
        """
        return sum(self.count_message_tokens(msg) for msg in messages)
    
    def _count_text_tokens(self, text: str) -> int:
        """
        Count tokens in text content.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Estimated token count
        """
        if not text:
            return 0
        
        # Simple character-based estimation
        # More sophisticated tokenizers could be used here
        char_count = len(text)
        token_count = max(1, int(char_count / self.chars_per_token))
        
        return token_count
    
    def estimate_response_tokens(self, prompt_tokens: int, max_response_ratio: float = 0.3) -> int:
        """
        Estimate tokens needed for response based on prompt size.
        
        Args:
            prompt_tokens: Number of tokens in the prompt
            max_response_ratio: Maximum ratio of response to prompt tokens
            
        Returns:
            Estimated response token count
        """
        return int(prompt_tokens * max_response_ratio)
    
    def check_token_limit(self, messages: List[Message], max_tokens: int, 
                         reserve_tokens: int = 500) -> bool:
        """
        Check if messages would exceed token limit.
        
        Args:
            messages: Messages to check
            max_tokens: Maximum allowed tokens
            reserve_tokens: Tokens to reserve for response
            
        Returns:
            True if within limits, False if would exceed
        """
        current_tokens = self.count_messages_tokens(messages)
        return (current_tokens + reserve_tokens) <= max_tokens 