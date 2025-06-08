"""
Token counting utility for LLM Context management system.
"""

from typing import List, Optional, TYPE_CHECKING
import tiktoken
from ..types import Message

if TYPE_CHECKING:
    from ..llm import BaseLLMClient


class TokenCounter:
    """Utility class for counting tokens in messages."""
    
    def __init__(self, model: Optional[str] = None, llm_client: Optional["BaseLLMClient"] = None):
        """
        Initialize with model encoding.
        
        Args:
            model: Specific model name for tokenization (optional if llm_client provided)
            llm_client: LLM client to get model from (optional)
            
        Priority order:
        1. If model is explicitly provided, use it
        2. If llm_client is provided, use its model_name
        3. Fallback to "gpt-4" as a sensible default
        """
        # Determine which model to use for tokenization
        tokenizer_model = self._determine_tokenizer_model(model, llm_client)
        
        try:
            self.encoding = tiktoken.encoding_for_model(tokenizer_model)
            self.model_used = tokenizer_model
        except KeyError:
            # Fallback to cl100k_base if model not found
            self.encoding = tiktoken.get_encoding("cl100k_base")
            self.model_used = "cl100k_base (fallback)"
    
    def _determine_tokenizer_model(self, model: Optional[str], llm_client: Optional["BaseLLMClient"]) -> str:
        """
        Determine which model to use for tokenization.
        
        Args:
            model: Explicitly provided model name
            llm_client: LLM client instance
            
        Returns:
            str: Model name to use for tokenization
        """
        # Priority 1: Explicit model parameter
        if model:
            return model
        
        # Priority 2: LLM client's model
        if llm_client and hasattr(llm_client, 'model_name') and llm_client.model_name:
            return llm_client.model_name
        
        # Priority 3: Sensible default for modern usage
        return "gpt-4"
    
    def update_from_llm_client(self, llm_client: "BaseLLMClient"):
        """
        Update tokenizer based on LLM client's model.
        
        Args:
            llm_client: LLM client to get model from
        """
        if hasattr(llm_client, 'model_name') and llm_client.model_name:
            new_model = llm_client.model_name
            if new_model != self.model_used:
                try:
                    self.encoding = tiktoken.encoding_for_model(new_model)
                    self.model_used = new_model
                except KeyError:
                    # Keep existing encoding if new model not found
                    pass

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