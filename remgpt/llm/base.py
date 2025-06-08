from abc import ABC, abstractmethod
from typing import List, Dict, Any, Generator, Optional
from .events import Event


class BaseLLMClient(ABC):
    """Abstract base class for all LLM client implementations."""
    
    def __init__(self, model_name: str, **kwargs):
        """
        Initialize the LLM client.
        
        Args:
            model_name: The specific model to use (e.g., "gpt-4", "claude-3-sonnet")
            **kwargs: Provider-specific configuration options
        """
        self.model_name = model_name
        self.config = kwargs
    
    @abstractmethod
    def generate_stream(self, messages: List[Dict[str, Any]], **kwargs) -> Generator[Event, None, None]:
        """
        Generate streaming response from the LLM.
        
        Args:
            messages: List of conversation messages in standard format
            **kwargs: Additional generation parameters (temperature, max_tokens, etc.)
            
        Yields:
            Event: Stream of events representing the LLM's response
        """
        pass
    
    @abstractmethod
    def send_tool_result(self, tool_call_id: str, result: Any) -> None:
        """
        Send tool execution result back to the LLM for continued processing.
        
        Args:
            tool_call_id: ID of the tool call that was executed
            result: The result of the tool execution
        """
        pass
    
    @abstractmethod
    def supports_tools(self) -> bool:
        """
        Check if this LLM provider supports tool/function calling.
        
        Returns:
            bool: True if tools are supported, False otherwise
        """
        pass
    
    @abstractmethod
    def get_supported_models(self) -> List[str]:
        """
        Get list of models supported by this provider.
        
        Returns:
            List[str]: List of supported model names
        """
        pass
    
    def validate_messages(self, messages: List[Dict[str, Any]]) -> bool:
        """
        Validate message format before sending to LLM.
        
        Args:
            messages: List of messages to validate
            
        Returns:
            bool: True if messages are valid, False otherwise
        """
        required_fields = {'role', 'content'}
        
        for message in messages:
            if not isinstance(message, dict):
                return False
            if not required_fields.issubset(message.keys()):
                return False
            if message.get('role') not in ['system', 'user', 'assistant', 'tool']:
                return False
                
        return True
    
    def format_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Format messages to provider-specific format. Override if needed.
        
        Args:
            messages: Standard format messages
            
        Returns:
            List[Dict[str, Any]]: Provider-formatted messages
        """
        return messages 