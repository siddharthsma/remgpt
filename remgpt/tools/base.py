from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseTool(ABC):
    """Abstract base class for all tool implementations."""
    
    def __init__(self, name: str, description: str):
        """
        Initialize the tool.
        
        Args:
            name: Tool name
            description: Tool description
        """
        self.name = name
        self.description = description
    
    @abstractmethod
    async def execute(self, **kwargs) -> Any:
        """
        Execute the tool with the given arguments.
        
        Args:
            **kwargs: Tool-specific arguments
            
        Returns:
            Any: Tool execution result
        """
        pass
    
    @abstractmethod
    def get_schema(self) -> Dict[str, Any]:
        """
        Get the tool's schema definition for LLM function calling.
        
        Returns:
            Dict[str, Any]: Tool schema in OpenAI function calling format
        """
        pass
    
    def validate_args(self, args: Dict[str, Any]) -> bool:
        """
        Validate tool arguments before execution.
        
        Args:
            args: Arguments to validate
            
        Returns:
            bool: True if arguments are valid, False otherwise
        """
        return True
    
    def __str__(self) -> str:
        return f"Tool({self.name}): {self.description}" 