"""
Base protocol interface for remote tool systems.
"""

from typing import Dict, Any, List
from abc import ABC, abstractmethod


class RemoteToolProtocol(ABC):
    """Base class for remote tool protocols."""
    
    @abstractmethod
    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools from the remote service."""
        raise NotImplementedError
    
    @abstractmethod
    async def call_tool(self, tool_name: str, args: Dict[str, Any]) -> Any:
        """Execute a tool on the remote service."""
        raise NotImplementedError
    
    async def cleanup(self):
        """Clean up resources. Default implementation does nothing."""
        pass 