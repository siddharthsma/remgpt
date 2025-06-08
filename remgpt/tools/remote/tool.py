"""
Remote tool adapter that wraps remote tools as RemGPT BaseTool instances.
"""

from typing import Dict, Any

from ..base import BaseTool
from .base import RemoteToolProtocol


class RemoteTool(BaseTool):
    """Adapter that wraps remote tools as RemGPT BaseTool instances."""
    
    def __init__(self, tool_info: Dict[str, Any], protocol: RemoteToolProtocol, source_prefix: str):
        """
        Initialize a remote tool adapter.
        
        Args:
            tool_info: Tool metadata from remote service
            protocol: Protocol client for executing the tool
            source_prefix: Prefix to identify the source (e.g., "mcp", "a2a_agent1")
        """
        self.tool_info = tool_info
        self.protocol = protocol
        self.source_prefix = source_prefix
        self.remote_tool_name = tool_info["name"]
        
        # Create a unique name for RemGPT's tool system
        unique_name = f"{source_prefix}_{tool_info['name']}"
        
        super().__init__(
            name=unique_name,
            description=tool_info["description"]
        )
    
    async def execute(self, **kwargs) -> Any:
        """Execute the remote tool."""
        try:
            return await self.protocol.call_tool(self.remote_tool_name, kwargs)
        except Exception as e:
            return {"error": f"Remote tool execution failed: {str(e)}"}
    
    def get_schema(self) -> Dict[str, Any]:
        """Get the tool schema for LLM function calling."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.tool_info.get("inputSchema", {
                    "type": "object",
                    "properties": {},
                    "required": []
                })
            }
        } 