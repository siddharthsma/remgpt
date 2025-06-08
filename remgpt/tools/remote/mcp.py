"""
Model Context Protocol (MCP) client implementation.
"""

import re
import shlex
from typing import Dict, Any, List
from contextlib import AsyncExitStack

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from mcp.client.sse import sse_client
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

from .base import RemoteToolProtocol


class MCPProtocol(RemoteToolProtocol):
    """Model Context Protocol client."""
    
    def __init__(self):
        if not MCP_AVAILABLE:
            raise ImportError("MCP package not available. Install with: pip install mcp")
        
        self.session = None
        self.exit_stack = AsyncExitStack()
        self._streams_context = None
        self._session_context = None
    
    @classmethod
    async def create(cls, server_path_or_url: str):
        """Create and connect to an MCP server."""
        client = cls()
        await client._connect_to_server(server_path_or_url)
        return client
    
    async def _connect_to_server(self, server_path_or_url: str):
        """Connect to an MCP server (stdio or SSE)."""
        url_pattern = re.compile(r'^https?://')
        
        if url_pattern.match(server_path_or_url):
            await self._connect_to_sse_server(server_path_or_url)
        else:
            await self._connect_to_stdio_server(server_path_or_url)
    
    async def _connect_to_sse_server(self, server_url: str):
        """Connect to an SSE MCP server."""
        self._streams_context = sse_client(url=server_url)
        streams = await self._streams_context.__aenter__()
        
        self._session_context = ClientSession(*streams)
        self.session = await self._session_context.__aenter__()
        
        await self.session.initialize()
    
    async def _connect_to_stdio_server(self, server_script_path: str):
        """Connect to a stdio MCP server."""
        split_args = shlex.split(server_script_path)
        
        if split_args and split_args[0] in ("npm", "npx", "uv", "uvx"):
            command = split_args[0]
            args = split_args[1:] if len(split_args) > 1 else []
        else:
            if server_script_path.endswith(".py"):
                command = "python"
            elif server_script_path.endswith(".js"):
                command = "node"
            else:
                raise ValueError("Server script must be a .py, .js file, or npm/npx/uv/uvx command")
            args = [server_script_path]
        
        server_params = StdioServerParameters(command=command, args=args, env=None)
        
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.writer = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.writer))
        
        await self.session.initialize()
    
    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available MCP tools."""
        response = await self.session.list_tools()
        tools = []
        for tool in response.tools:
            tools.append({
                "name": tool.name,
                "description": tool.description,
                "inputSchema": tool.inputSchema.model_dump() if hasattr(tool.inputSchema, 'model_dump') else tool.inputSchema
            })
        return tools
    
    async def call_tool(self, tool_name: str, args: Dict[str, Any]) -> Any:
        """Call an MCP tool."""
        response = await self.session.call_tool(tool_name, args)
        return response.content
    
    async def cleanup(self):
        """Clean up MCP resources."""
        await self.exit_stack.aclose()
        if self._session_context:
            await self._session_context.__aexit__(None, None, None)
        if self._streams_context:
            await self._streams_context.__aexit__(None, None, None) 