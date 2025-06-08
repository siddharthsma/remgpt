"""
Manager for remote tools from MCP and A2A sources.
"""

from typing import Dict, List

from .base import RemoteToolProtocol
from .tool import RemoteTool

# Import protocols with availability checks
try:
    from .mcp import MCPProtocol, MCP_AVAILABLE
except ImportError:
    MCP_AVAILABLE = False

try:
    from .a2a import A2AProtocol, HTTPX_AVAILABLE
except ImportError:
    HTTPX_AVAILABLE = False


class RemoteToolManager:
    """Manager for remote tools from MCP and A2A sources."""
    
    def __init__(self):
        self.protocols: Dict[str, RemoteToolProtocol] = {}
        self.remote_tools: List[RemoteTool] = []
    
    async def add_mcp_servers(self, mcp_urls_or_paths: List[str]) -> List[RemoteTool]:
        """Add MCP servers and discover their tools."""
        if not MCP_AVAILABLE:
            print("Warning: MCP package not available. Install with: pip install mcp")
            return []
        
        new_tools = []
        for i, url_or_path in enumerate(mcp_urls_or_paths):
            try:
                protocol = await MCPProtocol.create(url_or_path)
                source_id = f"mcp_{i}"
                self.protocols[source_id] = protocol
                
                tools_info = await protocol.list_tools()
                for tool_info in tools_info:
                    remote_tool = RemoteTool(tool_info, protocol, source_id)
                    self.remote_tools.append(remote_tool)
                    new_tools.append(remote_tool)
                    
            except Exception as e:
                # Log error but continue with other servers
                print(f"Failed to connect to MCP server {url_or_path}: {e}")
        
        return new_tools
    
    async def add_a2a_agents(self, agent_base_urls: List[str]) -> List[RemoteTool]:
        """Add A2A agents and discover their tools."""
        if not HTTPX_AVAILABLE:
            print("Warning: httpx package not available. Install with: pip install httpx")
            return []
        
        new_tools = []
        for i, base_url in enumerate(agent_base_urls):
            try:
                protocol = A2AProtocol(base_url, f"agent_{i}")
                source_id = f"a2a_{i}"
                self.protocols[source_id] = protocol
                
                tools_info = await protocol.list_tools()
                for tool_info in tools_info:
                    remote_tool = RemoteTool(tool_info, protocol, source_id)
                    self.remote_tools.append(remote_tool)
                    new_tools.append(remote_tool)
                    
            except Exception as e:
                # Log error but continue with other agents
                print(f"Failed to connect to A2A agent {base_url}: {e}")
        
        return new_tools
    
    def get_remote_tools(self) -> List[RemoteTool]:
        """Get all discovered remote tools."""
        return self.remote_tools
    
    async def cleanup(self):
        """Clean up all protocol connections."""
        for protocol in self.protocols.values():
            await protocol.cleanup()
        self.protocols.clear()
        self.remote_tools.clear() 