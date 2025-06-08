"""
Remote tools module for MCP and A2A protocol integration.
"""

from .base import RemoteToolProtocol
from .tool import RemoteTool
from .manager import RemoteToolManager

# Import protocols with availability checks
try:
    from .mcp import MCPProtocol
    MCP_AVAILABLE = True
except ImportError:
    MCPProtocol = None
    MCP_AVAILABLE = False

try:
    from .a2a import A2AProtocol
    A2A_AVAILABLE = True
except ImportError:
    A2AProtocol = None
    A2A_AVAILABLE = False

__all__ = [
    "RemoteToolProtocol",
    "RemoteTool", 
    "RemoteToolManager"
]

# Add protocols to exports if available
if MCP_AVAILABLE:
    __all__.append("MCPProtocol")

if A2A_AVAILABLE:
    __all__.append("A2AProtocol") 