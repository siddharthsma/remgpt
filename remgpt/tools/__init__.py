from .base import BaseTool
from .executor import ToolExecutor

try:
    from .remote import RemoteToolManager, RemoteTool, RemoteToolProtocol
    try:
        from .remote import MCPProtocol
        MCP_AVAILABLE = True
    except ImportError:
        MCP_AVAILABLE = False
    
    try:
        from .remote import A2AProtocol
        A2A_AVAILABLE = True
    except ImportError:
        A2A_AVAILABLE = False
    
    REMOTE_TOOLS_AVAILABLE = True
except ImportError:
    REMOTE_TOOLS_AVAILABLE = False
    MCP_AVAILABLE = False
    A2A_AVAILABLE = False

__all__ = ["BaseTool", "ToolExecutor"]

if REMOTE_TOOLS_AVAILABLE:
    __all__.extend(["RemoteToolManager", "RemoteTool", "RemoteToolProtocol"])
    
    if MCP_AVAILABLE:
        __all__.append("MCPProtocol")
    
    if A2A_AVAILABLE:
        __all__.append("A2AProtocol") 