"""
Unit tests for RemoteToolProtocol base class.
"""

import pytest
from remgpt.tools.remote.base import RemoteToolProtocol


class ConcreteProtocol(RemoteToolProtocol):
    """Concrete implementation for testing."""
    
    def __init__(self, connection_string="test://example"):
        self.connection_string = connection_string
        self.is_connected = False
        self.tools_data = [
            {
                "name": "test_tool_1",
                "description": "First test tool",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "param1": {"type": "string", "description": "First parameter"}
                    },
                    "required": ["param1"]
                }
            },
            {
                "name": "test_tool_2", 
                "description": "Second test tool",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        ]
    
    async def connect(self):
        """Mock connect implementation."""
        self.is_connected = True
    
    async def disconnect(self):
        """Mock disconnect implementation."""
        self.is_connected = False
    
    async def list_tools(self):
        """Mock list_tools implementation."""
        if not self.is_connected:
            raise RuntimeError("Not connected")
        return self.tools_data
    
    async def call_tool(self, name, arguments):
        """Mock call_tool implementation."""
        if not self.is_connected:
            raise RuntimeError("Not connected")
        
        if name == "test_tool_1":
            return {"result": f"Tool 1 called with {arguments}"}
        elif name == "test_tool_2":
            return {"result": "Tool 2 called"}
        else:
            raise ValueError(f"Unknown tool: {name}")


class TestRemoteToolProtocol:
    """Test cases for RemoteToolProtocol abstract class."""
    
    def test_abstract_class(self):
        """Test that RemoteToolProtocol is abstract and cannot be instantiated."""
        with pytest.raises(TypeError):
            RemoteToolProtocol()
    
    def test_concrete_implementation(self):
        """Test that concrete implementation can be instantiated."""
        protocol = ConcreteProtocol("test://example")
        assert protocol.connection_string == "test://example"
        assert protocol.is_connected is False
    
    def test_connection_string_property(self):
        """Test connection_string property."""
        protocol = ConcreteProtocol("custom://connection")
        assert protocol.connection_string == "custom://connection"
    
    @pytest.mark.asyncio
    async def test_connect(self):
        """Test connect method."""
        protocol = ConcreteProtocol()
        assert protocol.is_connected is False
        
        await protocol.connect()
        assert protocol.is_connected is True
    
    @pytest.mark.asyncio
    async def test_disconnect(self):
        """Test disconnect method."""
        protocol = ConcreteProtocol()
        await protocol.connect()
        assert protocol.is_connected is True
        
        await protocol.disconnect()
        assert protocol.is_connected is False
    
    @pytest.mark.asyncio
    async def test_list_tools_connected(self):
        """Test list_tools when connected."""
        protocol = ConcreteProtocol()
        await protocol.connect()
        
        tools = await protocol.list_tools()
        assert len(tools) == 2
        assert tools[0]["name"] == "test_tool_1"
        assert tools[1]["name"] == "test_tool_2"
    
    @pytest.mark.asyncio
    async def test_list_tools_not_connected(self):
        """Test list_tools when not connected."""
        protocol = ConcreteProtocol()
        
        with pytest.raises(RuntimeError, match="Not connected"):
            await protocol.list_tools()
    
    @pytest.mark.asyncio
    async def test_call_tool_success(self):
        """Test successful tool call."""
        protocol = ConcreteProtocol()
        await protocol.connect()
        
        result = await protocol.call_tool("test_tool_1", {"param1": "value1"})
        assert result["result"] == "Tool 1 called with {'param1': 'value1'}"
    
    @pytest.mark.asyncio
    async def test_call_tool_not_connected(self):
        """Test tool call when not connected."""
        protocol = ConcreteProtocol()
        
        with pytest.raises(RuntimeError, match="Not connected"):
            await protocol.call_tool("test_tool_1", {})
    
    @pytest.mark.asyncio
    async def test_call_tool_unknown(self):
        """Test calling unknown tool."""
        protocol = ConcreteProtocol()
        await protocol.connect()
        
        with pytest.raises(ValueError, match="Unknown tool: unknown_tool"):
            await protocol.call_tool("unknown_tool", {})


class FailingProtocol(RemoteToolProtocol):
    """Protocol that fails for testing error handling."""
    
    def __init__(self):
        self.connection_string = "failing://connection"
    
    async def connect(self):
        """Always fails to connect."""
        raise ConnectionError("Failed to connect to remote service")
    
    async def disconnect(self):
        """Always succeeds."""
        pass
    
    async def list_tools(self):
        """Always fails."""
        raise RuntimeError("Failed to list tools")
    
    async def call_tool(self, name, arguments):
        """Always fails."""
        raise RuntimeError("Failed to call tool")


class TestRemoteToolProtocolErrorHandling:
    """Test error handling in RemoteToolProtocol."""
    
    @pytest.mark.asyncio
    async def test_connect_failure(self):
        """Test connection failure handling."""
        protocol = FailingProtocol()
        
        with pytest.raises(ConnectionError, match="Failed to connect to remote service"):
            await protocol.connect()
    
    @pytest.mark.asyncio
    async def test_list_tools_failure(self):
        """Test list_tools failure handling."""
        protocol = FailingProtocol()
        
        with pytest.raises(RuntimeError, match="Failed to list tools"):
            await protocol.list_tools()
    
    @pytest.mark.asyncio
    async def test_call_tool_failure(self):
        """Test call_tool failure handling."""
        protocol = FailingProtocol()
        
        with pytest.raises(RuntimeError, match="Failed to call tool"):
            await protocol.call_tool("any_tool", {})


class TestRemoteToolProtocolLifecycle:
    """Test the lifecycle of RemoteToolProtocol."""
    
    @pytest.mark.asyncio
    async def test_connection_lifecycle(self):
        """Test complete connection lifecycle."""
        protocol = ConcreteProtocol("lifecycle://test")
        
        # Start disconnected
        assert protocol.is_connected is False
        
        # Connect
        await protocol.connect()
        assert protocol.is_connected is True
        
        # Use connection
        tools = await protocol.list_tools()
        assert len(tools) == 2
        
        result = await protocol.call_tool("test_tool_2", {})
        assert result["result"] == "Tool 2 called"
        
        # Disconnect
        await protocol.disconnect()
        assert protocol.is_connected is False
        
        # Should not work when disconnected
        with pytest.raises(RuntimeError):
            await protocol.list_tools()
    
    @pytest.mark.asyncio
    async def test_multiple_connections(self):
        """Test multiple connect/disconnect cycles."""
        protocol = ConcreteProtocol()
        
        # First cycle
        await protocol.connect()
        assert protocol.is_connected is True
        await protocol.disconnect()
        assert protocol.is_connected is False
        
        # Second cycle
        await protocol.connect()
        assert protocol.is_connected is True
        tools = await protocol.list_tools()
        assert len(tools) == 2
        await protocol.disconnect()
        assert protocol.is_connected is False 