"""
Unit tests for RemoteToolManager class.
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from remgpt.tools.remote.manager import RemoteToolManager
from remgpt.tools.remote.base import RemoteToolProtocol
from remgpt.tools.remote.tool import RemoteTool


class MockProtocol(RemoteToolProtocol):
    """Mock protocol for testing."""
    
    def __init__(self, connection_string, tools_data=None):
        self.connection_string = connection_string
        self.is_connected = False
        self.tools_data = tools_data or []
        self.call_results = {}
    
    async def connect(self):
        self.is_connected = True
    
    async def disconnect(self):
        self.is_connected = False
    
    async def list_tools(self):
        if not self.is_connected:
            raise RuntimeError("Not connected")
        return self.tools_data
    
    async def call_tool(self, name, arguments):
        if not self.is_connected:
            raise RuntimeError("Not connected")
        if name in self.call_results:
            return self.call_results[name]
        raise ValueError(f"Unknown tool: {name}")


class TestRemoteToolManagerInitialization:
    """Test RemoteToolManager initialization."""
    
    def test_empty_initialization(self):
        """Test initialization with no protocols."""
        manager = RemoteToolManager()
        assert len(manager.protocols) == 0
        assert len(manager.tools) == 0
    
    def test_initialization_with_protocols(self):
        """Test initialization with protocols."""
        protocol1 = MockProtocol("mock://test1")
        protocol2 = MockProtocol("mock://test2")
        
        manager = RemoteToolManager([protocol1, protocol2])
        
        assert len(manager.protocols) == 2
        assert protocol1 in manager.protocols
        assert protocol2 in manager.protocols
        assert len(manager.tools) == 0  # Tools not discovered yet


class TestRemoteToolManagerProtocolManagement:
    """Test protocol management in RemoteToolManager."""
    
    def test_add_protocol(self):
        """Test adding protocols."""
        manager = RemoteToolManager()
        protocol = MockProtocol("mock://test")
        
        manager.add_protocol(protocol)
        
        assert len(manager.protocols) == 1
        assert protocol in manager.protocols
    
    def test_add_multiple_protocols(self):
        """Test adding multiple protocols."""
        manager = RemoteToolManager()
        protocols = [
            MockProtocol("mock://test1"),
            MockProtocol("mock://test2"),
            MockProtocol("mock://test3")
        ]
        
        for protocol in protocols:
            manager.add_protocol(protocol)
        
        assert len(manager.protocols) == 3
        for protocol in protocols:
            assert protocol in manager.protocols
    
    def test_remove_protocol(self):
        """Test removing protocols."""
        protocol1 = MockProtocol("mock://test1")
        protocol2 = MockProtocol("mock://test2")
        manager = RemoteToolManager([protocol1, protocol2])
        
        result = manager.remove_protocol(protocol1)
        
        assert result is True
        assert len(manager.protocols) == 1
        assert protocol1 not in manager.protocols
        assert protocol2 in manager.protocols
    
    def test_remove_nonexistent_protocol(self):
        """Test removing protocol that doesn't exist."""
        manager = RemoteToolManager()
        protocol = MockProtocol("mock://test")
        
        result = manager.remove_protocol(protocol)
        assert result is False
    
    def test_clear_protocols(self):
        """Test clearing all protocols."""
        protocols = [MockProtocol(f"mock://test{i}") for i in range(3)]
        manager = RemoteToolManager(protocols)
        
        manager.clear_protocols()
        
        assert len(manager.protocols) == 0
        assert len(manager.tools) == 0


class TestRemoteToolManagerToolDiscovery:
    """Test tool discovery in RemoteToolManager."""
    
    @pytest.mark.asyncio
    async def test_discover_tools_single_protocol(self):
        """Test discovering tools from single protocol."""
        tools_data = [
            {
                "name": "calculator",
                "description": "Perform calculations",
                "parameters": {
                    "type": "object",
                    "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                    "required": ["a", "b"]
                }
            },
            {
                "name": "weather",
                "description": "Get weather info",
                "parameters": {"type": "object", "properties": {"location": {"type": "string"}}}
            }
        ]
        
        protocol = MockProtocol("mock://test", tools_data)
        manager = RemoteToolManager([protocol])
        
        await manager.discover_tools()
        
        assert len(manager.tools) == 2
        
        # Check tool properties
        calc_tool = next(tool for tool in manager.tools if tool.name == "calculator")
        weather_tool = next(tool for tool in manager.tools if tool.name == "weather")
        
        assert calc_tool.description == "Perform calculations"
        assert weather_tool.description == "Get weather info"
    
    @pytest.mark.asyncio
    async def test_discover_tools_multiple_protocols(self):
        """Test discovering tools from multiple protocols."""
        tools_data1 = [
            {"name": "tool1", "description": "First tool"},
            {"name": "tool2", "description": "Second tool"}
        ]
        tools_data2 = [
            {"name": "tool3", "description": "Third tool"},
            {"name": "tool4", "description": "Fourth tool"}
        ]
        
        protocol1 = MockProtocol("mock://test1", tools_data1)
        protocol2 = MockProtocol("mock://test2", tools_data2)
        manager = RemoteToolManager([protocol1, protocol2])
        
        await manager.discover_tools()
        
        assert len(manager.tools) == 4
        tool_names = {tool.name for tool in manager.tools}
        assert tool_names == {"tool1", "tool2", "tool3", "tool4"}
    
    @pytest.mark.asyncio
    async def test_discover_tools_empty_protocols(self):
        """Test discovering tools when protocols have no tools."""
        protocol1 = MockProtocol("mock://test1", [])
        protocol2 = MockProtocol("mock://test2", [])
        manager = RemoteToolManager([protocol1, protocol2])
        
        await manager.discover_tools()
        
        assert len(manager.tools) == 0
    
    @pytest.mark.asyncio
    async def test_discover_tools_duplicate_names(self):
        """Test discovering tools with duplicate names across protocols."""
        tools_data1 = [{"name": "shared_tool", "description": "Tool from protocol 1"}]
        tools_data2 = [{"name": "shared_tool", "description": "Tool from protocol 2"}]
        
        protocol1 = MockProtocol("mock://test1", tools_data1)
        protocol2 = MockProtocol("mock://test2", tools_data2)
        manager = RemoteToolManager([protocol1, protocol2])
        
        await manager.discover_tools()
        
        # Should have 2 tools even with same name (different protocols)
        assert len(manager.tools) == 2
        
        # Both should be accessible
        shared_tools = [tool for tool in manager.tools if tool.name == "shared_tool"]
        assert len(shared_tools) == 2
    
    @pytest.mark.asyncio
    async def test_discover_tools_connection_failure(self):
        """Test tool discovery when protocol connection fails."""
        protocol = MockProtocol("mock://test", [{"name": "test_tool", "description": "Test"}])
        
        # Override connect to fail
        async def failing_connect():
            raise ConnectionError("Failed to connect")
        protocol.connect = failing_connect
        
        manager = RemoteToolManager([protocol])
        
        # Should not raise exception but log error
        await manager.discover_tools()
        
        # No tools should be discovered
        assert len(manager.tools) == 0
    
    @pytest.mark.asyncio
    async def test_rediscover_tools(self):
        """Test rediscovering tools (clearing and discovering again)."""
        tools_data = [{"name": "tool1", "description": "First tool"}]
        protocol = MockProtocol("mock://test", tools_data)
        manager = RemoteToolManager([protocol])
        
        # First discovery
        await manager.discover_tools()
        assert len(manager.tools) == 1
        
        # Update protocol tools
        protocol.tools_data = [
            {"name": "tool1", "description": "Updated tool"},
            {"name": "tool2", "description": "New tool"}
        ]
        
        # Rediscover
        await manager.discover_tools()
        assert len(manager.tools) == 2
        
        # Check that tools were updated
        tool_names = {tool.name for tool in manager.tools}
        assert tool_names == {"tool1", "tool2"}


class TestRemoteToolManagerToolAccess:
    """Test tool access methods in RemoteToolManager."""
    
    @pytest.mark.asyncio
    async def test_get_tool_by_name(self):
        """Test getting tool by name."""
        tools_data = [
            {"name": "calculator", "description": "Math tool"},
            {"name": "weather", "description": "Weather tool"}
        ]
        protocol = MockProtocol("mock://test", tools_data)
        manager = RemoteToolManager([protocol])
        
        await manager.discover_tools()
        
        calc_tool = manager.get_tool("calculator")
        assert calc_tool is not None
        assert calc_tool.name == "calculator"
        assert calc_tool.description == "Math tool"
        
        weather_tool = manager.get_tool("weather")
        assert weather_tool is not None
        assert weather_tool.name == "weather"
    
    @pytest.mark.asyncio
    async def test_get_nonexistent_tool(self):
        """Test getting tool that doesn't exist."""
        protocol = MockProtocol("mock://test", [])
        manager = RemoteToolManager([protocol])
        
        await manager.discover_tools()
        
        tool = manager.get_tool("nonexistent")
        assert tool is None
    
    @pytest.mark.asyncio
    async def test_list_all_tools(self):
        """Test listing all discovered tools."""
        tools_data = [
            {"name": "tool1", "description": "First tool"},
            {"name": "tool2", "description": "Second tool"},
            {"name": "tool3", "description": "Third tool"}
        ]
        protocol = MockProtocol("mock://test", tools_data)
        manager = RemoteToolManager([protocol])
        
        await manager.discover_tools()
        
        all_tools = manager.list_tools()
        assert len(all_tools) == 3
        
        tool_names = {tool.name for tool in all_tools}
        assert tool_names == {"tool1", "tool2", "tool3"}
    
    @pytest.mark.asyncio
    async def test_get_tool_schemas(self):
        """Test getting schemas for all tools."""
        tools_data = [
            {
                "name": "api_tool",
                "description": "API tool",
                "parameters": {
                    "type": "object",
                    "properties": {"endpoint": {"type": "string"}},
                    "required": ["endpoint"]
                }
            }
        ]
        protocol = MockProtocol("mock://test", tools_data)
        manager = RemoteToolManager([protocol])
        
        await manager.discover_tools()
        
        schemas = manager.get_tool_schemas()
        assert len(schemas) == 1
        
        schema = schemas[0]
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "api_tool"
        assert schema["function"]["description"] == "API tool"


class TestRemoteToolManagerConnectionManagement:
    """Test connection management in RemoteToolManager."""
    
    @pytest.mark.asyncio
    async def test_connect_all_protocols(self):
        """Test connecting all protocols."""
        protocols = [MockProtocol(f"mock://test{i}") for i in range(3)]
        manager = RemoteToolManager(protocols)
        
        await manager.connect_all()
        
        for protocol in protocols:
            assert protocol.is_connected is True
    
    @pytest.mark.asyncio
    async def test_disconnect_all_protocols(self):
        """Test disconnecting all protocols."""
        protocols = [MockProtocol(f"mock://test{i}") for i in range(3)]
        manager = RemoteToolManager(protocols)
        
        # Connect first
        await manager.connect_all()
        for protocol in protocols:
            assert protocol.is_connected is True
        
        # Then disconnect
        await manager.disconnect_all()
        for protocol in protocols:
            assert protocol.is_connected is False
    
    @pytest.mark.asyncio
    async def test_connection_lifecycle(self):
        """Test complete connection lifecycle."""
        tools_data = [{"name": "test_tool", "description": "Test tool"}]
        protocol = MockProtocol("mock://test", tools_data)
        manager = RemoteToolManager([protocol])
        
        # Start disconnected
        assert protocol.is_connected is False
        
        # Connect and discover tools
        await manager.connect_all()
        assert protocol.is_connected is True
        
        await manager.discover_tools()
        assert len(manager.tools) == 1
        
        # Disconnect
        await manager.disconnect_all()
        assert protocol.is_connected is False
    
    @pytest.mark.asyncio
    async def test_partial_connection_failure(self):
        """Test behavior when some protocols fail to connect."""
        protocol1 = MockProtocol("mock://test1", [{"name": "tool1", "description": "Tool 1"}])
        protocol2 = MockProtocol("mock://test2", [{"name": "tool2", "description": "Tool 2"}])
        
        # Make protocol2 fail to connect
        async def failing_connect():
            raise ConnectionError("Connection failed")
        protocol2.connect = failing_connect
        
        manager = RemoteToolManager([protocol1, protocol2])
        
        # Should not raise exception
        await manager.connect_all()
        
        # Protocol1 should be connected, protocol2 should not
        assert protocol1.is_connected is True
        assert protocol2.is_connected is False
        
        # Discovery should work for protocol1 only
        await manager.discover_tools()
        assert len(manager.tools) == 1
        assert manager.tools[0].name == "tool1"


class TestRemoteToolManagerIntegration:
    """Integration tests for RemoteToolManager."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow."""
        # Setup protocols with different tools
        tools_data1 = [
            {
                "name": "math_add",
                "description": "Add two numbers",
                "parameters": {
                    "type": "object",
                    "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                    "required": ["a", "b"]
                }
            }
        ]
        tools_data2 = [
            {
                "name": "text_upper",
                "description": "Convert text to uppercase",
                "parameters": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"]
                }
            }
        ]
        
        protocol1 = MockProtocol("mock://math", tools_data1)
        protocol1.call_results["math_add"] = {"result": 8}
        
        protocol2 = MockProtocol("mock://text", tools_data2)
        protocol2.call_results["text_upper"] = {"result": "HELLO WORLD"}
        
        manager = RemoteToolManager([protocol1, protocol2])
        
        # Complete workflow
        await manager.connect_all()
        await manager.discover_tools()
        
        # Verify tools are available
        assert len(manager.tools) == 2
        
        math_tool = manager.get_tool("math_add")
        text_tool = manager.get_tool("text_upper")
        
        assert math_tool is not None
        assert text_tool is not None
        
        # Test tool execution
        math_result = await math_tool.execute(a=5, b=3)
        assert math_result == {"result": 8}
        
        text_result = await text_tool.execute(text="hello world")
        assert text_result == {"result": "HELLO WORLD"}
        
        # Cleanup
        await manager.disconnect_all()
        
        # Verify disconnection
        assert protocol1.is_connected is False
        assert protocol2.is_connected is False
    
    @pytest.mark.asyncio
    async def test_dynamic_protocol_management(self):
        """Test adding and removing protocols dynamically."""
        manager = RemoteToolManager()
        
        # Start with no tools
        await manager.discover_tools()
        assert len(manager.tools) == 0
        
        # Add first protocol
        protocol1 = MockProtocol("mock://test1", [{"name": "tool1", "description": "Tool 1"}])
        manager.add_protocol(protocol1)
        
        await manager.connect_all()
        await manager.discover_tools()
        assert len(manager.tools) == 1
        
        # Add second protocol
        protocol2 = MockProtocol("mock://test2", [{"name": "tool2", "description": "Tool 2"}])
        manager.add_protocol(protocol2)
        
        await manager.connect_all()  # Connect new protocol
        await manager.discover_tools()  # Rediscover tools
        assert len(manager.tools) == 2
        
        # Remove first protocol
        manager.remove_protocol(protocol1)
        await manager.discover_tools()  # Rediscover
        assert len(manager.tools) == 1
        assert manager.get_tool("tool1") is None
        assert manager.get_tool("tool2") is not None
        
        # Clean up
        await manager.disconnect_all() 