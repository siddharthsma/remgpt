"""
Unit tests for MCPProtocol class.
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from remgpt.tools.remote.mcp import MCPProtocol


class TestMCPProtocolInitialization:
    """Test MCPProtocol initialization and configuration parsing."""
    
    def test_stdio_command_parsing(self):
        """Test parsing of stdio commands."""
        # Test uvx command
        protocol = MCPProtocol("uvx weather-mcp@latest stdio")
        assert protocol.connection_type == "stdio"
        assert protocol.command == ["uvx", "weather-mcp@latest", "stdio"]
        
        # Test npm command
        protocol = MCPProtocol("npm run mcp-server")
        assert protocol.connection_type == "stdio"
        assert protocol.command == ["npm", "run", "mcp-server"]
        
        # Test python command
        protocol = MCPProtocol("python -m weather_mcp")
        assert protocol.connection_type == "stdio"
        assert protocol.command == ["python", "-m", "weather_mcp"]
        
        # Test node command
        protocol = MCPProtocol("node server.js")
        assert protocol.connection_type == "stdio"
        assert protocol.command == ["node", "server.js"]
    
    def test_sse_url_parsing(self):
        """Test parsing of SSE URLs."""
        protocol = MCPProtocol("http://localhost:8080/mcp")
        assert protocol.connection_type == "sse"
        assert protocol.sse_url == "http://localhost:8080/mcp"
        
        protocol = MCPProtocol("https://api.example.com/mcp")
        assert protocol.connection_type == "sse"
        assert protocol.sse_url == "https://api.example.com/mcp"
    
    def test_invalid_connection_string(self):
        """Test handling of invalid connection strings."""
        with pytest.raises(ValueError, match="Invalid MCP connection string"):
            MCPProtocol("")
        
        with pytest.raises(ValueError, match="Invalid MCP connection string"):
            MCPProtocol("invalid://protocol")


class TestMCPProtocolWithoutDependencies:
    """Test MCPProtocol behavior when dependencies are not available."""
    
    @patch('remgpt.tools.remote.mcp.mcp', None)
    def test_import_error_handling(self):
        """Test that ImportError is handled gracefully when mcp is not available."""
        with pytest.raises(ImportError, match="MCP library not available"):
            MCPProtocol("uvx weather-mcp@latest stdio")
    
    @patch('remgpt.tools.remote.mcp.httpx_sse', None)
    def test_sse_import_error(self):
        """Test that ImportError is handled for SSE connections when httpx-sse is not available."""
        with pytest.raises(ImportError, match="httpx-sse library not available"):
            MCPProtocol("http://localhost:8080/mcp")


@pytest.fixture
def mock_mcp_dependencies():
    """Mock MCP dependencies for testing."""
    with patch('remgpt.tools.remote.mcp.mcp') as mock_mcp, \
         patch('remgpt.tools.remote.mcp.httpx') as mock_httpx, \
         patch('remgpt.tools.remote.mcp.httpx_sse') as mock_httpx_sse:
        
        # Mock MCP client
        mock_client = AsyncMock()
        mock_mcp.Client.return_value = mock_client
        
        # Mock stdio transport
        mock_stdio_transport = AsyncMock()
        mock_mcp.stdio.stdio_client.return_value.__aenter__ = AsyncMock(return_value=mock_stdio_transport)
        mock_mcp.stdio.stdio_client.return_value.__aexit__ = AsyncMock(return_value=None)
        
        # Mock SSE transport
        mock_sse_transport = AsyncMock()
        mock_httpx_sse.aconnect_sse.return_value.__aenter__ = AsyncMock(return_value=mock_sse_transport)
        mock_httpx_sse.aconnect_sse.return_value.__aexit__ = AsyncMock(return_value=None)
        
        yield {
            'mcp': mock_mcp,
            'httpx': mock_httpx,
            'httpx_sse': mock_httpx_sse,
            'client': mock_client,
            'stdio_transport': mock_stdio_transport,
            'sse_transport': mock_sse_transport
        }


class TestMCPProtocolStdioConnection:
    """Test MCP protocol with stdio connections."""
    
    @pytest.mark.asyncio
    async def test_stdio_connect_success(self, mock_mcp_dependencies):
        """Test successful stdio connection."""
        mocks = mock_mcp_dependencies
        protocol = MCPProtocol("uvx weather-mcp@latest stdio")
        
        # Mock successful initialization
        mocks['client'].initialize.return_value = None
        
        await protocol.connect()
        
        # Verify connection setup
        mocks['mcp'].Client.assert_called_once()
        mocks['mcp'].stdio.stdio_client.assert_called_once_with(["uvx", "weather-mcp@latest", "stdio"])
        mocks['client'].initialize.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_stdio_connect_failure(self, mock_mcp_dependencies):
        """Test stdio connection failure."""
        mocks = mock_mcp_dependencies
        protocol = MCPProtocol("python -m weather_mcp")
        
        # Mock connection failure
        mocks['mcp'].stdio.stdio_client.side_effect = Exception("Failed to start process")
        
        with pytest.raises(Exception, match="Failed to start process"):
            await protocol.connect()
    
    @pytest.mark.asyncio
    async def test_stdio_disconnect(self, mock_mcp_dependencies):
        """Test stdio disconnection."""
        mocks = mock_mcp_dependencies
        protocol = MCPProtocol("npm run mcp-server")
        
        # Connect first
        mocks['client'].initialize.return_value = None
        await protocol.connect()
        
        # Test disconnect
        await protocol.disconnect()
        
        # Verify cleanup
        assert protocol.client is None
        assert protocol.transport is None


class TestMCPProtocolSSEConnection:
    """Test MCP protocol with SSE connections."""
    
    @pytest.mark.asyncio
    async def test_sse_connect_success(self, mock_mcp_dependencies):
        """Test successful SSE connection."""
        mocks = mock_mcp_dependencies
        protocol = MCPProtocol("http://localhost:8080/mcp")
        
        # Mock successful initialization
        mocks['client'].initialize.return_value = None
        
        await protocol.connect()
        
        # Verify connection setup
        mocks['mcp'].Client.assert_called_once()
        mocks['httpx_sse'].aconnect_sse.assert_called_once()
        mocks['client'].initialize.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_sse_connect_failure(self, mock_mcp_dependencies):
        """Test SSE connection failure."""
        mocks = mock_mcp_dependencies
        protocol = MCPProtocol("https://api.example.com/mcp")
        
        # Mock connection failure
        mocks['httpx_sse'].aconnect_sse.side_effect = Exception("Failed to connect to SSE endpoint")
        
        with pytest.raises(Exception, match="Failed to connect to SSE endpoint"):
            await protocol.connect()
    
    @pytest.mark.asyncio
    async def test_sse_disconnect(self, mock_mcp_dependencies):
        """Test SSE disconnection."""
        mocks = mock_mcp_dependencies
        protocol = MCPProtocol("http://localhost:8080/mcp")
        
        # Connect first
        mocks['client'].initialize.return_value = None
        await protocol.connect()
        
        # Test disconnect
        await protocol.disconnect()
        
        # Verify cleanup
        assert protocol.client is None
        assert protocol.transport is None


class TestMCPProtocolToolOperations:
    """Test MCP protocol tool operations."""
    
    @pytest.mark.asyncio
    async def test_list_tools_success(self, mock_mcp_dependencies):
        """Test successful tool listing."""
        mocks = mock_mcp_dependencies
        protocol = MCPProtocol("uvx weather-mcp@latest stdio")
        
        # Mock tools response
        mock_tools_result = Mock()
        mock_tools_result.tools = [
            Mock(name="get_weather", description="Get weather information", inputSchema={"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}),
            Mock(name="get_forecast", description="Get weather forecast", inputSchema={"type": "object", "properties": {}})
        ]
        mocks['client'].list_tools.return_value = mock_tools_result
        
        # Connect and list tools
        mocks['client'].initialize.return_value = None
        await protocol.connect()
        
        tools = await protocol.list_tools()
        
        # Verify results
        assert len(tools) == 2
        assert tools[0]["name"] == "get_weather"
        assert tools[0]["description"] == "Get weather information"
        assert tools[0]["parameters"] == {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}
        
        assert tools[1]["name"] == "get_forecast"
        assert tools[1]["description"] == "Get weather forecast"
        assert tools[1]["parameters"] == {"type": "object", "properties": {}}
        
        mocks['client'].list_tools.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_list_tools_not_connected(self, mock_mcp_dependencies):
        """Test listing tools when not connected."""
        protocol = MCPProtocol("uvx weather-mcp@latest stdio")
        
        with pytest.raises(RuntimeError, match="Not connected to MCP server"):
            await protocol.list_tools()
    
    @pytest.mark.asyncio
    async def test_call_tool_success(self, mock_mcp_dependencies):
        """Test successful tool call."""
        mocks = mock_mcp_dependencies
        protocol = MCPProtocol("python -m weather_mcp")
        
        # Mock tool call response
        mock_result = Mock()
        mock_result.content = [Mock(type="text", text="Weather in London: 20°C, sunny")]
        mocks['client'].call_tool.return_value = mock_result
        
        # Connect and call tool
        mocks['client'].initialize.return_value = None
        await protocol.connect()
        
        result = await protocol.call_tool("get_weather", {"location": "London"})
        
        # Verify results
        assert result == {"content": "Weather in London: 20°C, sunny"}
        mocks['client'].call_tool.assert_called_once_with("get_weather", {"location": "London"})
    
    @pytest.mark.asyncio
    async def test_call_tool_multiple_content(self, mock_mcp_dependencies):
        """Test tool call with multiple content items."""
        mocks = mock_mcp_dependencies
        protocol = MCPProtocol("uvx weather-mcp@latest stdio")
        
        # Mock tool call response with multiple content items
        mock_result = Mock()
        mock_result.content = [
            Mock(type="text", text="Temperature: 20°C"),
            Mock(type="text", text="Condition: Sunny"),
            Mock(type="image", data="base64encodedimage")
        ]
        mocks['client'].call_tool.return_value = mock_result
        
        # Connect and call tool
        mocks['client'].initialize.return_value = None
        await protocol.connect()
        
        result = await protocol.call_tool("get_weather", {"location": "London"})
        
        # Verify results - should join text content and include other types
        expected_content = "Temperature: 20°C\nCondition: Sunny\n[image data]"
        assert result == {"content": expected_content}
    
    @pytest.mark.asyncio
    async def test_call_tool_not_connected(self, mock_mcp_dependencies):
        """Test calling tool when not connected."""
        protocol = MCPProtocol("uvx weather-mcp@latest stdio")
        
        with pytest.raises(RuntimeError, match="Not connected to MCP server"):
            await protocol.call_tool("get_weather", {"location": "London"})
    
    @pytest.mark.asyncio
    async def test_call_tool_failure(self, mock_mcp_dependencies):
        """Test tool call failure."""
        mocks = mock_mcp_dependencies
        protocol = MCPProtocol("npm run mcp-server")
        
        # Mock tool call failure
        mocks['client'].call_tool.side_effect = Exception("Tool execution failed")
        
        # Connect and attempt tool call
        mocks['client'].initialize.return_value = None
        await protocol.connect()
        
        with pytest.raises(Exception, match="Tool execution failed"):
            await protocol.call_tool("failing_tool", {})


class TestMCPProtocolIntegration:
    """Integration tests for MCP protocol."""
    
    @pytest.mark.asyncio
    async def test_full_workflow(self, mock_mcp_dependencies):
        """Test complete MCP workflow."""
        mocks = mock_mcp_dependencies
        protocol = MCPProtocol("uvx weather-mcp@latest stdio")
        
        # Mock responses
        mocks['client'].initialize.return_value = None
        
        mock_tools_result = Mock()
        mock_tools_result.tools = [
            Mock(name="get_weather", description="Get weather", inputSchema={"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]})
        ]
        mocks['client'].list_tools.return_value = mock_tools_result
        
        mock_call_result = Mock()
        mock_call_result.content = [Mock(type="text", text="Weather: 20°C")]
        mocks['client'].call_tool.return_value = mock_call_result
        
        # Execute workflow
        await protocol.connect()
        
        tools = await protocol.list_tools()
        assert len(tools) == 1
        assert tools[0]["name"] == "get_weather"
        
        result = await protocol.call_tool("get_weather", {"location": "London"})
        assert result["content"] == "Weather: 20°C"
        
        await protocol.disconnect()
        
        # Verify all calls were made
        mocks['client'].initialize.assert_called_once()
        mocks['client'].list_tools.assert_called_once()
        mocks['client'].call_tool.assert_called_once_with("get_weather", {"location": "London"})
    
    @pytest.mark.asyncio
    async def test_connection_resilience(self, mock_mcp_dependencies):
        """Test connection resilience and cleanup."""
        mocks = mock_mcp_dependencies
        protocol = MCPProtocol("python -m weather_mcp")
        
        # Connect successfully
        mocks['client'].initialize.return_value = None
        await protocol.connect()
        assert protocol.client is not None
        
        # Simulate connection loss during operation
        mocks['client'].list_tools.side_effect = Exception("Connection lost")
        
        with pytest.raises(Exception, match="Connection lost"):
            await protocol.list_tools()
        
        # Should still be able to disconnect cleanly
        await protocol.disconnect()
        assert protocol.client is None 