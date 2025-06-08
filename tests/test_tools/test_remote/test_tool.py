"""
Unit tests for RemoteTool adapter class.
"""

import pytest
from unittest.mock import AsyncMock, Mock
from remgpt.tools.remote.tool import RemoteTool
from remgpt.tools.remote.base import RemoteToolProtocol


class MockProtocol(RemoteToolProtocol):
    """Mock protocol for testing RemoteTool."""
    
    def __init__(self, connection_string="mock://test"):
        self.connection_string = connection_string
        self.is_connected = False
        self.tool_schemas = []
        self.call_results = {}
    
    async def connect(self):
        """Mock connect."""
        self.is_connected = True
    
    async def disconnect(self):
        """Mock disconnect."""
        self.is_connected = False
    
    async def list_tools(self):
        """Mock list_tools."""
        if not self.is_connected:
            raise RuntimeError("Not connected")
        return self.tool_schemas
    
    async def call_tool(self, name, arguments):
        """Mock call_tool."""
        if not self.is_connected:
            raise RuntimeError("Not connected")
        
        if name in self.call_results:
            return self.call_results[name]
        else:
            raise ValueError(f"Unknown tool: {name}")


class TestRemoteToolInitialization:
    """Test RemoteTool initialization."""
    
    def test_initialization_with_all_parameters(self):
        """Test RemoteTool initialization with all parameters."""
        protocol = MockProtocol()
        tool_schema = {
            "name": "calculator",
            "description": "Perform calculations",
            "parameters": {
                "type": "object", 
                "properties": {
                    "operation": {"type": "string"},
                    "a": {"type": "number"},
                    "b": {"type": "number"}
                },
                "required": ["operation", "a", "b"]
            }
        }
        
        tool = RemoteTool(tool_schema, protocol, "test")
        
        assert tool.name == "test_calculator"
        assert tool.description == "Perform calculations"
        assert tool.protocol == protocol
        assert tool.tool_info == tool_schema
    
    def test_initialization_minimal_schema(self):
        """Test initialization with minimal schema."""
        protocol = MockProtocol()
        tool_schema = {
            "name": "simple_tool",
            "description": "A simple tool"
        }
        
        tool = RemoteTool(tool_schema, protocol, "test")
        
        assert tool.name == "test_simple_tool"
        assert tool.description == "A simple tool"
        assert tool.protocol == protocol
        assert tool.tool_info == tool_schema
    
    def test_initialization_missing_name(self):
        """Test initialization with missing name in schema."""
        protocol = MockProtocol()
        tool_schema = {"description": "Tool without name"}
        
        with pytest.raises(KeyError):
            RemoteTool(tool_schema, protocol, "test")
    
    def test_initialization_missing_description(self):
        """Test initialization with missing description in schema."""
        protocol = MockProtocol()
        tool_schema = {"name": "tool_without_description"}
        
        with pytest.raises(KeyError):
            RemoteTool(tool_schema, protocol, "test")


class TestRemoteToolSchemaGeneration:
    """Test RemoteTool schema generation for function calling."""
    
    def test_get_schema_with_parameters(self):
        """Test schema generation with parameters."""
        protocol = MockProtocol()
        tool_schema = {
            "name": "weather",
            "description": "Get weather information",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"},
                    "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["location"]
            }
        }
        
        tool = RemoteTool(tool_schema, protocol, "test")
        schema = tool.get_schema()
        
        expected = {
            "type": "function",
            "function": {
                "name": "test_weather",
                "description": "Get weather information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"},
                        "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                    },
                    "required": ["location"]
                }
            }
        }
        
        assert schema == expected
    
    def test_get_schema_without_parameters(self):
        """Test schema generation without parameters."""
        protocol = MockProtocol()
        tool_schema = {
            "name": "status",
            "description": "Get system status"
        }
        
        tool = RemoteTool(tool_schema, protocol, "test")
        schema = tool.get_schema()
        
        expected = {
            "type": "function",
            "function": {
                "name": "test_status",
                "description": "Get system status",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        }
        
        assert schema == expected
    
    def test_get_schema_empty_parameters(self):
        """Test schema generation with empty parameters."""
        protocol = MockProtocol()
        tool_schema = {
            "name": "ping",
            "description": "Ping the service",
            "inputSchema": {}
        }
        
        tool = RemoteTool(tool_schema, protocol, "test")
        schema = tool.get_schema()
        
        expected = {
            "type": "function",
            "function": {
                "name": "test_ping",
                "description": "Ping the service",
                "parameters": {}
            }
        }
        
        assert schema == expected


class TestRemoteToolExecution:
    """Test RemoteTool execution."""
    
    @pytest.mark.asyncio
    async def test_execute_success(self):
        """Test successful tool execution."""
        protocol = MockProtocol()
        protocol.call_results["calculator"] = {"result": "8"}
        
        tool_schema = {
            "name": "calculator",
            "description": "Perform calculations",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {"type": "string"},
                    "a": {"type": "number"},
                    "b": {"type": "number"}
                },
                "required": ["operation", "a", "b"]
            }
        }
        
        tool = RemoteTool(tool_schema, protocol, "test")
        
        # Connect protocol first
        await protocol.connect()
        
        # Execute tool
        result = await tool.execute(operation="add", a=5, b=3)
        
        assert result == {"result": "8"}
    
    @pytest.mark.asyncio
    async def test_execute_no_arguments(self):
        """Test tool execution with no arguments."""
        protocol = MockProtocol()
        protocol.call_results["status"] = {"status": "healthy", "uptime": "2h 30m"}
        
        tool_schema = {
            "name": "status",
            "description": "Get system status"
        }
        
        tool = RemoteTool(tool_schema, protocol, "test")
        
        await protocol.connect()
        result = await tool.execute()
        
        assert result == {"status": "healthy", "uptime": "2h 30m"}
    
    @pytest.mark.asyncio
    async def test_execute_protocol_not_connected(self):
        """Test execution when protocol is not connected."""
        protocol = MockProtocol()
        tool_schema = {
            "name": "test_tool",
            "description": "Test tool"
        }
        
        tool = RemoteTool(tool_schema, protocol, "test")
        
        # Don't connect protocol
        result = await tool.execute()
        assert "error" in result
        assert "Not connected" in result["error"]
    
    @pytest.mark.asyncio
    async def test_execute_tool_not_found(self):
        """Test execution when tool is not found on remote."""
        protocol = MockProtocol()
        tool_schema = {
            "name": "nonexistent_tool",
            "description": "Tool that doesn't exist"
        }
        
        tool = RemoteTool(tool_schema, protocol, "test")
        
        await protocol.connect()
        
        result = await tool.execute()
        assert "error" in result
        assert "Unknown tool: nonexistent_tool" in result["error"]
    
    @pytest.mark.asyncio
    async def test_execute_with_mixed_arguments(self):
        """Test execution with various argument types."""
        protocol = MockProtocol()
        protocol.call_results["complex_tool"] = {
            "processed": True,
            "data": {"input_count": 3, "output_size": 1024}
        }
        
        tool_schema = {
            "name": "complex_tool",
            "description": "A complex tool with various parameters",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "number": {"type": "integer"},
                    "flag": {"type": "boolean"},
                    "items": {"type": "array"}
                }
            }
        }
        
        tool = RemoteTool(tool_schema, protocol, "test")
        
        await protocol.connect()
        result = await tool.execute(
            text="hello world",
            number=42,
            flag=True,
            items=["a", "b", "c"]
        )
        
        expected = {
            "processed": True,
            "data": {"input_count": 3, "output_size": 1024}
        }
        assert result == expected


class TestRemoteToolValidation:
    """Test RemoteTool argument validation."""
    
    def test_validate_args_with_required_parameters(self):
        """Test validation with required parameters."""
        protocol = MockProtocol()
        tool_schema = {
            "name": "api_call",
            "description": "Make API call",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "endpoint": {"type": "string"},
                    "method": {"type": "string"},
                    "headers": {"type": "object"}
                },
                "required": ["endpoint", "method"]
            }
        }
        
        tool = RemoteTool(tool_schema, protocol, "test")
        
        # Valid arguments
        assert tool.validate_args({"endpoint": "/api/users", "method": "GET"}) is True
        assert tool.validate_args({
            "endpoint": "/api/users", 
            "method": "POST", 
            "headers": {"Content-Type": "application/json"}
        }) is True
        
        # RemoteTool uses default validation (always True)
        # Missing required parameters still pass default validation
        assert tool.validate_args({"endpoint": "/api/users"}) is True
        assert tool.validate_args({"method": "GET"}) is True
        assert tool.validate_args({}) is True
    
    def test_validate_args_no_required_parameters(self):
        """Test validation with no required parameters."""
        protocol = MockProtocol()
        tool_schema = {
            "name": "optional_tool",
            "description": "Tool with optional parameters",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "option1": {"type": "string"},
                    "option2": {"type": "integer"}
                }
            }
        }
        
        tool = RemoteTool(tool_schema, protocol, "test")
        
        # All combinations should be valid
        assert tool.validate_args({}) is True
        assert tool.validate_args({"option1": "value"}) is True
        assert tool.validate_args({"option2": 123}) is True
        assert tool.validate_args({"option1": "value", "option2": 123}) is True
    
    def test_validate_args_no_parameters_defined(self):
        """Test validation with no parameters defined."""
        protocol = MockProtocol()
        tool_schema = {
            "name": "simple_tool",
            "description": "Simple tool with no parameters"
        }
        
        tool = RemoteTool(tool_schema, protocol, "test")
        
        # Should accept empty arguments
        assert tool.validate_args({}) is True
        
        # Should also accept any arguments (default behavior)
        assert tool.validate_args({"unexpected": "argument"}) is True


class TestRemoteToolIntegration:
    """Integration tests for RemoteTool."""
    
    @pytest.mark.asyncio
    async def test_complete_workflow(self):
        """Test complete workflow from schema to execution."""
        protocol = MockProtocol()
        
        # Setup protocol with tool data
        tool_data = {
            "name": "file_reader",
            "description": "Read file contents",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"},
                    "encoding": {"type": "string", "default": "utf-8"}
                },
                "required": ["path"]
            }
        }
        
        protocol.tool_schemas = [tool_data]
        protocol.call_results["file_reader"] = {
            "content": "Hello, World!",
            "size": 13,
            "encoding": "utf-8"
        }
        
        # Create and test tool
        tool = RemoteTool(tool_data, protocol, "test")
        
        # Test schema generation
        schema = tool.get_schema()
        assert schema["function"]["name"] == "test_file_reader"
        assert "path" in schema["function"]["parameters"]["properties"]
        
        # Test validation (RemoteTool uses default validation)
        assert tool.validate_args({"path": "/tmp/test.txt"}) is True
        assert tool.validate_args({}) is True  # Default validation always passes
        
        # Test execution
        await protocol.connect()
        result = await tool.execute(path="/tmp/test.txt", encoding="utf-8")
        
        expected = {
            "content": "Hello, World!",
            "size": 13,
            "encoding": "utf-8"
        }
        assert result == expected
        
        await protocol.disconnect()
    
    @pytest.mark.asyncio
    async def test_error_propagation(self):
        """Test that errors from protocol are properly propagated."""
        protocol = MockProtocol()
        tool_schema = {
            "name": "error_tool",
            "description": "Tool that causes errors"
        }
        
        tool = RemoteTool(tool_schema, protocol, "test")
        
        # Test connection error propagation
        result = await tool.execute()  # Protocol not connected
        assert "error" in result
        assert "Not connected" in result["error"]
        
        # Test tool execution error propagation
        await protocol.connect()
        result = await tool.execute()  # Tool not configured in protocol
        assert "error" in result
        assert "Unknown tool" in result["error"]
    
    def test_tool_representation(self):
        """Test string representation of RemoteTool."""
        protocol = MockProtocol("test://connection")
        tool_schema = {
            "name": "test_tool",
            "description": "Test tool for representation"
        }
        
        tool = RemoteTool(tool_schema, protocol, "test")
        
        # Test that tool has reasonable string representation
        tool_str = str(tool)
        assert "test_test_tool" in tool_str or "RemoteTool" in tool_str
        
        # Test name and description are accessible
        assert tool.name == "test_test_tool"
        assert tool.description == "Test tool for representation" 