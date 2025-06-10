"""
Unit tests for ToolExecutor class.
"""

import pytest
from unittest.mock import AsyncMock, Mock
from remgpt.tools.executor import ToolExecutor
from remgpt.tools.base import BaseTool


class MockTool(BaseTool):
    """Mock tool for testing."""
    
    def __init__(self, name="mock_tool", description="Mock tool for testing", should_fail=False):
        super().__init__(name, description)
        self.should_fail = should_fail
        self.execution_count = 0
        self.last_args = None
    
    async def execute(self, **kwargs):
        """Mock execute implementation."""
        self.execution_count += 1
        self.last_args = kwargs
        
        if self.should_fail:
            raise ValueError("Mock tool execution failed")
        
        return {"tool": self.name, "args": kwargs, "count": self.execution_count}
    
    def get_schema(self):
        """Mock get_schema implementation."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        }
    
    def validate_args(self, args):
        """Mock validation - fails if 'invalid' key is present."""
        return "invalid" not in args


class TestToolExecutor:
    """Test cases for ToolExecutor class."""
    
    def test_initialization(self):
        """Test ToolExecutor initialization."""
        executor = ToolExecutor()
        assert len(executor._tools) == 0
        assert len(executor.get_registered_tools()) == 0
    
    def test_register_tool(self):
        """Test tool registration."""
        executor = ToolExecutor()
        tool = MockTool("test_tool", "Test description")
        
        # Register tool
        executor.register_tool(tool)
        
        # Verify tool is registered
        assert len(executor._tools) == 1
        assert "test_tool" in executor._tools
        assert executor.has_tool("test_tool") is True
    
    def test_register_multiple_tools(self):
        """Test registering multiple tools."""
        executor = ToolExecutor()
        tool1 = MockTool("tool1", "First tool")
        tool2 = MockTool("tool2", "Second tool")
        
        executor.register_tool(tool1)
        executor.register_tool(tool2)
        
        assert len(executor._tools) == 2
        assert executor.has_tool("tool1") is True
        assert executor.has_tool("tool2") is True
    
    def test_register_duplicate_tool(self):
        """Test registering tool with duplicate name."""
        executor = ToolExecutor()
        tool1 = MockTool("duplicate", "First tool")
        tool2 = MockTool("duplicate", "Second tool")
        
        executor.register_tool(tool1)
        
        # Registering duplicate should replace the first tool
        executor.register_tool(tool2)
        
        assert len(executor._tools) == 1
        assert executor.has_tool("duplicate") is True
        assert executor._tools["duplicate"].description == "Second tool"
    
    def test_unregister_tool(self):
        """Test tool unregistration."""
        executor = ToolExecutor()
        tool = MockTool("test_tool", "Test description")
        
        # Register and then unregister
        executor.register_tool(tool)
        assert len(executor._tools) == 1
        
        executor.unregister_tool("test_tool")
        assert len(executor._tools) == 0
        assert executor.has_tool("test_tool") is False
    
    def test_unregister_nonexistent_tool(self):
        """Test unregistering a tool that doesn't exist."""
        executor = ToolExecutor()
        # Should not raise an error, just do nothing
        executor.unregister_tool("nonexistent")
        assert len(executor._tools) == 0
    
    def test_has_tool_nonexistent(self):
        """Test checking for a tool that doesn't exist."""
        executor = ToolExecutor()
        has_tool = executor.has_tool("nonexistent")
        assert has_tool is False
    
    def test_list_tools(self):
        """Test listing available tools."""
        executor = ToolExecutor()
        tool1 = MockTool("tool1", "First tool")
        tool2 = MockTool("tool2", "Second tool")
        
        executor.register_tool(tool1)
        executor.register_tool(tool2)
        
        tool_names = executor.get_registered_tools()
        assert len(tool_names) == 2
        
        # Check that tool names are in the list
        assert "tool1" in tool_names
        assert "tool2" in tool_names
    
    def test_get_tool_schemas(self):
        """Test getting tool schemas for LLM function calling."""
        executor = ToolExecutor()
        tool1 = MockTool("tool1", "First tool")
        tool2 = MockTool("tool2", "Second tool")
        
        executor.register_tool(tool1)
        executor.register_tool(tool2)
        
        schemas = executor.get_tool_schemas()
        assert len(schemas) == 2
        
        # Check schema structure
        for schema in schemas:
            assert "type" in schema
            assert schema["type"] == "function"
            assert "function" in schema
            assert "name" in schema["function"]
            assert "description" in schema["function"]
            assert "parameters" in schema["function"]
    
    @pytest.mark.asyncio
    async def test_execute_tool_success(self):
        """Test successful tool execution."""
        executor = ToolExecutor()
        tool = MockTool("test_tool", "Test description")
        executor.register_tool(tool)
        
        # Execute tool
        result = await executor.execute_tool("call_1", "test_tool", {"param1": "value1"})
        
        # Verify execution
        assert result["tool"] == "test_tool"
        assert result["args"] == {"param1": "value1"}
        assert result["count"] == 1
        assert tool.execution_count == 1
        assert tool.last_args == {"param1": "value1"}
    
    @pytest.mark.asyncio
    async def test_execute_tool_not_found(self):
        """Test executing a tool that doesn't exist."""
        executor = ToolExecutor()
        
        with pytest.raises(ValueError, match="Tool 'nonexistent' not registered"):
            await executor.execute_tool("call_1", "nonexistent", {})
    
    @pytest.mark.asyncio
    async def test_execute_tool_validation_failure(self):
        """Test tool execution with validation failure."""
        executor = ToolExecutor()
        tool = MockTool("test_tool", "Test description")
        executor.register_tool(tool)
        
        # Execute with invalid arguments
        with pytest.raises(ValueError, match="Invalid arguments for tool test_tool"):
            await executor.execute_tool("call_1", "test_tool", {"invalid": "argument"})
    
    @pytest.mark.asyncio
    async def test_execute_tool_execution_failure(self):
        """Test tool execution failure."""
        executor = ToolExecutor()
        tool = MockTool("failing_tool", "Tool that fails", should_fail=True)
        executor.register_tool(tool)
        
        # Execute tool that will fail
        with pytest.raises(ValueError, match="Mock tool execution failed"):
            await executor.execute_tool("call_1", "failing_tool", {})
    
    @pytest.mark.asyncio
    async def test_execute_tool_with_result_storage(self):
        """Test tool execution with result storage."""
        executor = ToolExecutor()
        tool = MockTool("test_tool", "Test description")
        executor.register_tool(tool)
        
        # Execute tool
        result = await executor.execute_tool("call_1", "test_tool", {"param": "value"})
        
        # Check that result was stored
        stored_result = executor.get_tool_result("call_1")
        assert stored_result == result
        assert stored_result["tool"] == "test_tool"
    
    def test_empty_executor_properties(self):
        """Test properties of empty executor."""
        executor = ToolExecutor()
        
        assert len(executor._tools) == 0
        assert len(executor.get_registered_tools()) == 0
        assert len(executor.get_tool_schemas()) == 0
    
    def test_tool_registration_order(self):
        """Test that tools maintain registration order."""
        executor = ToolExecutor()
        tools = [
            MockTool("tool_a", "Tool A"),
            MockTool("tool_b", "Tool B"),
            MockTool("tool_c", "Tool C")
        ]
        
        for tool in tools:
            executor.register_tool(tool)
        
        registered_tool_names = executor.get_registered_tools()
        expected_names = [tool.name for tool in tools]
        assert registered_tool_names == expected_names


class TestToolExecutorIntegration:
    """Integration tests for ToolExecutor."""
    
    @pytest.mark.asyncio
    async def test_multiple_tool_execution(self):
        """Test executing multiple tools in sequence."""
        executor = ToolExecutor()
        
        tool1 = MockTool("calculator", "Performs calculations")
        tool2 = MockTool("formatter", "Formats output")
        
        executor.register_tool(tool1)
        executor.register_tool(tool2)
        
        # Execute first tool
        result1 = await executor.execute_tool("call_1", "calculator", {"operation": "add", "a": 5, "b": 3})
        assert result1["tool"] == "calculator"
        assert result1["count"] == 1
        
        # Execute second tool
        result2 = await executor.execute_tool("call_2", "formatter", {"data": result1})
        assert result2["tool"] == "formatter"
        assert result2["count"] == 1
        
        # Execute first tool again
        result3 = await executor.execute_tool("call_3", "calculator", {"operation": "multiply", "a": 2, "b": 4})
        assert result3["tool"] == "calculator"
        assert result3["count"] == 2  # Second execution of calculator
    
    @pytest.mark.asyncio
    async def test_tool_replacement_execution(self):
        """Test that replaced tools execute correctly."""
        executor = ToolExecutor()
        
        # Register initial tool
        old_tool = MockTool("processor", "Old processor")
        executor.register_tool(old_tool)
        
        # Execute old tool
        result1 = await executor.execute_tool("call_1", "processor", {"version": "old"})
        assert result1["args"]["version"] == "old"
        
        # Replace with new tool
        new_tool = MockTool("processor", "New processor")
        executor.register_tool(new_tool)
        
        # Execute new tool
        result2 = await executor.execute_tool("call_2", "processor", {"version": "new"})
        assert result2["args"]["version"] == "new"
        assert new_tool.execution_count == 1
        assert old_tool.execution_count == 1  # Old tool was only executed once 