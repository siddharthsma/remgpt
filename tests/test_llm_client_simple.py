"""
Simple tests for the LLM client system core functionality.
"""

import pytest
import asyncio
from remgpt.llm import LLMClientFactory, Event, EventType
from remgpt.tools import ToolExecutor, BaseTool


class MockTool(BaseTool):
    """Mock tool for testing."""
    
    def __init__(self):
        super().__init__("mock_tool", "A mock tool for testing")
    
    async def execute(self, **kwargs):
        return {"result": "mock_result", "args": kwargs}
    
    def get_schema(self):
        return {
            "type": "function",
            "function": {
                "name": "mock_tool",
                "description": "A mock tool for testing",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "test_arg": {
                            "type": "string",
                            "description": "A test argument"
                        }
                    },
                    "required": ["test_arg"]
                }
            }
        }


class TestLLMClientFactory:
    """Test the LLM client factory core functionality."""
    
    def test_get_supported_providers(self):
        """Test getting supported providers."""
        providers = LLMClientFactory.get_supported_providers()
        assert len(providers) > 0
        assert "openai" in providers
        assert "claude" in providers
        assert "gemini" in providers
    
    def test_unsupported_provider(self):
        """Test error for unsupported provider."""
        with pytest.raises(ValueError, match="Unsupported provider"):
            LLMClientFactory.create_client(
                provider="unsupported",
                model_name="test",
                api_key="test"
            )
    
    def test_missing_required_params(self):
        """Test error for missing required parameters."""
        with pytest.raises(TypeError, match="Missing required parameters"):
            LLMClientFactory.create_client(provider="openai")


class TestToolExecutor:
    """Test the tool executor."""
    
    @pytest.fixture
    def executor(self):
        """Create a tool executor with a mock tool."""
        executor = ToolExecutor()
        mock_tool = MockTool()
        executor.register_tool(mock_tool)
        return executor
    
    def test_register_tool(self, executor):
        """Test tool registration."""
        assert "mock_tool" in executor.get_registered_tools()
        assert executor.has_tool("mock_tool")
    
    def test_unregister_tool(self, executor):
        """Test tool unregistration."""
        executor.unregister_tool("mock_tool")
        assert "mock_tool" not in executor.get_registered_tools()
        assert not executor.has_tool("mock_tool")
    
    def test_get_tool_schemas(self, executor):
        """Test getting tool schemas."""
        schemas = executor.get_tool_schemas()
        assert len(schemas) == 1
        assert schemas[0]["function"]["name"] == "mock_tool"
    
    @pytest.mark.asyncio
    async def test_execute_tool(self, executor):
        """Test tool execution."""
        result = await executor.execute_tool(
            tool_call_id="test_id",
            tool_name="mock_tool",
            tool_args={"test_arg": "test_value"}
        )
        
        assert result["result"] == "mock_result"
        assert result["args"]["test_arg"] == "test_value"
        
        # Check stored result
        stored_result = executor.get_tool_result("test_id")
        assert stored_result == result
    
    @pytest.mark.asyncio
    async def test_execute_nonexistent_tool(self, executor):
        """Test executing non-existent tool."""
        with pytest.raises(ValueError, match="Tool 'nonexistent' not registered"):
            await executor.execute_tool(
                tool_call_id="test_id",
                tool_name="nonexistent",
                tool_args={}
            )
    
    @pytest.mark.asyncio
    async def test_execute_multiple_tools(self, executor):
        """Test executing multiple tools concurrently."""
        tool_calls = [
            {
                "id": "call_1",
                "name": "mock_tool",
                "args": {"test_arg": "value_1"}
            },
            {
                "id": "call_2", 
                "name": "mock_tool",
                "args": {"test_arg": "value_2"}
            }
        ]
        
        results = await executor.execute_multiple_tools(tool_calls)
        
        assert len(results) == 2
        assert results["call_1"]["args"]["test_arg"] == "value_1"
        assert results["call_2"]["args"]["test_arg"] == "value_2"
    
    def test_clear_results(self, executor):
        """Test clearing tool results."""
        # Store a result first
        executor._tool_results["test_id"] = {"test": "result"}
        assert executor.get_tool_result("test_id") is not None
        
        # Clear results
        executor.clear_results()
        assert executor.get_tool_result("test_id") is None


class TestEvent:
    """Test the Event class."""
    
    def test_create_basic_event(self):
        """Test creating a basic event."""
        event = Event(type=EventType.TEXT_MESSAGE_START)
        assert event.type == EventType.TEXT_MESSAGE_START
        assert event.data is None
    
    def test_create_content_event(self):
        """Test creating a content event."""
        event = Event(
            type=EventType.TEXT_MESSAGE_CONTENT,
            content="Hello world"
        )
        assert event.type == EventType.TEXT_MESSAGE_CONTENT
        assert event.content == "Hello world"
    
    def test_tool_call_event_validation(self):
        """Test tool call event validation."""
        # Should raise error without tool_call_id
        with pytest.raises(ValueError, match="tool_call_id is required"):
            Event(type=EventType.TOOL_CALL_START)
        
        # Should raise error without tool_name for TOOL_CALL_START
        with pytest.raises(ValueError, match="tool_name is required"):
            Event(
                type=EventType.TOOL_CALL_START,
                tool_call_id="test_id"
            )
        
        # Should raise error without tool_args for TOOL_CALL_ARGS
        with pytest.raises(ValueError, match="tool_args is required"):
            Event(
                type=EventType.TOOL_CALL_ARGS,
                tool_call_id="test_id"
            )
        
        # Should raise error without error for RUN_ERROR
        with pytest.raises(ValueError, match="error is required"):
            Event(type=EventType.RUN_ERROR)
    
    def test_valid_tool_call_events(self):
        """Test creating valid tool call events."""
        # Valid TOOL_CALL_START
        start_event = Event(
            type=EventType.TOOL_CALL_START,
            tool_call_id="test_id",
            tool_name="test_tool"
        )
        assert start_event.tool_call_id == "test_id"
        assert start_event.tool_name == "test_tool"
        
        # Valid TOOL_CALL_ARGS
        args_event = Event(
            type=EventType.TOOL_CALL_ARGS,
            tool_call_id="test_id",
            tool_args={"arg1": "value1"}
        )
        assert args_event.tool_call_id == "test_id" 
        assert args_event.tool_args == {"arg1": "value1"}
        
        # Valid TOOL_CALL_END
        end_event = Event(
            type=EventType.TOOL_CALL_END,
            tool_call_id="test_id"
        )
        assert end_event.tool_call_id == "test_id"


def test_basic_integration():
    """Test basic integration between factory and tools."""
    # Test factory functionality
    factory = LLMClientFactory()
    providers = factory.get_supported_providers()
    
    assert len(providers) > 0
    assert "openai" in providers
    
    # Test tool executor
    executor = ToolExecutor()
    mock_tool = MockTool()
    executor.register_tool(mock_tool)
    
    schemas = executor.get_tool_schemas()
    assert len(schemas) == 1
    assert schemas[0]["function"]["name"] == "mock_tool" 