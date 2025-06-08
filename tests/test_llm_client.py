"""
Tests for the LLM client system.
"""

import pytest
import asyncio
import sys
from unittest.mock import Mock, patch, AsyncMock
from remgpt.llm import (
    BaseLLMClient, LLMClientFactory, OpenAIClient, 
    ClaudeClient, GeminiClient, Event, EventType
)
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
    """Test the LLM client factory."""
    
    def test_get_supported_providers(self):
        """Test getting supported providers."""
        providers = LLMClientFactory.get_supported_providers()
        assert "openai" in providers
        assert "claude" in providers
        assert "gemini" in providers
    
    def test_create_openai_client(self):
        """Test creating OpenAI client."""
        # Mock the openai module
        mock_openai = Mock()
        mock_openai.OpenAI = Mock()
        
        with patch.dict('sys.modules', {'openai': mock_openai}):
            client = LLMClientFactory.create_client(
                provider="openai",
                model_name="gpt-4",
                api_key="test_key"
            )
            assert isinstance(client, OpenAIClient)
            assert client.model_name == "gpt-4"
    
    def test_create_claude_client(self):
        """Test creating Claude client."""
        with patch('anthropic.Anthropic'):
            client = LLMClientFactory.create_client(
                provider="claude",
                model_name="claude-3-sonnet-20240229",
                api_key="test_key"
            )
            assert isinstance(client, ClaudeClient)
            assert client.model_name == "claude-3-sonnet-20240229"
    
    def test_create_gemini_client(self):
        """Test creating Gemini client."""
        with patch('google.generativeai.configure'), \
             patch('google.generativeai.GenerativeModel'):
            client = LLMClientFactory.create_client(
                provider="gemini",
                model_name="gemini-1.5-pro",
                api_key="test_key"
            )
            assert isinstance(client, GeminiClient)
            assert client.model_name == "gemini-1.5-pro"
    
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


def test_integration_example():
    """Test that shows how to use the LLM client system."""
    # This would be how to use the system in practice
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"}
    ]
    
    # Create factory and test that we can create clients
    factory = LLMClientFactory()
    providers = factory.get_supported_providers()
    
    assert len(providers) > 0
    assert "openai" in providers
    
    # Test validation
    with patch('openai.OpenAI'):
        client = factory.create_client(
            provider="openai",
            model_name="gpt-4",
            api_key="test_key"
        )
        
        # Test message validation
        assert client.validate_messages(messages) == True
        
        # Test message formatting
        formatted = client.format_messages(messages)
        assert len(formatted) == 2
        assert formatted[0]["role"] == "system" 