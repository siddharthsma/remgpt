"""
Integration tests for LLM Client + Tool Executor + Orchestrator.
"""

import pytest
import asyncio
import json
from unittest.mock import AsyncMock, Mock
from typing import AsyncGenerator

from remgpt import (
    create_context_manager,
    ConversationOrchestrator,
    UserMessage,
    AssistantMessage,
    ToolMessage,
    BaseLLMClient,
    Event,
    EventType,
    ToolExecutor,
    BaseTool
)


class CalculatorTool(BaseTool):
    """Test calculator tool."""
    
    def __init__(self):
        super().__init__("calculator", "Perform basic arithmetic")
    
    async def execute(self, operation: str, a: float, b: float) -> dict:
        """Execute calculation."""
        if operation == "add":
            return {"result": a + b, "operation": operation}
        elif operation == "multiply":
            return {"result": a * b, "operation": operation}
        else:
            return {"error": f"Unknown operation: {operation}"}
    
    def get_schema(self) -> dict:
        """Get tool schema."""
        return {
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "Perform basic arithmetic",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "operation": {"type": "string", "enum": ["add", "multiply"]},
                        "a": {"type": "number"},
                        "b": {"type": "number"}
                    },
                    "required": ["operation", "a", "b"]
                }
            }
        }


class MockLLMClientWithTools(BaseLLMClient):
    """Mock LLM client that can make tool calls."""
    
    def __init__(self, **kwargs):
        super().__init__("mock-model", **kwargs)
        self.should_call_tool = False
        self.tool_call_data = None
        
    def set_tool_call(self, should_call: bool, tool_name: str = None, tool_args: dict = None):
        """Configure whether this client should make a tool call."""
        self.should_call_tool = should_call
        self.tool_call_data = {"name": tool_name, "args": tool_args}
    
    async def generate_stream(self, messages: list, **kwargs) -> AsyncGenerator[Event, None]:
        """Generate mock streaming response with optional tool calls."""
        yield Event(type=EventType.RUN_STARTED)
        
        if self.should_call_tool and self.tool_call_data:
            # Simulate tool call
            tool_call_id = "test_call_1"
            
            yield Event(
                type=EventType.TOOL_CALL_START,
                tool_call_id=tool_call_id,
                tool_name=self.tool_call_data["name"]
            )
            
            yield Event(
                type=EventType.TOOL_CALL_ARGS,
                tool_call_id=tool_call_id,
                tool_args=self.tool_call_data["args"]
            )
            
            yield Event(
                type=EventType.TOOL_CALL_END,
                tool_call_id=tool_call_id
            )
            
            # Simulate response after tool execution
            yield Event(
                type=EventType.TEXT_MESSAGE_CONTENT,
                content="I calculated the result using the calculator tool."
            )
        else:
            # Regular text response
            yield Event(
                type=EventType.TEXT_MESSAGE_CONTENT,
                content="Hello! I'm a helpful assistant."
            )
        
        yield Event(type=EventType.RUN_FINISHED)
    
    def send_tool_result(self, tool_call_id: str, result: any) -> None:
        """Mock tool result handling."""
        pass
    
    def supports_tools(self) -> bool:
        return True
    
    def get_supported_models(self) -> list:
        return ["mock-model"]


@pytest.mark.asyncio
async def test_orchestrator_with_llm_client_no_tools():
    """Test orchestrator with LLM client but no tool calls."""
    # Set up context manager
    context_manager = create_context_manager(
        max_tokens=1000,
        system_instructions="You are a helpful assistant."
    )
    
    # Create LLM client
    llm_client = MockLLMClientWithTools()
    
    # Create tool executor (empty for this test)
    tool_executor = ToolExecutor()
    
    # Create orchestrator
    orchestrator = ConversationOrchestrator(
        context_manager=context_manager,
        llm_client=llm_client,
        tool_executor=tool_executor
    )
    
    # Process a simple message
    user_message = UserMessage(content="Hello there!")
    
    events = []
    async for event in orchestrator.process_message(user_message):
        events.append(event)
    
    # Verify we got events
    assert len(events) > 0
    
    # Should have response events
    response_chunks = [e for e in events if e.type == EventType.TEXT_MESSAGE_CONTENT]
    assert len(response_chunks) > 0
    assert "helpful assistant" in response_chunks[0].content


@pytest.mark.asyncio
async def test_orchestrator_with_tool_execution():
    """Test orchestrator with LLM client making tool calls."""
    # Set up context manager
    context_manager = create_context_manager(
        max_tokens=1000,
        system_instructions="You are a helpful assistant with calculator access."
    )
    
    # Create LLM client configured to make tool calls
    llm_client = MockLLMClientWithTools()
    llm_client.set_tool_call(
        should_call=True,
        tool_name="calculator",
        tool_args={"operation": "add", "a": 5, "b": 3}
    )
    
    # Create tool executor with calculator
    tool_executor = ToolExecutor()
    calculator_tool = CalculatorTool()
    tool_executor.register_tool(calculator_tool)
    
    # Create orchestrator
    orchestrator = ConversationOrchestrator(
        context_manager=context_manager,
        llm_client=llm_client,
        tool_executor=tool_executor
    )
    
    # Process a message that should trigger tool use
    user_message = UserMessage(content="What is 5 + 3?")
    
    events = []
    async for event in orchestrator.process_message(user_message):
        events.append(event)
    
    # Verify we got events
    assert len(events) > 0
    
    # Should have function call events
    function_calls = [e for e in events if e.type == EventType.TOOL_CALL_START]
    assert len(function_calls) > 0
    assert function_calls[0].tool_name == "calculator"
    
    # Should have function result events (using CUSTOM events with tool_result subtype)
    function_results = [e for e in events if e.type == EventType.CUSTOM and e.data.get("event_subtype") == "tool_result"]
    assert len(function_results) > 0
    assert function_results[0].data["success"] is True
    assert function_results[0].data["result"]["result"] == 8  # 5 + 3
    
    # Should have response content after tool execution
    response_chunks = [e for e in events if e.type == EventType.TEXT_MESSAGE_CONTENT]
    assert len(response_chunks) > 0


@pytest.mark.asyncio
async def test_tool_executor_concurrent_execution():
    """Test tool executor can handle multiple concurrent tool calls."""
    tool_executor = ToolExecutor()
    calculator_tool = CalculatorTool()
    tool_executor.register_tool(calculator_tool)
    
    # Execute multiple tools concurrently
    tool_calls = [
        {"id": "call_1", "name": "calculator", "args": {"operation": "add", "a": 1, "b": 2}},
        {"id": "call_2", "name": "calculator", "args": {"operation": "multiply", "a": 3, "b": 4}},
        {"id": "call_3", "name": "calculator", "args": {"operation": "add", "a": 10, "b": 20}}
    ]
    
    results = await tool_executor.execute_multiple_tools(tool_calls)
    
    # Verify results
    assert len(results) == 3
    assert results["call_1"]["result"] == 3  # 1 + 2
    assert results["call_2"]["result"] == 12  # 3 * 4
    assert results["call_3"]["result"] == 30  # 10 + 20


@pytest.mark.asyncio
async def test_tool_execution_error_handling():
    """Test tool execution error handling."""
    tool_executor = ToolExecutor()
    calculator_tool = CalculatorTool()
    tool_executor.register_tool(calculator_tool)
    
    # Execute with invalid operation (should return error result, not raise)
    result = await tool_executor.execute_tool(
        tool_call_id="error_call",
        tool_name="calculator",
        tool_args={"operation": "divide", "a": 5, "b": 0}  # Invalid operation
    )
    
    # Result should contain error
    assert result is not None
    assert "error" in result
    assert "Unknown operation" in result["error"]
    
    # Check stored result
    stored_result = tool_executor.get_tool_result("error_call")
    assert stored_result == result


@pytest.mark.asyncio 
async def test_context_management_integration():
    """Test that context management works with the full system."""
    # Set up context manager with small token limit to trigger warnings
    context_manager = create_context_manager(
        max_tokens=100,  # Very small to trigger token warnings
        system_instructions="You are a helpful assistant."
    )
    
    # Create LLM client
    llm_client = MockLLMClientWithTools()
    
    # Create tool executor
    tool_executor = ToolExecutor()
    
    # Create orchestrator
    orchestrator = ConversationOrchestrator(
        context_manager=context_manager,
        llm_client=llm_client,
        tool_executor=tool_executor
    )
    
    # Add several messages to fill up context
    messages = [
        "Hello, how are you today?",
        "Tell me about artificial intelligence and machine learning.",
        "What are the benefits of using neural networks?",
        "Can you explain deep learning architectures?",
        "How do transformers work in natural language processing?"
    ]
    
    all_events = []
    for msg_content in messages:
        user_message = UserMessage(content=msg_content)
        async for event in orchestrator.process_message(user_message):
            all_events.append(event)
    
    # Should have processed multiple messages
    assert len(all_events) > 0
    
    # Check context summary
    context_summary = context_manager.get_context_summary()
    assert context_summary["total_tokens"] > 0
    assert context_summary["queue_size"] > 0


def test_llm_client_message_formatting():
    """Test message formatting for LLM clients."""
    from remgpt.orchestration.orchestrator import ConversationOrchestrator
    
    # Create minimal orchestrator to test message formatting
    context_manager = create_context_manager(max_tokens=1000)
    orchestrator = ConversationOrchestrator(context_manager=context_manager)
    
    # Test different message types
    messages = [
        UserMessage(content="Hello"),
        AssistantMessage(content="Hi there!"),
        {"role": "system", "content": "You are helpful"},
        "Plain string message"
    ]
    
    formatted = orchestrator._format_messages_for_llm(messages)
    
    assert len(formatted) == 4
    assert formatted[0]["role"] == "user"
    assert formatted[0]["content"] == "Hello"
    assert formatted[1]["role"] == "assistant"
    assert formatted[1]["content"] == "Hi there!"
    assert formatted[2]["role"] == "system"
    assert formatted[3]["role"] == "user"  # String gets converted to user message


@pytest.mark.asyncio
async def test_full_conversation_flow():
    """Test a complete conversation flow with tool calls."""
    # Set up the full system
    context_manager = create_context_manager(
        max_tokens=2000,
        system_instructions="You are a helpful assistant with calculator access.",
        tools=[CalculatorTool().get_schema()]
    )
    
    llm_client = MockLLMClientWithTools()
    tool_executor = ToolExecutor()
    tool_executor.register_tool(CalculatorTool())
    
    orchestrator = ConversationOrchestrator(
        context_manager=context_manager,
        llm_client=llm_client,
        tool_executor=tool_executor
    )
    
    # Conversation flow
    conversations = [
        ("Hello!", False),  # No tool call
        ("What is 7 times 8?", True),  # Should trigger tool call
        ("Thank you!", False)  # No tool call
    ]
    
    all_events = []
    for msg_content, should_have_tools in conversations:
        # Configure LLM client for this message
        if should_have_tools:
            llm_client.set_tool_call(
                should_call=True,
                tool_name="calculator", 
                tool_args={"operation": "multiply", "a": 7, "b": 8}
            )
        else:
            llm_client.set_tool_call(should_call=False)
        
        user_message = UserMessage(content=msg_content)
        message_events = []
        async for event in orchestrator.process_message(user_message):
            message_events.append(event)
        
        all_events.extend(message_events)
        
        # Verify tool calls when expected
        if should_have_tools:
            function_calls = [e for e in message_events if e.type == EventType.TOOL_CALL_START]
            assert len(function_calls) > 0, f"Expected tool call for: {msg_content}"
        else:
            function_calls = [e for e in message_events if e.type == EventType.TOOL_CALL_START]
            assert len(function_calls) == 0, f"Unexpected tool call for: {msg_content}"
    
    # Verify we processed all messages
    assert len(all_events) > 0
    
    # Check final context state
    context_summary = context_manager.get_context_summary()
    assert context_summary["queue_size"] >= len(conversations) * 2  # User + assistant messages 