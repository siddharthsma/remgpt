"""
Example demonstrating the MockLLMClient for testing and development.
"""

import asyncio
import logging
from remgpt import (
    create_context_manager,
    ConversationOrchestrator,
    UserMessage,
    AssistantMessage,
    ToolExecutor
)
from remgpt.llm import MockLLMClient
from remgpt.tools import BaseTool

# Set up logging to see what's happening
logging.basicConfig(level=logging.INFO)


class CalculatorTool(BaseTool):
    """Example calculator tool for testing."""
    
    def __init__(self):
        super().__init__("calculator", "Perform basic arithmetic operations")
    
    async def execute(self, operation: str, a: float, b: float) -> dict:
        """Execute a calculation."""
        if operation == "add":
            result = a + b
        elif operation == "subtract":
            result = a - b
        elif operation == "multiply":
            result = a * b
        elif operation == "divide":
            result = a / b if b != 0 else "Error: Division by zero"
        else:
            return {"error": f"Unknown operation: {operation}"}
        
        return {
            "operation": operation,
            "operands": [a, b],
            "result": result
        }
    
    def get_schema(self) -> dict:
        """Get the tool schema for LLM function calling."""
        return {
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "Perform basic arithmetic operations",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "operation": {
                            "type": "string",
                            "enum": ["add", "subtract", "multiply", "divide"],
                            "description": "The arithmetic operation to perform"
                        },
                        "a": {
                            "type": "number",
                            "description": "First number"
                        },
                        "b": {
                            "type": "number", 
                            "description": "Second number"
                        }
                    },
                    "required": ["operation", "a", "b"]
                }
            }
        }


async def demonstrate_mock_llm_client():
    """Demonstrate MockLLMClient usage."""
    
    print("🤖 MockLLMClient Demonstration")
    print("=" * 50)
    
    # 1. Create MockLLMClient with custom settings
    mock_client = MockLLMClient(
        model_name="mock-gpt-4",
        response_delay=0.05,  # Faster for demo
        simulate_streaming=True
    )
    
    # 2. Set up context manager
    context_manager = create_context_manager(
        max_tokens=2000,
        system_instructions="You are a helpful assistant with access to a calculator."
    )
    
    # 3. Set up tool executor with calculator
    tool_executor = ToolExecutor()
    calculator = CalculatorTool()
    tool_executor.register_tool(calculator)
    
    # 4. Create orchestrator
    orchestrator = ConversationOrchestrator(
        context_manager=context_manager,
        llm_client=mock_client,
        tool_executor=tool_executor
    )
    
    # 5. Register context management tools
    orchestrator._register_context_management_functions()
    
    print("\n📝 Test 1: Simple conversation")
    print("-" * 30)
    
    # Test simple conversation
    user_msg = UserMessage(content="Hello! How are you today?")
    
    async for event in orchestrator.process_message(user_msg):
        if event.type.value == "TEXT_MESSAGE_CONTENT":
            print(f"Assistant: {event.content}", end="", flush=True)
        elif event.type.value == "TEXT_MESSAGE_END":
            print()  # New line after complete message
    
    print("\n📝 Test 2: Topic drift scenario")
    print("-" * 30)
    
    # Simulate topic drift by adding a drift warning message
    drift_warning = AssistantMessage(
        content="TOPIC DRIFT DETECTED: The conversation has shifted to a new topic. I should save the current conversation topic before continuing."
    )
    context_manager.add_message_to_queue(drift_warning)
    
    # Now send a message that should trigger context management
    user_msg2 = UserMessage(content="Let's talk about something completely different - space exploration!")
    
    print("Processing message with topic drift...")
    tool_calls_made = []
    
    async for event in orchestrator.process_message(user_msg2):
        if event.type.value == "TOOL_CALL_START":
            print(f"🔧 Tool call started: {event.tool_name}")
        elif event.type.value == "CUSTOM" and event.data.get("event_subtype") == "tool_result":
            tool_calls_made.append(event.data)
            if event.data.get("success"):
                print(f"✅ Tool '{event.data['function_name']}' succeeded: {event.data.get('result')}")
            else:
                print(f"❌ Tool '{event.data['function_name']}' failed: {event.data.get('error')}")
        elif event.type.value == "TEXT_MESSAGE_CONTENT":
            print(f"Assistant: {event.content}", end="", flush=True)
        elif event.type.value == "TEXT_MESSAGE_END":
            print()
    
    print(f"\nTotal tool calls made: {len(tool_calls_made)}")
    
    print("\n📝 Test 3: Token limit scenario")
    print("-" * 30)
    
    # Simulate approaching token limit
    token_warning = AssistantMessage(
        content="APPROACHING TOKEN LIMIT: My context is getting full (70%+ of limit). I should consider evicting an old topic to make room for new content."
    )
    context_manager.add_message_to_queue(token_warning)
    
    user_msg3 = UserMessage(content="Can you help me with a math problem?")
    
    print("Processing message with token limit warning...")
    
    async for event in orchestrator.process_message(user_msg3):
        if event.type.value == "TOOL_CALL_START":
            print(f"🔧 Tool call started: {event.tool_name}")
        elif event.type.value == "CUSTOM" and event.data.get("event_subtype") == "tool_result":
            if event.data.get("success"):
                print(f"✅ Tool '{event.data['function_name']}' succeeded: {event.data.get('result')}")
            else:
                print(f"❌ Tool '{event.data['function_name']}' failed: {event.data.get('error')}")
        elif event.type.value == "TEXT_MESSAGE_CONTENT":
            print(f"Assistant: {event.content}", end="", flush=True)
        elif event.type.value == "TEXT_MESSAGE_END":
            print()
    
    print("\n📊 MockLLMClient Features:")
    print("-" * 30)
    print(f"• Model: {mock_client.model_name}")
    print(f"• Supports tools: {mock_client.supports_tools()}")
    print(f"• Supported models: {mock_client.get_supported_models()}")
    print(f"• Response delay: {mock_client.response_delay}s")
    print(f"• Streaming mode: {mock_client.simulate_streaming}")
    
    print("\n✨ MockLLMClient Benefits:")
    print("-" * 30)
    print("• No API keys required")
    print("• Deterministic responses for testing")
    print("• Configurable delays and streaming")
    print("• Automatic context management tool calls")
    print("• Perfect for development and CI/CD")


if __name__ == "__main__":
    asyncio.run(demonstrate_mock_llm_client()) 