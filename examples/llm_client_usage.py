"""
Example usage of the LLM Client system with RemGPT.

This example demonstrates:
1. Creating LLM clients for different providers
2. Setting up tool execution
3. Integrating with the conversation orchestrator
4. Handling streaming events and tool calls
"""

import asyncio
import logging
from typing import AsyncGenerator

from remgpt import (
    # LLM Client System
    LLMClientFactory, BaseLLMClient, Event, EventType,
    
    # Tool System
    ToolExecutor, BaseTool,
    
    # Context Management
    create_context_manager,
    
    # Orchestration
    ConversationOrchestrator,
    
    # Types
    UserMessage, AssistantMessage
)


# Example custom tool
class CalculatorTool(BaseTool):
    """Simple calculator tool for demonstration."""
    
    def __init__(self):
        super().__init__("calculator", "Perform basic arithmetic calculations")
    
    async def execute(self, operation: str, a: float, b: float) -> dict:
        """Execute a calculation."""
        try:
            if operation == "add":
                result = a + b
            elif operation == "subtract":
                result = a - b
            elif operation == "multiply":
                result = a * b
            elif operation == "divide":
                if b == 0:
                    return {"error": "Division by zero"}
                result = a / b
            else:
                return {"error": f"Unknown operation: {operation}"}
            
            return {"result": result, "operation": operation, "operands": [a, b]}
        except Exception as e:
            return {"error": str(e)}
    
    def get_schema(self) -> dict:
        """Get the tool schema for LLM function calling."""
        return {
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "Perform basic arithmetic calculations",
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
                            "description": "First operand"
                        },
                        "b": {
                            "type": "number", 
                            "description": "Second operand"
                        }
                    },
                    "required": ["operation", "a", "b"]
                }
            }
        }


class WeatherTool(BaseTool):
    """Mock weather tool for demonstration."""
    
    def __init__(self):
        super().__init__("get_weather", "Get current weather information")
    
    async def execute(self, location: str) -> dict:
        """Get weather for a location (mock implementation)."""
        # Mock weather data
        mock_weather = {
            "new york": {"temperature": 72, "condition": "sunny", "humidity": 45},
            "london": {"temperature": 65, "condition": "cloudy", "humidity": 80},
            "tokyo": {"temperature": 78, "condition": "rainy", "humidity": 90},
        }
        
        location_lower = location.lower()
        if location_lower in mock_weather:
            weather = mock_weather[location_lower]
            return {
                "location": location,
                "temperature": weather["temperature"],
                "condition": weather["condition"],
                "humidity": weather["humidity"],
                "unit": "fahrenheit"
            }
        else:
            return {"error": f"Weather data not available for {location}"}
    
    def get_schema(self) -> dict:
        """Get the tool schema for LLM function calling."""
        return {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather information for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city or location to get weather for"
                        }
                    },
                    "required": ["location"]
                }
            }
        }


class MockLLMClient(BaseLLMClient):
    """Mock LLM client for demonstration purposes."""
    
    def __init__(self, model_name: str = "mock-model", **kwargs):
        super().__init__(model_name, **kwargs)
        self.tool_executor = kwargs.get('tool_executor')
    
    async def generate_stream(self, messages: list, **kwargs) -> AsyncGenerator[Event, None]:
        """Generate mock streaming response."""
        yield Event(type=EventType.RUN_STARTED)
        
        # Check if the last message mentions calculation or weather
        last_message = messages[-1] if messages else {}
        content = last_message.get('content', '').lower()
        
        if 'calculate' in content or 'math' in content or any(op in content for op in ['add', 'multiply', 'divide', 'subtract']):
            # Simulate tool call for calculation
            yield Event(type=EventType.TOOL_CALL_START, tool_call_id="calc_1", tool_name="calculator")
            yield Event(
                type=EventType.TOOL_CALL_ARGS,
                tool_call_id="calc_1", 
                tool_args={"operation": "add", "a": 10, "b": 5}
            )
            yield Event(type=EventType.TOOL_CALL_END, tool_call_id="calc_1")
            
            # Simulate response after tool execution
            yield Event(type=EventType.TEXT_MESSAGE_START)
            yield Event(type=EventType.TEXT_MESSAGE_CONTENT, content="I'll calculate that for you. ")
            yield Event(type=EventType.TEXT_MESSAGE_CONTENT, content="The result is 15.")
            yield Event(type=EventType.TEXT_MESSAGE_END, content="I'll calculate that for you. The result is 15.")
            
        elif 'weather' in content:
            # Simulate tool call for weather
            yield Event(type=EventType.TOOL_CALL_START, tool_call_id="weather_1", tool_name="get_weather")
            yield Event(
                type=EventType.TOOL_CALL_ARGS,
                tool_call_id="weather_1",
                tool_args={"location": "New York"}
            )
            yield Event(type=EventType.TOOL_CALL_END, tool_call_id="weather_1")
            
            # Simulate response after tool execution
            yield Event(type=EventType.TEXT_MESSAGE_START)
            yield Event(type=EventType.TEXT_MESSAGE_CONTENT, content="Let me check the weather for you. ")
            yield Event(type=EventType.TEXT_MESSAGE_CONTENT, content="It's currently 72°F and sunny in New York.")
            yield Event(type=EventType.TEXT_MESSAGE_END, content="Let me check the weather for you. It's currently 72°F and sunny in New York.")
            
        else:
            # Regular text response
            yield Event(type=EventType.TEXT_MESSAGE_START)
            yield Event(type=EventType.TEXT_MESSAGE_CONTENT, content="Hello! I'm a mock LLM assistant. ")
            yield Event(type=EventType.TEXT_MESSAGE_CONTENT, content="I can help with calculations and weather queries.")
            yield Event(type=EventType.TEXT_MESSAGE_END, content="Hello! I'm a mock LLM assistant. I can help with calculations and weather queries.")
        
        yield Event(type=EventType.RUN_FINISHED)
    
    def send_tool_result(self, tool_call_id: str, result: any) -> None:
        """Mock implementation of sending tool result."""
        print(f"Tool result for {tool_call_id}: {result}")
    
    def supports_tools(self) -> bool:
        return True
    
    def get_supported_models(self) -> list:
        return ["mock-model"]


async def demonstrate_llm_client_creation():
    """Demonstrate creating different LLM clients."""
    print("=== LLM Client Creation Demo ===")
    
    factory = LLMClientFactory()
    
    # Show supported providers
    providers = factory.get_supported_providers()
    print(f"Supported providers: {providers}")
    
    # Note: These would require actual API keys and installed packages
    # For demonstration, we'll show the interface
    
    try:
        # Example of how you would create real clients:
        # openai_client = factory.create_client(
        #     provider="openai",
        #     model_name="gpt-4",
        #     api_key="your-openai-api-key"
        # )
        
        # claude_client = factory.create_client(
        #     provider="claude", 
        #     model_name="claude-3-sonnet-20240229",
        #     api_key="your-anthropic-api-key"
        # )
        
        # gemini_client = factory.create_client(
        #     provider="gemini",
        #     model_name="gemini-1.5-pro", 
        #     api_key="your-google-api-key"
        # )
        
        print("✓ LLM clients can be created with proper API keys")
        
    except Exception as e:
        print(f"Note: Real clients require API keys and installed packages: {e}")
    
    # Create mock client for demonstration
    mock_client = MockLLMClient()
    print(f"✓ Created mock client: {mock_client.model_name}")
    
    return mock_client


async def demonstrate_tool_system():
    """Demonstrate the tool execution system."""
    print("\n=== Tool System Demo ===")
    
    # Create tool executor
    executor = ToolExecutor()
    
    # Register tools
    calc_tool = CalculatorTool()
    weather_tool = WeatherTool()
    
    executor.register_tool(calc_tool)
    executor.register_tool(weather_tool)
    
    print(f"Registered tools: {executor.get_registered_tools()}")
    
    # Get tool schemas (for LLM function calling)
    schemas = executor.get_tool_schemas()
    print(f"Tool schemas available: {len(schemas)} tools")
    
    # Test tool execution
    calc_result = await executor.execute_tool(
        tool_call_id="test_calc",
        tool_name="calculator",
        tool_args={"operation": "multiply", "a": 7, "b": 6}
    )
    print(f"Calculator result: {calc_result}")
    
    weather_result = await executor.execute_tool(
        tool_call_id="test_weather",
        tool_name="get_weather", 
        tool_args={"location": "London"}
    )
    print(f"Weather result: {weather_result}")
    
    return executor


async def demonstrate_integration():
    """Demonstrate integration with the orchestrator."""
    print("\n=== Integration Demo ===")
    
    # Create context manager
    context_manager = create_context_manager(
        max_tokens=4000,
        system_instructions="You are a helpful assistant with access to tools.",
        model="gpt-4"
    )
    
    # Create tool executor and register tools
    tool_executor = ToolExecutor()
    tool_executor.register_tool(CalculatorTool())
    tool_executor.register_tool(WeatherTool())
    
    # Create mock LLM client
    llm_client = MockLLMClient(tool_executor=tool_executor)
    
    # Create orchestrator with LLM client and tool executor
    orchestrator = ConversationOrchestrator(
        context_manager=context_manager,
        llm_client=llm_client,  # Pass the client instance, not the method
        tool_executor=tool_executor
    )
    
    print("✓ Created orchestrator with LLM client integration")
    
    # Process a message that should trigger tool usage
    user_message = UserMessage(content="Can you calculate 15 + 25 for me?")
    
    print(f"\nProcessing message: {user_message.content}")
    print("Streaming events:")
    
    async for event in orchestrator.process_message(user_message):
        print(f"  {event.type}: {getattr(event, 'data', getattr(event, 'content', ''))}")
    
    return orchestrator


async def demonstrate_event_handling():
    """Demonstrate handling different types of events."""
    print("\n=== Event Handling Demo ===")
    
    # Create mock client
    client = MockLLMClient()
    
    # Test different message types
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What's the weather like?"}
    ]
    
    print("Processing weather query...")
    async for event in client.generate_stream(messages):
        if event.type == EventType.TOOL_CALL_START:
            print(f"  🔧 Tool call started: {event.tool_name} (ID: {event.tool_call_id})")
        elif event.type == EventType.TOOL_CALL_ARGS:
            print(f"  📝 Tool arguments: {event.tool_args}")
        elif event.type == EventType.TOOL_CALL_END:
            print(f"  ✅ Tool call completed: {event.tool_call_id}")
        elif event.type == EventType.TEXT_MESSAGE_CONTENT:
            print(f"  💬 Content: {event.content}")
        elif event.type in [EventType.RUN_STARTED, EventType.RUN_FINISHED]:
            print(f"  🏃 {event.type}")


async def main():
    """Main demonstration function."""
    print("RemGPT LLM Client System Demonstration")
    print("=" * 50)
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Demonstrate each component
        await demonstrate_llm_client_creation()
        await demonstrate_tool_system()
        await demonstrate_integration()
        await demonstrate_event_handling()
        
        print("\n" + "=" * 50)
        print("✅ All demonstrations completed successfully!")
        print("\nKey takeaways:")
        print("1. LLM clients provide a unified interface across providers")
        print("2. Tool execution is handled separately from LLM generation")
        print("3. Events provide a streaming interface for real-time responses")
        print("4. Integration with orchestrator enables context management")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main()) 