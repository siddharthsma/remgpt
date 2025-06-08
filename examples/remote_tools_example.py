"""
Example demonstrating RemGPT's MCP and A2A remote tool capabilities.

This example shows how to:
1. Create an orchestrator with MCP servers
2. Create an orchestrator with A2A agents
3. Create an orchestrator with both MCP and A2A tools
4. Use the API to configure remote tools dynamically
"""

import asyncio
import logging
from remgpt.context import create_context_manager
from remgpt.orchestration import create_orchestrator
from remgpt.llm.providers.mock_client import MockLLMClient
from remgpt.types import UserMessage

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def example_mcp_only():
    """Example using only MCP servers."""
    print("\n🔧 Example 1: MCP Tools Only")
    print("=" * 50)
    
    # Create context manager
    context_manager = create_context_manager(
        max_tokens=4000,
        system_instructions="You are a helpful assistant with access to MCP tools."
    )
    
    # Create orchestrator with MCP servers
    orchestrator = await create_orchestrator(
        context_manager=context_manager,
        llm_client=MockLLMClient("gpt-4"),
        mcp_servers=[
            "uvx pymupdf4llm-mcp@latest stdio",  # PDF processing
            "uvx weather-mcp@latest stdio"        # Weather data
        ]
    )
    
    print(f"✅ Orchestrator created with MCP tools")
    print(f"📊 Available tools: {len(orchestrator.tool_executor.tools)}")
    
    # Test message processing
    message = UserMessage(content="What tools do I have available?")
    
    print("\n💬 Processing message...")
    async for event in orchestrator.process_message(message):
        if event.type == "response":
            print(f"🤖 Response: {event.data.get('content', '')}")
    
    # Cleanup
    await orchestrator.cleanup_remote_tools()
    print("🧹 Cleaned up MCP connections")


async def example_a2a_only():
    """Example using only A2A agents."""
    print("\n🤝 Example 2: A2A Agents Only")
    print("=" * 50)
    
    # Create context manager
    context_manager = create_context_manager(
        max_tokens=4000,
        system_instructions="You are a helpful assistant with access to other AI agents."
    )
    
    # Create orchestrator with A2A agents
    orchestrator = await create_orchestrator(
        context_manager=context_manager,
        llm_client=MockLLMClient("gpt-4"),
        a2a_agents=[
            "http://localhost:8000",  # Weather agent
            "http://localhost:8002",  # Travel agent
            "http://localhost:8003"   # Research agent
        ]
    )
    
    print(f"✅ Orchestrator created with A2A agents")
    print(f"📊 Available tools: {len(orchestrator.tool_executor.tools)}")
    
    # Test message processing
    message = UserMessage(content="Can you help me plan a trip to Paris?")
    
    print("\n💬 Processing message...")
    async for event in orchestrator.process_message(message):
        if event.type == "response":
            print(f"🤖 Response: {event.data.get('content', '')}")
    
    # Cleanup
    await orchestrator.cleanup_remote_tools()
    print("🧹 Cleaned up A2A connections")


async def example_mixed_tools():
    """Example using both MCP and A2A tools."""
    print("\n🌐 Example 3: Mixed MCP + A2A Tools")
    print("=" * 50)
    
    # Create context manager
    context_manager = create_context_manager(
        max_tokens=4000,
        system_instructions="You are a powerful assistant with access to both MCP tools and AI agents."
    )
    
    # Create orchestrator with both MCP and A2A
    orchestrator = await create_orchestrator(
        context_manager=context_manager,
        llm_client=MockLLMClient("gpt-4"),
        mcp_servers=[
            "uvx pymupdf4llm-mcp@latest stdio",
            "http://localhost:3000/mcp"
        ],
        a2a_agents=[
            "http://localhost:8000",
            "http://localhost:8002"
        ]
    )
    
    print(f"✅ Orchestrator created with mixed tools")
    print(f"📊 Available tools: {len(orchestrator.tool_executor.tools)}")
    
    # List all available tools
    for tool in orchestrator.tool_executor.tools:
        print(f"  🔧 {tool.name}: {tool.description}")
    
    # Test message processing
    message = UserMessage(content="Analyze this PDF and then help me book a hotel based on the findings.")
    
    print("\n💬 Processing message...")
    async for event in orchestrator.process_message(message):
        if event.type == "response":
            print(f"🤖 Response: {event.data.get('content', '')}")
        elif event.type == "tool_call":
            print(f"🔧 Tool call: {event.data.get('tool_name', '')} with args: {event.data.get('args', {})}")
    
    # Cleanup
    await orchestrator.cleanup_remote_tools()
    print("🧹 Cleaned up all remote connections")


async def example_dynamic_configuration():
    """Example showing dynamic configuration via factory patterns."""
    print("\n⚙️ Example 4: Dynamic Configuration")
    print("=" * 50)
    
    # Configuration as a dictionary (like from API)
    config = {
        "context_manager": create_context_manager(
            max_tokens=8000,
            system_instructions="Dynamic assistant with configurable tools."
        ),
        "llm_client": MockLLMClient("gpt-4"),
        "mcp_servers": ["uvx weather-mcp@latest stdio"],
        "a2a_agents": ["http://localhost:8000"],
        "drift_detection_config": {
            "similarity_threshold": 0.8,
            "drift_threshold": 0.6
        }
    }
    
    # Import the config-based factory
    from remgpt.orchestration import create_orchestrator_with_config
    
    orchestrator = await create_orchestrator_with_config(config)
    
    print(f"✅ Orchestrator created from config")
    print(f"📊 Available tools: {len(orchestrator.tool_executor.tools)}")
    
    # Test configuration
    status = orchestrator.get_status()
    print(f"📈 Status: {status['status']}")
    print(f"🧠 Context summary: {status['context_summary']['total_tokens']} tokens")
    
    # Cleanup
    await orchestrator.cleanup_remote_tools()
    print("🧹 Cleaned up configured connections")


async def example_api_simulation():
    """Example simulating API usage with remote tools."""
    print("\n🌐 Example 5: API-style Configuration")
    print("=" * 50)
    
    # Simulate API request payload
    api_config = {
        "max_tokens": 4000,
        "system_instructions": "You are an assistant with remote tool capabilities.",
        "memory_content": "Previous conversation context...",
        "tools": [],  # Local tools
        "mcp_servers": ["uvx weather-mcp@latest stdio"],
        "a2a_agents": ["http://localhost:8000", "http://localhost:8002"]
    }
    
    print("📥 Received API configuration:")
    print(f"  🔧 MCP servers: {len(api_config['mcp_servers'])}")
    print(f"  🤝 A2A agents: {len(api_config['a2a_agents'])}")
    
    # Create context manager
    context_manager = create_context_manager(
        max_tokens=api_config["max_tokens"],
        system_instructions=api_config["system_instructions"],
        memory_content=api_config["memory_content"],
        tools=api_config["tools"]
    )
    
    # Create orchestrator
    orchestrator = await create_orchestrator(
        context_manager=context_manager,
        llm_client=MockLLMClient("gpt-4"),
        mcp_servers=api_config["mcp_servers"],
        a2a_agents=api_config["a2a_agents"]
    )
    
    print(f"✅ API-configured orchestrator ready")
    print(f"📊 Total tools available: {len(orchestrator.tool_executor.tools)}")
    
    # Simulate processing a user message
    user_message = {
        "content": "What's the weather like and can you help me find a restaurant?",
        "role": "user",
        "name": "api_user"
    }
    
    message = UserMessage(**user_message)
    
    print("\n💬 Processing API message...")
    response_events = []
    async for event in orchestrator.process_message(message):
        response_events.append(event)
        if event.type == "response":
            print(f"🤖 Final response ready: {len(event.data.get('content', ''))} characters")
    
    print(f"📤 Processed {len(response_events)} events")
    
    # Cleanup
    await orchestrator.cleanup_remote_tools()
    print("🧹 API session cleaned up")


async def main():
    """Run all examples."""
    print("🚀 RemGPT Remote Tools Examples")
    print("=" * 60)
    print("These examples demonstrate MCP and A2A integration.")
    print("Note: Some examples may show connection errors if servers aren't running.")
    print()
    
    try:
        # Run examples
        await example_mcp_only()
        await asyncio.sleep(1)
        
        await example_a2a_only()
        await asyncio.sleep(1)
        
        await example_mixed_tools()
        await asyncio.sleep(1)
        
        await example_dynamic_configuration()
        await asyncio.sleep(1)
        
        await example_api_simulation()
        
    except Exception as e:
        logger.error(f"Example error: {e}")
        print(f"❌ Error: {e}")
    
    print("\n✨ Examples completed!")
    print("\n📚 Key Takeaways:")
    print("  • Use `create_orchestrator()` with mcp_servers and/or a2a_agents")
    print("  • MCP servers auto-detect stdio vs SSE protocols")
    print("  • A2A agents expose communication tools automatically")
    print("  • Always call `cleanup_remote_tools()` when done")
    print("  • API can dynamically configure tools via ContextConfig")


if __name__ == "__main__":
    asyncio.run(main()) 