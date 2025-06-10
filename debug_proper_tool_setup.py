#!/usr/bin/env python3
"""
Debug script that properly sets up tools like the orchestrator does.
"""

import asyncio
import os
import json
from dotenv import load_dotenv

# Load API key
load_dotenv('tests/integration/.env')

from remgpt.llm.providers.openai_client import OpenAIClient
from remgpt.tools.executor import ToolExecutor
from remgpt.context.factory import create_context_manager
from remgpt.context.context_tools import ContextManagementToolFactory

async def test_proper_tool_setup():
    print("🔍 Testing with proper tool setup like orchestrator...")
    
    # Create context manager
    context_manager = create_context_manager(max_tokens=2000)
    
    # Create tool executor
    tool_executor = ToolExecutor()
    
    # Create context tools factory and register tools (like orchestrator does)
    context_tools_factory = ContextManagementToolFactory(context_manager)
    context_tools_factory.register_tools_with_executor(tool_executor)
    
    print(f"📋 Registered tools: {len(tool_executor.get_registered_tools())}")
    for tool_name in tool_executor.get_registered_tools():
        print(f"   • {tool_name}")
    
    # Get tool schemas
    tool_schemas = tool_executor.get_tool_schemas()
    print(f"🔧 Tool schemas: {len(tool_schemas)}")
    for schema in tool_schemas:
        print(f"   • {schema.get('function', {}).get('name')}")
    
    # Create OpenAI client
    client = OpenAIClient(
        model_name='gpt-4o-mini',
        api_key=os.getenv('OPENAI_API_KEY'),
        max_tokens=150,
        temperature=0.3
    )
    
    # Test with properly setup tools
    messages = [{
        "role": "user",
        "content": "Please call save_current_topic with topic_summary='Proper setup test' and topic_key_facts=['test fact 1', 'test fact 2']. Execute this function call now."
    }]
    
    print(f"\n📤 Testing with properly registered tools...")
    print(f"Message: {messages[0]['content']}")
    print(f"Tools available: {len(tool_schemas)}")
    
    tool_calls_found = []
    response_content = ""
    
    async for event in client.generate_stream(messages, tools=tool_schemas):
        if hasattr(event, 'content') and event.content:
            response_content += event.content
        
        if hasattr(event, 'tool_name') and event.tool_name:
            tool_calls_found.append(event.tool_name)
            print(f"   🔧 Tool called: {event.tool_name}")
        
        if hasattr(event, 'tool_args') and event.tool_args:
            print(f"   📋 Tool args: {event.tool_args}")
    
    print(f"\n📊 Proper Setup Test Results:")
    print(f"   Tool calls detected: {len(tool_calls_found)} - {tool_calls_found}")
    print(f"   Response content length: {len(response_content)}")
    
    if len(tool_calls_found) > 0:
        print(f"✅ Tool calling works with proper setup!")
        
        # Test actual tool execution
        print(f"\n🔧 Testing tool execution...")
        for tool_name in tool_calls_found:
            try:
                result = await tool_executor.execute_tool(
                    "test_call",
                    tool_name,
                    {"topic_summary": "Test execution", "topic_key_facts": ["fact 1", "fact 2"]}
                )
                print(f"   ✅ {tool_name} executed successfully: {result}")
            except Exception as e:
                print(f"   ❌ {tool_name} execution failed: {e}")
    else:
        print(f"❌ No tool calls detected even with proper setup")
        print(f"   Response: {response_content[:200]}...")
        
        # Show tool schema for debugging
        if tool_schemas:
            print(f"\n🔧 First tool schema for debugging:")
            print(json.dumps(tool_schemas[0], indent=2))

if __name__ == "__main__":
    asyncio.run(test_proper_tool_setup()) 