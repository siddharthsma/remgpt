#!/usr/bin/env python3
"""
Direct OpenAI client test to verify tool calling works at the API level.
"""

import asyncio
import os
import json
from dotenv import load_dotenv

# Load API key
load_dotenv('tests/integration/.env')

from remgpt.llm.providers.openai_client import OpenAIClient
from remgpt.tools.executor import ToolExecutor

async def test_openai_direct():
    print("🔍 Testing OpenAI client directly...")
    
    # Create OpenAI client
    client = OpenAIClient(
        model_name='gpt-4o-mini',
        api_key=os.getenv('OPENAI_API_KEY'),
        max_tokens=150,
        temperature=0.3
    )
    
    # Get tool schemas
    tool_executor = ToolExecutor()
    tool_schemas = tool_executor.get_tool_schemas()
    
    print(f"📋 Available tool schemas: {len(tool_schemas)}")
    for schema in tool_schemas:
        print(f"   • {schema.get('function', {}).get('name')}")
    
    # Create messages that explicitly request tool usage
    messages = [{
        "role": "user",
        "content": "Please call save_current_topic with topic_summary='Direct API test' and topic_key_facts=['test fact 1', 'test fact 2']. Then call recall_similar_topic with user_message='test query'. Execute these function calls now."
    }]
    
    print(f"\n📤 Sending direct API request with tools...")
    print(f"Message: {messages[0]['content']}")
    print(f"Tools provided: {len(tool_schemas)} tools")
    
    # Test with tools enabled
    tool_calls_found = []
    response_content = ""
    
    async for event in client.generate_stream(messages, tools=tool_schemas):
        print(f"Event: {event.type.value if hasattr(event.type, 'value') else event.type}")
        
        if hasattr(event, 'content') and event.content:
            response_content += event.content
            print(f"   Content: {event.content}")
        
        if hasattr(event, 'tool_name') and event.tool_name:
            tool_calls_found.append(event.tool_name)
            print(f"   🔧 Tool call: {event.tool_name}")
        
        if hasattr(event, 'tool_args') and event.tool_args:
            print(f"   📋 Tool args: {event.tool_args}")
    
    print(f"\n📊 Direct API Test Results:")
    print(f"   Tool calls detected: {len(tool_calls_found)} - {tool_calls_found}")
    print(f"   Response content length: {len(response_content)}")
    
    if len(tool_calls_found) == 0:
        print(f"❌ No tool calls detected even with direct API call")
        print(f"   This suggests an issue with:")
        print(f"   1. Tool schemas format")
        print(f"   2. OpenAI client configuration") 
        print(f"   3. LLM model not responding to tool availability")
        
        # Show first tool schema for debugging
        if tool_schemas:
            print(f"\n🔧 First tool schema for debugging:")
            print(json.dumps(tool_schemas[0], indent=2))
    else:
        print(f"✅ Direct API tool calling works!")

if __name__ == "__main__":
    asyncio.run(test_openai_direct()) 