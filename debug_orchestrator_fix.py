#!/usr/bin/env python3
"""
Debug script to test if the orchestrator fix for always-available tools is working.
"""

import asyncio
import os
import logging
from dotenv import load_dotenv

# Load API key
load_dotenv('tests/integration/.env')

from remgpt.core.types import UserMessage
from remgpt.orchestration.orchestrator import ConversationOrchestrator
from remgpt.context.factory import create_context_manager
from remgpt.context.context_tools import ContextManagementToolFactory
from remgpt.tools.executor import ToolExecutor
from remgpt.storage.memory_database import InMemoryVectorDatabase
from remgpt.llm.providers.openai_client import OpenAIClient

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

async def debug_orchestrator_fix():
    print("🔍 Debugging orchestrator fix for always-available tools...")
    
    # Create components
    context_manager = create_context_manager(max_tokens=2000)
    tool_executor = ToolExecutor()
    vector_db = InMemoryVectorDatabase()
    
    # Register tools
    context_tools_factory = ContextManagementToolFactory(context_manager)
    context_tools_factory.register_tools_with_executor(tool_executor)
    
    print(f"📋 Tools registered: {tool_executor.get_registered_tools()}")
    print(f"📋 Tools count: {len(tool_executor.get_registered_tools())}")
    
    # Create LLM client
    llm_client = OpenAIClient(
        model_name='gpt-4o-mini',
        api_key=os.getenv('OPENAI_API_KEY'),
        max_tokens=150,
        temperature=0.3
    )
    
    # Create orchestrator with debug logging
    drift_config = {
        "similarity_threshold": 0.6,
        "window_size": 5,
        "drift_threshold": 1.0,
        "alpha": 0.1,
        "min_messages": 3
    }
    
    orchestrator = ConversationOrchestrator(
        context_manager=context_manager,
        llm_client=llm_client,
        tool_executor=tool_executor,
        vector_database=vector_db,
        drift_detection_config=drift_config,
        logger=logger
    )
    
    # Check orchestrator status
    status = orchestrator.get_status()
    registered_tools = status.get('registered_tools', [])
    print(f"📋 Orchestrator registered tools: {registered_tools}")
    
    # Manually test the logic
    tools_available = len(orchestrator.tool_executor.get_registered_tools()) > 0
    print(f"🔧 Manual check - tools_available: {tools_available}")
    print(f"🔧 Tool schemas available: {len(orchestrator.tool_executor.get_tool_schemas())}")
    
    # Test first message
    message = UserMessage(content="Please call save_current_topic with topic_summary='Debug test' and topic_key_facts=['test']. Execute this now.")
    
    print(f"\n📤 Sending test message...")
    
    all_events = []
    tool_calls = []
    
    async for event in orchestrator.process_message(message):
        all_events.append(event)
        
        # Print all event types to debug
        event_type = event.type.value if hasattr(event.type, 'value') else str(event.type)
        print(f"   📋 Event: {event_type}")
        
        # Look for function calling enabled indicator
        if hasattr(event, 'type') and event.type.value == 'run_started':
            event_data = getattr(event, 'data', {})
            function_calling = event_data.get('function_calling', False)
            print(f"   🔧 Function calling enabled: {function_calling}")
            print(f"   📊 Event data: {event_data}")
        
        # Track tool calls
        if (hasattr(event, 'type') and hasattr(event, 'data') and 
            event.data and event.data.get('event_subtype') == 'tool_call'):
            tool_name = event.data.get('function_name')
            tool_calls.append(tool_name)
            print(f"   🎯 Tool called: {tool_name}")
    
    print(f"\n📊 Debug results:")
    print(f"   • Total events: {len(all_events)}")
    print(f"   • Tool executor has {len(tool_executor.get_registered_tools())} tools")
    print(f"   • Tool calls made: {len(tool_calls)} - {tool_calls}")
    
    # Check event types
    event_types = [event.type.value if hasattr(event.type, 'value') else str(event.type) for event in all_events]
    print(f"   • Event types: {set(event_types)}")

if __name__ == "__main__":
    asyncio.run(debug_orchestrator_fix()) 