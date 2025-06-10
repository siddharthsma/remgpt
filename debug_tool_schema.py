#!/usr/bin/env python3
"""
Debug script to examine tool schemas and parameters when drift is detected.
"""

import asyncio
import os
import json
from dotenv import load_dotenv

# Load API key
load_dotenv('tests/integration/.env')

from remgpt.core.types import UserMessage
from remgpt.orchestration.orchestrator import ConversationOrchestrator
from remgpt.context.factory import create_context_manager
from remgpt.tools.executor import ToolExecutor
from remgpt.storage.memory_database import InMemoryVectorDatabase
from remgpt.llm.providers.openai_client import OpenAIClient

async def debug_tool_schemas():
    print("🔍 Debugging tool schemas and function calling...")
    
    context_manager = create_context_manager(max_tokens=2000)
    tool_executor = ToolExecutor()
    vector_db = InMemoryVectorDatabase()
    
    llm_client = OpenAIClient(
        model_name='gpt-4o-mini',
        api_key=os.getenv('OPENAI_API_KEY'),
        max_tokens=150,
        temperature=0.3
    )
    
    # Aggressive drift config
    drift_config = {
        'similarity_threshold': 0.9,
        'window_size': 5,
        'drift_threshold': 0.1,
        'alpha': 0.5,
        'min_messages': 1
    }
    
    orchestrator = ConversationOrchestrator(
        context_manager=context_manager,
        llm_client=llm_client,
        tool_executor=tool_executor,
        vector_database=vector_db,
        drift_detection_config=drift_config
    )
    
    # Check what tools are registered
    print("\n📋 Checking registered tools...")
    status = orchestrator.get_status()
    registered_tools = status.get('registered_tools', [])
    print(f"   Registered tools: {len(registered_tools)}")
    for tool in registered_tools:
        print(f"   • {tool}")
    
    # Get tool schemas from executor
    print("\n🔧 Tool schemas from executor:")
    tool_schemas = tool_executor.get_tool_schemas()
    print(f"   Tool schemas count: {len(tool_schemas)}")
    for schema in tool_schemas:
        print(f"   • {schema.get('function', {}).get('name', 'Unknown')}")
        # print(f"     Schema: {json.dumps(schema, indent=2)}")
    
    # Send messages to trigger drift
    messages = [
        'Tell me about Python programming.',
        'Explain quantum physics principles.',  # Should trigger drift
        'Describe molecular biology.',          # Should trigger drift again
    ]
    
    for i, msg_text in enumerate(messages, 1):
        print(f"\n📤 Message {i}: {msg_text}")
        message = UserMessage(content=msg_text)
        
        # Track what actually gets sent to LLM
        tool_calls_detected = []
        function_calling_enabled = False
        
        async for event in orchestrator.process_message(message):
            # Check for RUN_STARTED events which contain LLM parameters
            if hasattr(event, 'type') and event.type.value == 'run_started':
                event_data = getattr(event, 'data', {})
                function_calling_enabled = event_data.get('function_calling', False)
                print(f"   🔧 Function calling enabled: {function_calling_enabled}")
            
            # Check for tool calls
            if (hasattr(event, 'type') and hasattr(event, 'data') and 
                event.data and event.data.get('event_subtype') == 'tool_call'):
                tool_name = event.data.get('function_name')
                tool_calls_detected.append(tool_name)
                print(f"   🎯 Tool called: {tool_name}")
        
        print(f"   Tool calls detected: {len(tool_calls_detected)} - {tool_calls_detected}")
        
        # Check drift status
        status = orchestrator.get_status()
        drift_stats = status.get('topic_drift', {})
        drift_detections = drift_stats.get('drift_detections', 0)
        topics_created = drift_stats.get('topics_created', 0)
        
        print(f"   Drift detections so far: {drift_detections}")
        print(f"   Topics created so far: {topics_created}")

if __name__ == "__main__":
    asyncio.run(debug_tool_schemas()) 