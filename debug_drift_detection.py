#!/usr/bin/env python3
"""
Debug script to understand why drift detection isn't triggering tool calls.
"""

import asyncio
import os
from dotenv import load_dotenv

# Load API key
load_dotenv('tests/integration/.env')

from remgpt.core.types import UserMessage
from remgpt.orchestration.orchestrator import ConversationOrchestrator
from remgpt.context.factory import create_context_manager
from remgpt.tools.executor import ToolExecutor
from remgpt.storage.memory_database import InMemoryVectorDatabase
from remgpt.llm.providers.openai_client import OpenAIClient

async def debug_drift_detection():
    print("🔍 Debugging drift detection and tool calling...")
    
    context_manager = create_context_manager(max_tokens=2000)
    tool_executor = ToolExecutor()
    vector_db = InMemoryVectorDatabase()
    
    llm_client = OpenAIClient(
        model_name='gpt-4o-mini',
        api_key=os.getenv('OPENAI_API_KEY'),
        max_tokens=150,
        temperature=0.3
    )
    
    # Very aggressive drift config to force detection
    drift_config = {
        'similarity_threshold': 0.9,   # Very high - almost everything should be drift
        'window_size': 5,              # Reasonable window
        'drift_threshold': 0.1,        # Very low threshold
        'alpha': 0.5,                  # Very lenient
        'min_messages': 1              # Detect immediately
    }
    
    orchestrator = ConversationOrchestrator(
        context_manager=context_manager,
        llm_client=llm_client,
        tool_executor=tool_executor,
        vector_database=vector_db,
        drift_detection_config=drift_config
    )
    
    # Start with one topic, then switch to very different topics repeatedly
    # This should provide the "sustained evidence" needed for drift detection
    messages = [
        'Tell me about Python programming and software development.',
        'Explain quantum physics and particle behavior.',
        'Describe molecular biology and DNA structure.',
        'How do you repair automotive engines?',
        'What are the principles of cooking French cuisine?',
        'Explain cryptocurrency and blockchain technology.',
    ]
    
    for i, msg_text in enumerate(messages, 1):
        print(f"\n📤 Message {i}: {msg_text}")
        message = UserMessage(content=msg_text)
        
        # Track events
        tool_calls = []
        
        async for event in orchestrator.process_message(message):
            # Check for tool calls
            if (hasattr(event, 'type') and hasattr(event, 'data') and 
                event.data and event.data.get('event_subtype') == 'tool_call'):
                tool_name = event.data.get('function_name')
                tool_calls.append(tool_name)
                print(f"🔧 Tool called: {tool_name}")
        
        print(f"   Tool calls triggered: {len(tool_calls)} - {tool_calls}")
        
        # Check drift stats after each message
        status = orchestrator.get_status()
        drift_stats = status.get('topic_drift', {})
        drift_detections = drift_stats.get('drift_detections', 0)
        topics_created = drift_stats.get('topics_created', 0)
        
        print(f"   Drift detections so far: {drift_detections}")
        print(f"   Topics created so far: {topics_created}")
        
        # Show similarity data
        recent_similarities = drift_stats.get('recent_similarities', [])
        if recent_similarities:
            print(f"   Recent similarities: {[f'{s:.3f}' for s in recent_similarities]}")
            print(f"   Mean recent similarity: {drift_stats.get('mean_recent_similarity', 0):.3f}")
    
    print(f"\n📊 Final orchestrator status:")
    final_status = orchestrator.get_status()
    final_drift_stats = final_status.get('topic_drift', {})
    print(f"   Status: {final_status.get('status')}")
    print(f"   Registered tools: {len(final_status.get('registered_tools', []))}")
    print(f"   Total drift detections: {final_drift_stats.get('drift_detections', 0)}")
    print(f"   Total topics created: {final_drift_stats.get('topics_created', 0)}")
    print(f"   Messages processed: {final_drift_stats.get('messages_processed', 0)}")

if __name__ == "__main__":
    asyncio.run(debug_drift_detection()) 