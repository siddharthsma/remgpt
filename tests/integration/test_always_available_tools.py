"""
Test that tools are always available when registered, regardless of drift detection.
This validates the fix for the issue where tools were only available during drift/token warnings.
"""

import pytest
import asyncio
import os
from typing import Dict, Any
from dotenv import load_dotenv

from remgpt.core.types import UserMessage
from remgpt.llm import Event, EventType
from remgpt.orchestration.orchestrator import ConversationOrchestrator
from remgpt.context.factory import create_context_manager
from remgpt.context.context_tools import ContextManagementToolFactory
from remgpt.tools.executor import ToolExecutor
from remgpt.storage.memory_database import InMemoryVectorDatabase
from remgpt.llm.providers.openai_client import OpenAIClient

# Load environment variables from .env file in this directory
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))


class TestAlwaysAvailableTools:
    """
    Test that tools are always available when registered, not just during drift/token warnings.
    """

    @pytest.fixture
    def api_key(self):
        """Get OpenAI API key from environment."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("OPENAI_API_KEY not found in environment")
        return api_key

    @pytest.fixture
    def orchestrator_setup(self, api_key):
        """Set up orchestrator with tools registered."""
        
        # Create real components
        context_manager = create_context_manager(max_tokens=2000)
        tool_executor = ToolExecutor()
        vector_db = InMemoryVectorDatabase()
        
        # Register context management tools
        context_tools_factory = ContextManagementToolFactory(context_manager)
        context_tools_factory.register_tools_with_executor(tool_executor)
        
        # Create real OpenAI client
        llm_client = OpenAIClient(
            model_name="gpt-4o-mini",
            api_key=api_key,
            max_tokens=150,
            temperature=0.3
        )
        
        # Normal drift configuration (not overly aggressive)
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
            drift_detection_config=drift_config
        )
        
        return orchestrator

    async def _process_message_and_extract_tools(self, orchestrator, message_text: str) -> Dict[str, Any]:
        """Process a message and extract tool call information."""
        message = UserMessage(content=message_text)
        
        events = []
        tool_calls = []
        
        async for event in orchestrator.process_message(message):
            events.append(event)
            
            # Track tool calls - look for both tool execution events and tool results
            if (event.type == EventType.CUSTOM and 
                event.data and event.data.get("event_subtype") == "tool_result"):
                # Tool execution completed successfully
                tool_call_info = {
                    "function_name": event.data.get("function_name"),
                    "arguments": event.data.get("arguments", {}),
                    "result": event.data.get("result", {}),
                    "success": event.data.get("success", False),
                    "timestamp": event.timestamp
                }
                tool_calls.append(tool_call_info)
                
            # Also track tool call start events
            elif hasattr(event, 'type') and hasattr(event.type, 'value') and event.type.value == 'TOOL_CALL_START':
                tool_call_info = {
                    "function_name": getattr(event, 'tool_name', 'unknown'),
                    "call_id": getattr(event, 'tool_call_id', 'unknown'),
                    "stage": "started",
                    "timestamp": event.timestamp
                }
                tool_calls.append(tool_call_info)
        
        return {
            "events": events,
            "tool_calls": tool_calls,
            "total_events": len(events)
        }

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_tools_available_without_drift_detection(self, orchestrator_setup):
        """
        Test that tools are available immediately on the first message,
        before any drift detection could occur.
        """
        orchestrator = orchestrator_setup
        
        print(f"\n🧪 Testing tools available without drift detection...")
        
        # First message - no drift possible yet
        message_text = "Please call save_current_topic with topic_summary='First message test' and topic_key_facts=['no drift yet', 'tools should work']. Execute this function call now."
        
        print(f"📤 First message (no drift possible): {message_text[:60]}...")
        
        result = await self._process_message_and_extract_tools(orchestrator, message_text)
        tool_calls = result["tool_calls"]
        
        print(f"🔧 Tool calls on first message: {len(tool_calls)}")
        for tc in tool_calls:
            func_name = tc.get('function_name', 'unknown')
            stage = tc.get('stage', 'completed')  
            args = tc.get('arguments', {})
            result = tc.get('result', {})
            if stage == 'started':
                print(f"   • {func_name} (started)")
            else:
                print(f"   • {func_name} (completed): args={args}, result={result}")
        
        # Check orchestrator status
        status = orchestrator.get_status()
        drift_stats = status.get('topic_drift', {})
        drift_detections = drift_stats.get('drift_detections', 0)
        
        print(f"📊 Drift status: {drift_detections} detections")
        
        # Validate that tools work without drift
        assert len(tool_calls) > 0, "Tools should be available on first message without drift detection"
        assert drift_detections == 0, "No drift should be detected on first message"
        
        tool_names = [tc.get("function_name", "") for tc in tool_calls]
        assert "save_current_topic" in tool_names, "save_current_topic should be called"
        
        print(f"✅ Tools are available without drift detection!")

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_tools_available_with_similar_topics(self, orchestrator_setup):
        """
        Test that tools remain available even when topics are similar (no drift).
        """
        orchestrator = orchestrator_setup
        
        print(f"\n🧪 Testing tools available with similar topics (no drift)...")
        
        # Send similar messages that shouldn't trigger drift
        similar_messages = [
            "Tell me about Python programming basics.",
            "Can you explain more about Python syntax?",
            "Please call save_current_topic with topic_summary='Python discussion' and topic_key_facts=['basic syntax', 'programming']. Execute this now."
        ]
        
        all_tool_calls = []
        
        for i, msg_text in enumerate(similar_messages, 1):
            print(f"\n📤 Similar message {i}: {msg_text[:50]}...")
            
            result = await self._process_message_and_extract_tools(orchestrator, msg_text)
            tool_calls = result["tool_calls"]
            all_tool_calls.extend(tool_calls)
            
            print(f"   🔧 Tool calls: {len(tool_calls)}")
            for tc in tool_calls:
                func_name = tc.get('function_name', 'unknown')
                stage = tc.get('stage', 'completed')
                if stage == 'started':
                    print(f"      • {func_name} (started)")
                else:
                    print(f"      • {func_name} (completed)")
            
            # Check drift status
            status = orchestrator.get_status()
            drift_stats = status.get('topic_drift', {})
            drift_detections = drift_stats.get('drift_detections', 0)
            print(f"   📊 Drift detections so far: {drift_detections}")
        
        # Final validation
        status = orchestrator.get_status()
        final_drift_stats = status.get('topic_drift', {})
        final_drift_detections = final_drift_stats.get('drift_detections', 0)
        
        print(f"\n📊 Final results:")
        print(f"   • Total tool calls: {len(all_tool_calls)}")
        print(f"   • Final drift detections: {final_drift_detections}")
        
        # Should have tool calls even without drift
        assert len(all_tool_calls) > 0, "Tools should be available even without drift detection"
        
        # The last message explicitly requested save_current_topic
        tool_names = [tc.get("function_name", "") for tc in all_tool_calls]
        assert "save_current_topic" in tool_names, "save_current_topic should be called when requested"
        
        print(f"✅ Tools remain available with similar topics!")

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_tools_work_for_general_requests(self, orchestrator_setup):
        """
        Test that tools work for general requests, not just context management.
        """
        orchestrator = orchestrator_setup
        
        print(f"\n🧪 Testing tools work for general requests...")
        
        # Test different types of tool requests
        tool_requests = [
            "Please call save_current_topic to save our conversation with topic_summary='General test' and topic_key_facts=['testing', 'validation'].",
            "Can you call recall_similar_topic with user_message='find related topics'?",
            "Call update_topic with topic_id='test123' and additional_summary='Updated information'."
        ]
        
        successful_calls = 0
        
        for i, request in enumerate(tool_requests, 1):
            print(f"\n📤 Request {i}: {request[:50]}...")
            
            result = await self._process_message_and_extract_tools(orchestrator, request)
            tool_calls = result["tool_calls"]
            
            if tool_calls:
                successful_calls += 1
                tool_names = [tc.get('function_name', 'unknown') for tc in tool_calls]
                print(f"   ✅ Tool called: {tool_names}")
            else:
                print(f"   ⚠️  No tools called")
        
        print(f"\n📊 General tool request results:")
        print(f"   • Successful tool calls: {successful_calls}/{len(tool_requests)}")
        print(f"   • Success rate: {(successful_calls/len(tool_requests)*100):.1f}%")
        
        # Should have at least some successful tool calls
        assert successful_calls > 0, "Tools should work for general requests"
        
        print(f"✅ Tools work for general requests!")

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_tools_available_with_low_token_usage(self, orchestrator_setup):
        """
        Test that tools are available even with very low token usage (no token warnings).
        """
        orchestrator = orchestrator_setup
        
        print(f"\n🧪 Testing tools available with low token usage...")
        
        # Short message that won't trigger token warnings
        short_message = "Call save_current_topic with topic_summary='Short' and topic_key_facts=['brief']."
        
        print(f"📤 Short message: {short_message}")
        
        result = await self._process_message_and_extract_tools(orchestrator, short_message)
        tool_calls = result["tool_calls"]
        
        # Check context status
        status = orchestrator.get_status()
        context_summary = status.get('context_summary', {})
        total_tokens = context_summary.get('total_tokens', 0)
        max_tokens = context_summary.get('max_tokens', 2000)
        token_usage = (total_tokens / max_tokens) * 100 if max_tokens > 0 else 0
        
        print(f"📊 Token usage: {total_tokens}/{max_tokens} ({token_usage:.1f}%)")
        print(f"🔧 Tool calls: {len(tool_calls)}")
        
        # Should have tools available even with low token usage
        assert token_usage < 50, "Token usage should be low for this test"
        assert len(tool_calls) > 0, "Tools should be available even with low token usage"
        
        print(f"✅ Tools work with low token usage!")

if __name__ == "__main__":
    # Run always available tools tests
    pytest.main([__file__, "-v", "-s", "-m", "integration"]) 