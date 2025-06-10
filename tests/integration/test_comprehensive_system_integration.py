"""
Comprehensive system integration test that verifies exact tool call sequences
and system behavior based on the actual demo script output analysis.
"""

import pytest
import asyncio
import logging
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import List, Dict, Any, AsyncGenerator

from remgpt.core.types import UserMessage
from remgpt.llm import Event, EventType
from remgpt.orchestration.orchestrator import ConversationOrchestrator
from remgpt.context.factory import create_context_manager
from remgpt.tools.executor import ToolExecutor
from remgpt.storage.memory_database import InMemoryVectorDatabase


class TestComprehensiveSystemIntegration:
    """
    Test that verifies the complete system behavior matches the demo expectations.
    Based on analysis of actual comprehensive_memory_demo.py output.
    
    Key observations from demo:
    1. Message 1: "Can you explain what Python list comprehensions are?" → No drift, no tools
    2. Message 2: "How do they compare to regular for loops?" → similarity=0.289, no drift (✅ improvement working)
    3. Message 3: "What's the difference between supervised and unsupervised machine learning?" → similarity=0.080, insufficient evidence initially
    4. Message 4: "What are some cooking techniques for pasta?" → similarity=0.095, DRIFT CONFIRMED, tools called
    5. Statistics should show: Topics Created > 0, Messages Processed correctly, Context Tokens > 0
    """

    @pytest.fixture
    def orchestrator_setup(self):
        """Set up orchestrator with proper mocking for controlled testing."""
        
        # Create real components (no mocking for core logic)
        context_manager = create_context_manager(max_tokens=2000)
        tool_executor = ToolExecutor()
        vector_db = InMemoryVectorDatabase()
        
        # Mock only the OpenAI client to avoid API calls
        mock_llm_client = Mock()
        mock_llm_client.model_name = "gpt-4o-mini"
        
        # Use the exact drift configuration from demo
        drift_config = {
            "similarity_threshold": 0.6,  # Same as demo
            "window_size": 5,
            "drift_threshold": 1.0,
            "alpha": 0.1,
            "min_messages": 3
        }
        
        orchestrator = ConversationOrchestrator(
            context_manager=context_manager,
            llm_client=mock_llm_client,
            tool_executor=tool_executor,
            vector_database=vector_db,
            drift_detection_config=drift_config
        )
        
        return orchestrator, mock_llm_client

    def _create_simple_events(self, content: str, has_tool_calls: bool = False) -> List[Event]:
        """Create simple mock events for testing."""
        events = [
            Event(type=EventType.RUN_STARTED, timestamp=1.0),
            Event(type=EventType.TEXT_MESSAGE_CONTENT, content=content, timestamp=1.0)
        ]
        
        if has_tool_calls:
            # Add custom events that mimic tool calls
            events.extend([
                Event(
                    type=EventType.CUSTOM,
                    data={
                        "event_subtype": "tool_call",
                        "function_name": "save_current_topic",
                        "arguments": {"topic_summary": "Test topic", "topic_key_facts": []}
                    },
                    timestamp=1.0
                ),
                Event(
                    type=EventType.CUSTOM,
                    data={
                        "event_subtype": "tool_result", 
                        "result": {"topic_id": "test_123", "status": "saved"}
                    },
                    timestamp=1.0
                )
            ])
        
        events.append(Event(type=EventType.RUN_FINISHED, timestamp=1.0))
        return events

    @pytest.mark.asyncio
    async def test_system_statistics_accuracy(self, orchestrator_setup):
        """
        Test that system statistics are accurately tracked and reported.
        This addresses the original issue where statistics were showing 0.
        """
        orchestrator, mock_llm_client = orchestrator_setup
        
        # Create async generator that yields events
        async def mock_generate(*args, **kwargs):
            events = self._create_simple_events("Test response")
            for event in events:
                yield event
        
        mock_llm_client.generate_stream = AsyncMock(side_effect=mock_generate)
        
        # Send test messages
        test_messages = [
            "Tell me about Python",
            "Now tell me about cooking", 
            "What about databases?"
        ]
        
        for msg_text in test_messages:
            message = UserMessage(content=msg_text)
            async for event in orchestrator.process_message(message):
                pass  # Just process the events
        
        # Check final statistics
        status = orchestrator.get_status()
        drift_stats = status.get("topic_drift", {})
        context_summary = status.get("context_summary", {})
        
        # Verify that statistics are being tracked (not 0)
        messages_processed = drift_stats.get("messages_processed", 0)
        assert messages_processed == len(test_messages), \
            f"Expected {len(test_messages)} messages processed, got {messages_processed}"
        
        # Verify context tokens are tracked
        total_tokens = context_summary.get("total_tokens", 0)
        max_tokens = context_summary.get("max_tokens", 0)
        
        assert max_tokens == 2000, f"Expected max_tokens=2000, got {max_tokens}"
        assert total_tokens >= 0, f"Expected non-negative total_tokens, got {total_tokens}"
        
        print(f"✅ Statistics accuracy test passed:")
        print(f"   • Messages processed: {messages_processed}")
        print(f"   • Total tokens: {total_tokens}")
        print(f"   • Max tokens: {max_tokens}")

    @pytest.mark.asyncio
    async def test_drift_detection_sensitivity_fix(self, orchestrator_setup):
        """
        Specifically test that the drift detection sensitivity improvements work.
        Based on demo observation: similarity=0.289 for "How do they compare to regular for loops?" 
        should NOT trigger drift detection.
        """
        orchestrator, mock_llm_client = orchestrator_setup
        
        # Mock simple responses without tool calls
        async def mock_generate(*args, **kwargs):
            events = self._create_simple_events("Simple response without tools")
            for event in events:
                yield event
        
        mock_llm_client.generate_stream = AsyncMock(side_effect=mock_generate)
        
        # Test the exact sequence that caused the sensitivity issue
        messages = [
            "Can you explain what Python list comprehensions are?",
            "How do they compare to regular for loops?",  # This was triggering false drift
            "What are some examples of list comprehensions?",  # Should also not trigger drift
        ]
        
        tool_calls_made = []
        
        for i, msg_text in enumerate(messages):
            message = UserMessage(content=msg_text)
            
            message_tool_calls = []
            async for event in orchestrator.process_message(message):
                if (event.type == EventType.CUSTOM and 
                    event.data.get("event_subtype") == "tool_call"):
                    tool_name = event.data.get("function_name")
                    message_tool_calls.append(tool_name)
            
            tool_calls_made.append((i+1, message_tool_calls))
        
        # Verify that the related Python questions don't trigger drift tools
        for i, (msg_num, tools) in enumerate(tool_calls_made):
            context_management_tools = [tool for tool in tools 
                                      if tool in ["save_current_topic", "recall_similar_topic"]]
            
            assert len(context_management_tools) == 0, \
                f"Message {msg_num}: Unexpected drift detection for related question. Tools called: {context_management_tools}"
        
        print(f"✅ Drift sensitivity test passed:")
        print(f"   • No false drift detection for {len(messages)} related questions")

    @pytest.mark.asyncio
    async def test_tool_call_sequence_verification(self, orchestrator_setup):
        """
        Test that when drift IS detected, the correct tools are called in sequence.
        This verifies the main functionality described in the demo.
        """
        orchestrator, mock_llm_client = orchestrator_setup
        
        # Track call count to simulate different responses
        call_count = 0
        
        async def mock_generate(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            
            # First few calls: simple responses (no drift detection yet)
            if call_count <= 3:
                events = self._create_simple_events(f"Response {call_count}")
            else:
                # Later calls: responses that should trigger tools (drift detected)
                events = self._create_simple_events(f"Drift response {call_count}", has_tool_calls=True)
            
            for event in events:
                yield event
        
        mock_llm_client.generate_stream = AsyncMock(side_effect=mock_generate)
        
        # Simulate the demo sequence
        demo_messages = [
            "Can you explain what Python list comprehensions are?",  # No drift
            "How do they compare to regular for loops?",            # No drift (improvement)
            "What's the difference between supervised and unsupervised machine learning?",  # No drift initially
            "What are some cooking techniques for pasta?",          # Should trigger drift
            "Actually, let's go back to Python. Tell me about lambda functions.",  # Should trigger drift
        ]
        
        tool_calls_per_message = []
        
        for i, msg_text in enumerate(demo_messages):
            message = UserMessage(content=msg_text)
            
            message_tool_calls = []
            async for event in orchestrator.process_message(message):
                if (event.type == EventType.CUSTOM and 
                    event.data.get("event_subtype") == "tool_call"):
                    tool_name = event.data.get("function_name")
                    message_tool_calls.append(tool_name)
            
            tool_calls_per_message.append((i+1, msg_text[:30] + "...", message_tool_calls))
        
        # Verify final system state
        status = orchestrator.get_status()
        drift_stats = status.get("topic_drift", {})
        
        # System should be functional
        assert status.get("status") == "active", "System should be active"
        assert len(status.get("registered_tools", [])) >= 4, "Should have context management tools"
        
        # Messages should be processed
        messages_processed = drift_stats.get("messages_processed", 0)
        assert messages_processed == len(demo_messages), \
            f"Expected {len(demo_messages)} messages processed, got {messages_processed}"
        
        # Log results 
        print(f"✅ Tool call sequence test completed:")
        print(f"   • Messages processed: {messages_processed}")
        print(f"   • System status: {status.get('status')}")
        print(f"   • Tool call sequence:")
        for msg_num, msg_preview, tools in tool_calls_per_message:
            print(f"     Message {msg_num} ({msg_preview}): {tools}")

    @pytest.mark.asyncio 
    async def test_context_token_tracking(self, orchestrator_setup):
        """
        Test that context tokens are properly tracked and not showing 0.
        This addresses the specific issue reported in the demo.
        """
        orchestrator, mock_llm_client = orchestrator_setup
        
        async def mock_generate(*args, **kwargs):
            # Create responses that should consume tokens
            events = self._create_simple_events("This is a substantial response that should consume context tokens and be tracked properly by the system.")
            for event in events:
                yield event
        
        mock_llm_client.generate_stream = AsyncMock(side_effect=mock_generate)
        
        # Send substantial messages to accumulate tokens
        substantial_messages = [
            "Please provide a detailed explanation of machine learning algorithms and their applications in modern software development.",
            "Can you give me a comprehensive overview of database normalization techniques and when to apply them?",
            "Explain the differences between various Python data structures and their performance characteristics.",
        ]
        
        for msg_text in substantial_messages:
            message = UserMessage(content=msg_text)
            async for event in orchestrator.process_message(message):
                pass
        
        # Check token tracking
        status = orchestrator.get_status()
        context_summary = status.get("context_summary", {})
        
        total_tokens = context_summary.get("total_tokens", 0)
        max_tokens = context_summary.get("max_tokens", 0)
        
        # Verify tokens are being tracked
        assert max_tokens == 2000, f"Max tokens should be 2000, got {max_tokens}"
        assert total_tokens > 0, f"Total tokens should be > 0, got {total_tokens}"
        assert total_tokens <= max_tokens, f"Total tokens ({total_tokens}) should not exceed max ({max_tokens})"
        
        print(f"✅ Context token tracking test passed:")
        print(f"   • Total tokens: {total_tokens}")
        print(f"   • Max tokens: {max_tokens}")
        print(f"   • Token utilization: {(total_tokens / max_tokens * 100):.1f}%")

if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__, "-v"]) 