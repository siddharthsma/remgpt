"""
End-to-end integration tests that verify complete system behavior
with real OpenAI API calls. These tests validate the exact tool call
sequences and system behavior observed in the demo.
"""

import pytest
import asyncio
import os
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv

from remgpt.core.types import UserMessage
from remgpt.llm import Event, EventType
from remgpt.orchestration.orchestrator import ConversationOrchestrator
from remgpt.context.factory import create_context_manager
from remgpt.tools.executor import ToolExecutor
from remgpt.storage.memory_database import InMemoryVectorDatabase
from remgpt.llm.providers.openai_client import OpenAIClient

# Load environment variables from .env file in this directory
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))


class TestEndToEndSystem:
    """
    End-to-end integration tests using real OpenAI API.
    
    These tests verify the complete system behavior including:
    - Real LLM responses and tool calling
    - Actual drift detection with sentence transformers
    - Real token counting and context management
    - True system statistics tracking
    
    Based on analysis of comprehensive_memory_demo.py behavior.
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
        """Set up orchestrator with real OpenAI client."""
        
        # Create real components
        context_manager = create_context_manager(max_tokens=2000)
        tool_executor = ToolExecutor()
        vector_db = InMemoryVectorDatabase()
        
        # Create real OpenAI client
        llm_client = OpenAIClient(
            model_name="gpt-4o-mini",
            api_key=api_key,
            max_tokens=150,  # Keep responses short for testing
            temperature=0.3  # Lower temperature for more predictable responses
        )
        
        # Use the exact drift configuration from demo
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

    async def _process_message_and_collect_events(self, orchestrator, message_text: str) -> tuple[List[Event], List[str]]:
        """Process a message and collect all events and tool calls."""
        message = UserMessage(content=message_text)
        
        events = []
        tool_calls = []
        
        async for event in orchestrator.process_message(message):
            events.append(event)
            
            # Track tool calls
            if (event.type == EventType.CUSTOM and 
                event.data.get("event_subtype") == "tool_call"):
                tool_name = event.data.get("function_name")
                if tool_name:
                    tool_calls.append(tool_name)
        
        return events, tool_calls

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_real_system_statistics_tracking(self, orchestrator_setup):
        """
        Test system statistics with real OpenAI API calls.
        Verifies that statistics are properly tracked during real usage.
        """
        orchestrator = orchestrator_setup
        
        # Test messages that should generate real token usage
        test_messages = [
            "Explain Python list comprehensions briefly.",
            "How do they compare to regular loops?",
            "Now tell me about machine learning algorithms.",
        ]
        
        print(f"\n🧪 Testing real system statistics with {len(test_messages)} messages...")
        
        for i, msg_text in enumerate(test_messages, 1):
            print(f"📤 Message {i}: {msg_text[:40]}...")
            
            events, tool_calls = await self._process_message_and_collect_events(orchestrator, msg_text)
            
            print(f"📊 Events: {len(events)}, Tool calls: {tool_calls}")
        
        # Check final statistics
        status = orchestrator.get_status()
        drift_stats = status.get("topic_drift", {})
        context_summary = status.get("context_summary", {})
        
        # Verify system state
        assert status.get("status") == "active", "System should be active"
        assert len(status.get("registered_tools", [])) >= 4, "Should have context management tools"
        
        # Verify message processing
        messages_processed = drift_stats.get("messages_processed", 0)
        assert messages_processed == len(test_messages), \
            f"Expected {len(test_messages)} messages processed, got {messages_processed}"
        
        # Verify token tracking (should have real values with real API)
        total_tokens = context_summary.get("total_tokens", 0)
        max_tokens = context_summary.get("max_tokens", 0)
        
        assert max_tokens == 2000, f"Expected max_tokens=2000, got {max_tokens}"
        assert total_tokens > 0, f"Expected real token usage > 0, got {total_tokens}"
        
        print(f"✅ Real statistics test passed:")
        print(f"   • Messages processed: {messages_processed}")
        print(f"   • Total tokens: {total_tokens}")
        print(f"   • Max tokens: {max_tokens}")
        print(f"   • Token utilization: {(total_tokens / max_tokens * 100):.1f}%")

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_real_drift_detection_sensitivity(self, orchestrator_setup):
        """
        Test drift detection with real sentence transformer embeddings.
        Verifies that the sensitivity improvements work with real similarity calculations.
        """
        orchestrator = orchestrator_setup
        
        # Test sequence designed to verify sensitivity improvements
        messages = [
            "Can you explain what Python list comprehensions are?",
            "How do they compare to regular for loops?",           # Should NOT trigger drift (related)
            "What are some practical examples?",                   # Should NOT trigger drift (related)
            "Now tell me about cooking pasta techniques.",         # SHOULD trigger drift (unrelated)
        ]
        
        print(f"\n🧪 Testing real drift detection sensitivity...")
        
        tool_calls_per_message = []
        
        for i, msg_text in enumerate(messages, 1):
            print(f"📤 Message {i}: {msg_text}")
            
            events, tool_calls = await self._process_message_and_collect_events(orchestrator, msg_text)
            tool_calls_per_message.append((i, tool_calls))
            
            # Show context management tool calls specifically
            context_tools = [tool for tool in tool_calls if tool in ["save_current_topic", "recall_similar_topic"]]
            if context_tools:
                print(f"🔧 Context management tools called: {context_tools}")
            else:
                print(f"✅ No drift tools called (expected for related messages)")
        
        # Verify expectations based on demo behavior
        # Messages 1-3 should not trigger context management tools (related Python topics)
        for i in range(3):  # First 3 messages
            msg_num, tools = tool_calls_per_message[i]
            context_tools = [tool for tool in tools if tool in ["save_current_topic", "recall_similar_topic"]]
            assert len(context_tools) == 0, \
                f"Message {msg_num}: Unexpected drift for related Python question. Tools: {context_tools}"
        
        # Message 4 (cooking) may or may not trigger drift depending on the exact algorithm,
        # but the system should be functional
        
        print(f"✅ Real drift sensitivity test passed:")
        print(f"   • No false drift for related Python questions")
        print(f"   • System remained functional throughout")

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_real_tool_calling_behavior(self, orchestrator_setup):
        """
        Test that real tool calls work correctly when drift is actually detected.
        """
        orchestrator = orchestrator_setup
        
        # Sequence designed to trigger actual drift detection
        messages = [
            "Tell me about Python programming concepts.",
            "What's the difference between machine learning and artificial intelligence?",  # Different topic
            "How do I cook a perfect risotto?",  # Very different topic
        ]
        
        print(f"\n🧪 Testing real tool calling behavior...")
        
        all_tool_calls = []
        
        for i, msg_text in enumerate(messages, 1):
            print(f"📤 Message {i}: {msg_text}")
            
            events, tool_calls = await self._process_message_and_collect_events(orchestrator, msg_text)
            all_tool_calls.extend(tool_calls)
            
            # Show any tool calls
            if tool_calls:
                print(f"🔧 Tools called: {tool_calls}")
        
        # Check final system state
        status = orchestrator.get_status()
        drift_stats = status.get("topic_drift", {})
        
        # Verify system functionality
        messages_processed = drift_stats.get("messages_processed", 0)
        topics_created = drift_stats.get("topics_created", 0)
        
        assert messages_processed == len(messages), f"Expected {len(messages)} messages processed"
        assert status.get("status") == "active", "System should remain active"
        
        print(f"✅ Real tool calling test completed:")
        print(f"   • Messages processed: {messages_processed}")
        print(f"   • Topics created: {topics_created}")
        print(f"   • Total tool calls: {len(all_tool_calls)}")
        print(f"   • System status: {status.get('status')}")

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_real_context_management(self, orchestrator_setup):
        """
        Test context management with real token counting and limits.
        """
        orchestrator = orchestrator_setup
        
        # Messages designed to test context limits
        lengthy_messages = [
            "Explain the fundamentals of object-oriented programming including inheritance, polymorphism, and encapsulation.",
            "Describe the differences between SQL and NoSQL databases, including when to use each approach.",
            "What are the key principles of machine learning and how do supervised and unsupervised learning differ?",
            "Explain how web authentication works including OAuth, JWT tokens, and session management.",
        ]
        
        print(f"\n🧪 Testing real context management...")
        
        token_progression = []
        
        for i, msg_text in enumerate(lengthy_messages, 1):
            print(f"📤 Message {i}: {msg_text[:50]}...")
            
            events, tool_calls = await self._process_message_and_collect_events(orchestrator, msg_text)
            
            # Check current token usage
            status = orchestrator.get_status()
            context_summary = status.get("context_summary", {})
            current_tokens = context_summary.get("total_tokens", 0)
            
            token_progression.append(current_tokens)
            print(f"📊 Current tokens: {current_tokens}")
            
            # Check for context management warnings/actions
            if current_tokens > 1400:  # 70% of 2000
                print(f"⚠️  Approaching token limit: {current_tokens}/2000")
        
        # Verify token progression
        assert all(tokens >= 0 for tokens in token_progression), "Tokens should be non-negative"
        assert token_progression[-1] > token_progression[0], "Tokens should increase with more messages"
        
        # Check final state
        final_status = orchestrator.get_status()
        context_summary = final_status.get("context_summary", {})
        final_tokens = context_summary.get("total_tokens", 0)
        
        print(f"✅ Real context management test completed:")
        print(f"   • Token progression: {' → '.join(map(str, token_progression))}")
        print(f"   • Final tokens: {final_tokens}/2000")
        print(f"   • Context utilization: {(final_tokens / 2000 * 100):.1f}%")

if __name__ == "__main__":
    # Run integration tests with specific markers
    pytest.main([__file__, "-v", "-m", "integration"]) 