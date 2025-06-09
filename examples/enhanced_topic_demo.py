#!/usr/bin/env python3
"""
Enhanced Topic Management Demo for RemGPT

This demo showcases the advanced topic management features:
1. Intelligent topic similarity detection
2. Automatic topic updating vs. creation
3. Topic recall from long-term memory
4. Enhanced key facts integration
5. Real OpenAI integration with topic management
"""

import asyncio
import logging
import os
import time
from typing import AsyncGenerator
from dotenv import load_dotenv

from remgpt import (
    LLMClientFactory, Event, EventType,
    ToolExecutor, create_context_manager,
    ConversationOrchestrator, UserMessage, AssistantMessage
)


class EnhancedTopicDemo:
    """Comprehensive demo of enhanced topic management features."""
    
    def __init__(self):
        """Initialize the demo with OpenAI client and enhanced context management."""
        # Load environment variables
        load_dotenv()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.llm_client = None
        self.context_manager = None
        self.orchestrator = None
        
    async def setup_openai_client(self):
        """Setup real OpenAI client for the demo."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        # Create OpenAI client
        self.llm_client = LLMClientFactory.create_client(
            provider="openai",
            model="gpt-4o-mini",
            api_key=api_key,
            temperature=0.7,
            max_tokens=500
        )
        
        print(f"✅ OpenAI client created: {self.llm_client.model_name}")
        return self.llm_client
    
    def setup_enhanced_context_manager(self):
        """Setup context manager with enhanced topic management."""
        # Create context manager with enhanced features
        self.context_manager = create_context_manager(
            max_tokens=4000,
            vector_database=None  # For demo, we'll use in-memory only
        )
        
        # Sync with LLM client for proper token counting
        if self.llm_client:
            self.context_manager.sync_with_llm_client(self.llm_client)
        
        print(f"✅ Enhanced context manager created: {self.context_manager.get_context_summary()['total_tokens']} tokens")
        return self.context_manager
    
    def setup_orchestrator(self):
        """Setup orchestrator with enhanced topic management tools."""
        tool_executor = ToolExecutor()
        
        self.orchestrator = ConversationOrchestrator(
            context_manager=self.context_manager,
            llm_client=self.llm_client,
            tool_executor=tool_executor,
            vector_database=None,  # For demo purposes
            drift_detection_config={
                "similarity_threshold": 0.7,
                "drift_threshold": 0.5,
                "alpha": 0.05
            }
        )
        
        registered_tools = list(self.orchestrator.tool_executor.get_registered_tools())
        print(f"✅ Orchestrator created with {len(registered_tools)} tools: {registered_tools}")
        return self.orchestrator
    
    async def demonstrate_topic_creation_and_similarity(self):
        """Demonstrate topic creation and similarity detection."""
        print("\n🎯 === TOPIC CREATION & SIMILARITY DETECTION ===")
        
        # First conversation about microservices
        print("\n💬 Starting conversation about microservices...")
        microservices_message = UserMessage(content="Can you explain the key principles of microservices architecture?")
        
        print("👤 User: Can you explain the key principles of microservices architecture?")
        
        async for event in self.orchestrator.process_message(microservices_message):
            if event.type == EventType.TEXT_MESSAGE_CONTENT:
                print(f"🤖 Assistant: {event.content}", end="", flush=True)
            elif event.type == EventType.TEXT_MESSAGE_END:
                print()  # New line after complete response
            elif event.type == EventType.CUSTOM and event.data.get("event_subtype") == "tool_result":
                tool_name = event.data.get("function_name")
                result = event.data.get("result", {})
                if tool_name == "save_current_topic":
                    topic_id = result.get("topic_id")
                    print(f"📋 Topic saved: {topic_id}")
        
        # Show current context state
        context_summary = self.context_manager.get_context_summary()
        print(f"📊 Context: {context_summary['topics_count']} topics, {context_summary['total_tokens']} tokens")
        
        # Second conversation - similar topic (should trigger update)
        print("\n💬 Continuing with similar microservices topic...")
        similar_message = UserMessage(content="What are the best practices for microservices deployment and scaling?")
        
        print("👤 User: What are the best practices for microservices deployment and scaling?")
        
        async for event in self.orchestrator.process_message(similar_message):
            if event.type == EventType.TEXT_MESSAGE_CONTENT:
                print(f"🤖 Assistant: {event.content}", end="", flush=True)
            elif event.type == EventType.TEXT_MESSAGE_END:
                print()
            elif event.type == EventType.CUSTOM and event.data.get("event_subtype") == "tool_result":
                tool_name = event.data.get("function_name")
                result = event.data.get("result", {})
                if tool_name in ["save_current_topic", "update_topic"]:
                    topic_id = result.get("topic_id")
                    status = result.get("status", "unknown")
                    print(f"📋 Topic {status}: {topic_id}")
        
        # Show updated context
        context_summary = self.context_manager.get_context_summary()
        print(f"📊 Updated Context: {context_summary['topics_count']} topics, {context_summary['total_tokens']} tokens")
        
        # Display current topics with key facts
        topics = self.context_manager.get_working_context_topics()
        for i, topic in enumerate(topics, 1):
            print(f"\n📋 Topic {i}: {topic['summary']}")
            print(f"🔑 Key Facts ({topic['key_facts_count']}):")
            for j, fact in enumerate(topic['key_facts'], 1):
                print(f"   {j}. {fact}")
    
    async def demonstrate_topic_drift_and_recall(self):
        """Demonstrate topic drift detection and automatic recall."""
        print("\n🎯 === TOPIC DRIFT & AUTOMATIC RECALL ===")
        
        # Switch to a completely different topic
        print("\n💬 Switching to a different topic (Docker)...")
        docker_message = UserMessage(content="I want to learn about Docker containers and how they work.")
        
        print("👤 User: I want to learn about Docker containers and how they work.")
        
        async for event in self.orchestrator.process_message(docker_message):
            if event.type == EventType.TEXT_MESSAGE_CONTENT:
                print(f"🤖 Assistant: {event.content}", end="", flush=True)
            elif event.type == EventType.TEXT_MESSAGE_END:
                print()
            elif event.type == EventType.CUSTOM and event.data.get("event_subtype") == "tool_result":
                tool_name = event.data.get("function_name")
                result = event.data.get("result", {})
                if tool_name == "save_current_topic":
                    topic_id = result.get("topic_id")
                    print(f"📋 New topic saved: {topic_id}")
                elif tool_name == "recall_similar_topic":
                    topic_id = result.get("topic_id")
                    status = result.get("status")
                    if status == "recalled":
                        print(f"🔄 Similar topic recalled: {topic_id}")
                    else:
                        print(f"🔍 No similar topic found for recall")
        
        # Show context after topic drift
        context_summary = self.context_manager.get_context_summary()
        print(f"📊 Context after drift: {context_summary['topics_count']} topics, {context_summary['total_tokens']} tokens")
        
        # Now return to microservices topic (should trigger recall)
        print("\n💬 Returning to microservices topic (should trigger recall)...")
        return_message = UserMessage(content="Let's go back to discussing microservices. What about service discovery patterns?")
        
        print("👤 User: Let's go back to discussing microservices. What about service discovery patterns?")
        
        async for event in self.orchestrator.process_message(return_message):
            if event.type == EventType.TEXT_MESSAGE_CONTENT:
                print(f"🤖 Assistant: {event.content}", end="", flush=True)
            elif event.type == EventType.TEXT_MESSAGE_END:
                print()
            elif event.type == EventType.CUSTOM and event.data.get("event_subtype") == "tool_result":
                tool_name = event.data.get("function_name")
                result = event.data.get("result", {})
                if tool_name == "recall_similar_topic":
                    topic_id = result.get("topic_id")
                    status = result.get("status")
                    if status == "recalled":
                        print(f"🔄 Microservices topic recalled: {topic_id}")
                elif tool_name == "save_current_topic":
                    topic_id = result.get("topic_id")
                    print(f"📋 Topic saved: {topic_id}")
    
    async def demonstrate_key_facts_integration(self):
        """Demonstrate enhanced key facts display and search."""
        print("\n🎯 === ENHANCED KEY FACTS INTEGRATION ===")
        
        # Display all current topics with enhanced formatting
        topics = self.context_manager.get_working_context_topics()
        print(f"\n📚 Current Working Context ({len(topics)} topics):")
        
        for i, topic in enumerate(topics, 1):
            print(f"\n📋 Topic {i}: {topic['summary']}")
            print(f"🔑 Key Facts ({topic['key_facts_count']}):")
            for j, fact in enumerate(topic['key_facts'], 1):
                print(f"   {j}. {fact}")
            print(f"💬 Messages: {topic['message_count']} | 🕒 Created: {topic['created_at']}")
        
        # Demonstrate key facts search
        print(f"\n🔍 Searching for key facts containing 'microservices':")
        microservices_facts = self.context_manager.search_key_facts("microservices")
        for fact_info in microservices_facts:
            print(f"📋 Topic: {fact_info['topic_summary']}")
            for fact in fact_info['key_facts']:
                if 'microservices' in fact.lower():
                    print(f"   🔑 {fact}")
    
    async def demonstrate_memory_efficiency(self):
        """Demonstrate memory efficiency and context management."""
        print("\n🎯 === MEMORY EFFICIENCY & CONTEXT MANAGEMENT ===")
        
        # Show current memory usage
        context_summary = self.context_manager.get_context_summary()
        print(f"📊 Current Memory Usage:")
        print(f"   Total Tokens: {context_summary['total_tokens']}/{context_summary['max_tokens']}")
        print(f"   Token Usage: {context_summary['token_usage_percentage']:.1f}%")
        print(f"   Topics: {context_summary['topics_count']}")
        print(f"   Queue Size: {context_summary['queue_size']}")
        print(f"   Within Limit: {context_summary['within_limit']}")
        print(f"   Near Limit: {context_summary['near_limit']}")
        
        # Show topic statistics
        if hasattr(self.orchestrator, '_get_drift_statistics'):
            drift_stats = self.orchestrator._get_drift_statistics()
            print(f"\n📈 Topic Drift Statistics:")
            print(f"   Topics Created: {drift_stats.get('topics_created', 0)}")
            print(f"   Drift Detections: {drift_stats.get('drift_detections', 0)}")
            print(f"   Messages Processed: {drift_stats.get('messages_processed', 0)}")
    
    async def run_complete_demo(self):
        """Run the complete enhanced topic management demo."""
        print("🚀 RemGPT Enhanced Topic Management Demo")
        print("=" * 50)
        
        try:
            # Setup
            await self.setup_openai_client()
            self.setup_enhanced_context_manager()
            self.setup_orchestrator()
            
            # Run demonstrations
            await self.demonstrate_topic_creation_and_similarity()
            await self.demonstrate_topic_drift_and_recall()
            await self.demonstrate_key_facts_integration()
            await self.demonstrate_memory_efficiency()
            
            print("\n✅ Enhanced Topic Management Demo Complete!")
            print("\n🎉 Key Features Demonstrated:")
            print("   ✅ Intelligent topic similarity detection")
            print("   ✅ Automatic topic updating vs. creation")
            print("   ✅ Topic drift detection and recall")
            print("   ✅ Enhanced key facts integration")
            print("   ✅ Memory efficiency and context management")
            print("   ✅ Real OpenAI integration with streaming")
            
        except Exception as e:
            print(f"❌ Demo failed: {e}")
            raise


async def main():
    """Main function to run the enhanced topic management demo."""
    demo = EnhancedTopicDemo()
    await demo.run_complete_demo()


if __name__ == "__main__":
    asyncio.run(main()) 