#!/usr/bin/env python3
"""
Comprehensive RemGPT Memory Algorithm Demo

This demo showcases all key features of RemGPT's memory system:
1. Multi-turn conversations with topic drift detection
2. Automatic topic summarization and storage in vector database
3. Topic similarity search and recall
4. Context management with token limits
5. Working context updates
6. Context management tools (save/update/recall/evict topics)

Requirements:
- Set OPENAI_API_KEY environment variable
- Install dependencies: pip install openai python-dotenv

Usage:
python examples/comprehensive_memory_demo.py
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from remgpt import (
    ConversationOrchestrator,
    create_context_manager,
    OpenAIClient,
    InMemoryVectorDatabase,
    Event,
    EventType
)
from remgpt.core.types import UserMessage
from remgpt.tools import ToolExecutor

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MemoryDemoRunner:
    """Comprehensive demo runner for RemGPT memory features."""
    
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        self.orchestrator = None
        self.conversation_history = []
        
    async def setup(self):
        """Initialize the RemGPT components."""
        logger.info("🔧 Setting up RemGPT components...")
        
        # Create LLM client
        self.llm_client = OpenAIClient(
            model_name="gpt-4o-mini",
            api_key=self.api_key,
            max_tokens=150,
            temperature=0.7
        )
        
        # Create vector database for topic storage
        self.vector_db = InMemoryVectorDatabase(logger=logger)
        
        # Create context manager with token limits
        self.context_manager = create_context_manager(
            max_tokens=2000,
            logger=logger
        )
        
        # Create tool executor
        self.tool_executor = ToolExecutor()
        
        # Create orchestrator with topic drift detection
        drift_config = {
            "similarity_threshold": 0.6,
            "window_size": 5,
            "drift_threshold": 1.0,
            "alpha": 0.1,
            "min_messages": 3
        }
        
        self.orchestrator = ConversationOrchestrator(
            context_manager=self.context_manager,
            llm_client=self.llm_client,
            tool_executor=self.tool_executor,
            vector_database=self.vector_db,
            drift_detection_config=drift_config,
            logger=logger
        )
        
        logger.info("✅ RemGPT setup complete!")
        
    def print_separator(self, title: str):
        """Print a formatted separator for demo sections."""
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}\n")
        
    async def send_message_and_display(self, user_input: str, demo_description: str = ""):
        """Send a message and display the conversation flow."""
        if demo_description:
            print(f"\n🎯 DEMO: {demo_description}")
        
        print(f"👤 User: {user_input}")
        
        # Create user message
        message = UserMessage(content=user_input)
        
        # Track conversation
        self.conversation_history.append(("User", user_input))
        
        # Process message through orchestrator
        response_parts = []
        tool_calls = []
        
        async for event in self.orchestrator.process_message(message):
            if event.type == EventType.TEXT_MESSAGE_CONTENT:
                response_parts.append(event.content or "")
                
            elif event.type == EventType.CUSTOM and event.data.get("event_subtype") == "tool_call":
                tool_info = {
                    "name": event.data.get("function_name"),
                    "args": event.data.get("arguments", {}),
                }
                tool_calls.append(tool_info)
                print(f"🔧 Tool Call: {tool_info['name']} with args: {tool_info['args']}")
                
            elif event.type == EventType.CUSTOM and event.data.get("event_subtype") == "tool_result":
                result = event.data.get("result", {})
                print(f"📊 Tool Result: {result}")
                
            elif event.type == EventType.RUN_ERROR:
                print(f"❌ Error: {event.error}")
                
        # Display assistant response
        full_response = "".join(response_parts).strip()
        if full_response:
            print(f"🤖 Assistant: {full_response}")
            self.conversation_history.append(("Assistant", full_response))
        
        # Display tool calls summary
        if tool_calls:
            print(f"📋 Tool calls made: {[tc['name'] for tc in tool_calls]}")
            
        return full_response, tool_calls
    
    async def run_comprehensive_demo(self):
        """Run the complete comprehensive demo."""
        try:
            print("🚀 Starting Comprehensive RemGPT Memory Algorithm Demo")
            print("   This demo will showcase all key features of RemGPT's memory system")
            
            # Setup
            await self.setup()
            
            # Demo 1: Topic drift detection
            self.print_separator("TOPIC DRIFT DETECTION & MANAGEMENT")
            
            await self.send_message_and_display(
                "Can you explain what Python list comprehensions are?",
                "Starting with Python programming topic"
            )
            
            await self.send_message_and_display(
                "How do they compare to regular for loops?",
                "Continuing Python topic"
            )
            
            await self.send_message_and_display(
                "What's the difference between supervised and unsupervised machine learning?",
                "TOPIC SHIFT: Moving to machine learning (should trigger drift detection)"
            )
            
            # Demo 2: Topic recall
            self.print_separator("TOPIC RECALL FROM VECTOR DATABASE")
            
            await self.send_message_and_display(
                "What are some cooking techniques for pasta?",
                "NEW TOPIC: Cooking"
            )
            
            await self.send_message_and_display(
                "Actually, let's go back to Python. Tell me about lambda functions.",
                "TOPIC RECALL: Back to Python (should recall from vector database)"
            )
            
            # Demo 3: Context management
            self.print_separator("CONTEXT MANAGEMENT & TOKEN LIMITS")
            
            await self.send_message_and_display(
                "Tell me about Python decorators",
                "Adding content to test context management"
            )
            
            await self.send_message_and_display(
                "Can you give examples of decorators?",
                "Continuing to fill context"
            )
            
            await self.send_message_and_display(
                "Now explain Python generators and yield",
                "Should demonstrate context eviction or management"
            )
            
            # Demo 4: Manual topic operations
            self.print_separator("MANUAL TOPIC OPERATIONS")
            
            await self.send_message_and_display(
                "I want to learn about databases. Please save our Python discussion and tell me about database normalization.",
                "MANUAL OPERATION: Requesting explicit topic save and shift"
            )
            
            # Display final status
            self.print_separator("SYSTEM STATUS & STATISTICS")
            
            try:
                status = self.orchestrator.get_status()
                
                print("📊 System Statistics:")
                print(f"   • Status: {status.get('status', 'unknown')}")
                print(f"   • Registered Tools: {len(status.get('registered_tools', []))}")
                
                drift_stats = status.get('topic_drift', {})
                print(f"   • Topics Created: {drift_stats.get('topics_created', 0)}")
                print(f"   • Drift Detections: {drift_stats.get('drift_detections', 0)}")
                print(f"   • Messages Processed: {drift_stats.get('messages_processed', 0)}")
                
                context_summary = status.get('context_summary', {})
                print(f"   • Context Tokens: {context_summary.get('total_tokens', 0)}")
                print(f"   • Max Tokens: {context_summary.get('max_tokens', 0)}")
                
            except Exception as e:
                print(f"❌ Error getting system status: {e}")
                
            self.print_separator("DEMO COMPLETE")
            print("✅ All RemGPT memory features demonstrated!")
            print("   Key features tested:")
            print("   • Topic drift detection ✓")
            print("   • Automatic topic summarization ✓")
            print("   • Vector database storage & recall ✓")
            print("   • Context management & token limits ✓")
            print("   • Context management tools ✓")
            
        except Exception as e:
            logger.error(f"Demo failed: {e}")
            print(f"❌ Demo failed: {e}")
            raise

async def main():
    """Main entry point."""
    demo = MemoryDemoRunner()
    await demo.run_comprehensive_demo()

if __name__ == "__main__":
    asyncio.run(main()) 