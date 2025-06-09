#!/usr/bin/env python3
"""
RemGPT Key Facts Demo

This script demonstrates how key facts are incorporated into the working context
and how they can be searched and retrieved.
"""

import asyncio
import time
import numpy as np
from typing import List, Dict, Any

# Import RemGPT components
from remgpt.context import create_context_manager
from remgpt.summarization import Topic
from remgpt.types import UserMessage, AssistantMessage


def create_sample_topic(topic_id: str, summary: str, key_facts: List[str], message_count: int = 3) -> Topic:
    """Create a sample topic with key facts."""
    return Topic(
        id=topic_id,
        summary=summary,
        key_facts=key_facts,
        mean_embedding=np.random.rand(384),  # Random embedding for demo
        original_messages=[
            UserMessage(content="Sample user message"),
            AssistantMessage(content="Sample assistant response"),
            UserMessage(content="Follow-up question")
        ][:message_count],
        timestamp=time.time(),
        message_count=message_count
    )


def main():
    """Demonstrate key facts functionality."""
    print("🔑 RemGPT Key Facts Integration Demo")
    print("=" * 50)
    
    # Create context manager
    context_manager = create_context_manager(
        max_tokens=4000,
        system_instructions="You are a helpful assistant with enhanced memory."
    )
    
    print("✅ Context manager created")
    
    # Create sample topics with rich key facts
    topics = [
        create_sample_topic(
            "topic_microservices_001",
            "Discussion about microservices architecture principles and implementation",
            [
                "Microservices are loosely coupled services that communicate via APIs",
                "Each service should have single responsibility and own its data",
                "Service discovery patterns include client-side and server-side discovery",
                "Circuit breaker pattern prevents cascade failures between services",
                "User prefers Docker containerization with Kubernetes orchestration",
                "Database per service pattern ensures data isolation"
            ]
        ),
        create_sample_topic(
            "topic_python_async_002",
            "Python async/await patterns and best practices discussion",
            [
                "async/await introduced in Python 3.5 for asynchronous programming",
                "asyncio.gather() runs multiple coroutines concurrently",
                "Use asyncio.create_task() to schedule coroutines for background execution",
                "Avoid blocking calls in async functions - use async alternatives",
                "aiohttp is preferred for async HTTP requests over requests library"
            ]
        ),
        create_sample_topic(
            "topic_database_design_003",
            "Database design patterns and optimization strategies",
            [
                "Normalization reduces data redundancy but may impact query performance",
                "Indexing improves query speed but slows down write operations",
                "Connection pooling reduces database connection overhead",
                "User mentioned PostgreSQL as preferred database for new projects",
                "ACID properties ensure data consistency in transactions"
            ]
        )
    ]
    
    # Add topics to working context
    print("\n📋 Adding topics to working context...")
    for topic in topics:
        # Simulate saving topics
        context_manager.context.working_context.add_topic(topic)
        print(f"   Added: {topic.summary[:50]}...")
    
    # Display working context messages
    print("\n💬 Working Context Messages:")
    print("-" * 30)
    messages = context_manager.get_messages_for_llm()
    for i, message in enumerate(messages, 1):
        if hasattr(message, 'content'):
            content_preview = message.content[:200] + "..." if len(message.content) > 200 else message.content
            print(f"{i}. {content_preview}")
            print()
    
    # Demonstrate key facts retrieval
    print("\n🔍 Key Facts Functionality:")
    print("-" * 30)
    
    # Get all key facts
    all_facts = context_manager.get_all_key_facts()
    print(f"📊 Total topics with key facts: {len(all_facts)}")
    
    total_facts = sum(len(topic_facts['key_facts']) for topic_facts in all_facts)
    print(f"📈 Total key facts across all topics: {total_facts}")
    
    # Search for specific terms
    search_terms = ["async", "database", "service"]
    print(f"\n🔎 Searching key facts for terms: {search_terms}")
    
    for term in search_terms:
        matches = context_manager.search_key_facts(term)
        print(f"\n'{term}' found in {len(matches)} topics:")
        for match in matches:
            print(f"  📋 Topic: {match['topic_summary'][:60]}...")
            for fact in match['matching_facts']:
                print(f"     • {fact}")
    
    # Display working context statistics
    print("\n📊 Working Context Statistics:")
    print("-" * 30)
    stats = context_manager.context.working_context.get_statistics()
    
    print(f"Current topics: {stats['current_topics']}")
    print(f"Total key facts: {stats['total_key_facts']}")
    print(f"Token count: {stats['token_count']}")
    
    print("\nTopic details:")
    for topic_stat in stats['topics']:
        print(f"  📋 {topic_stat['summary']}")
        print(f"     🔑 Key facts: {topic_stat['key_facts_count']} ({topic_stat['key_facts_preview']})")
        print(f"     💬 Messages: {topic_stat['message_count']} | 🟢 Tokens: {topic_stat['token_count']}")
        print()
    
    # Demonstrate context summary
    print("📝 Context Summary:")
    print("-" * 30)
    summary = context_manager.get_context_summary()
    print(f"Total tokens: {summary['total_tokens']}/{summary['max_tokens']}")
    print(f"Topics count: {summary['topics_count']}")
    print(f"Within limit: {'✅' if summary['within_limit'] else '❌'}")
    print(f"Near limit: {'⚠️' if summary['near_limit'] else '✅'}")
    
    print("\n🎉 Demo completed! Key facts are now prominently incorporated into working context.")


if __name__ == "__main__":
    main() 