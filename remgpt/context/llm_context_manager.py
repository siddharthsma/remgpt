"""
LLM Context Manager class for token limit monitoring.
"""

from typing import List, Dict, Any, Optional, TYPE_CHECKING
import logging
import numpy as np
from .token_counter import TokenCounter
from .llm_context import LLMContext
from ..types import Message
from ..summarization import Topic

if TYPE_CHECKING:
    from ..llm import BaseLLMClient


class LLMContextManager:
    """Manager for LLM context with token limit monitoring."""
    
    def __init__(self, max_tokens: int, logger: Optional[logging.Logger] = None):
        """
        Initialize context manager.
        
        Args:
            max_tokens: Maximum total tokens allowed (for monitoring)
            logger: Optional logger instance
            
        Note: Token counting will automatically use the correct tokenizer when an LLM client
        is connected via the orchestrator. Until then, it uses a sensible default (GPT-4).
        """
        self.max_tokens = max_tokens
        self.logger = logger or logging.getLogger(__name__)
        # TokenCounter will use GPT-4 tokenizer as default
        self.token_counter = TokenCounter()
        self.context = LLMContext(self.token_counter, logger=self.logger)
    
    def sync_with_llm_client(self, llm_client: "BaseLLMClient"):
        """
        Synchronize token counting with the actual LLM client being used.
        
        This ensures that token counting matches the actual model being used,
        providing accurate token limits and usage tracking.
        
        Args:
            llm_client: The LLM client that will be used for generation
        """
        self.token_counter.update_from_llm_client(llm_client)
        self.logger.info(f"Token counter synchronized with LLM model: {self.token_counter.model_used}")
    
    def check_token_limit(self) -> bool:
        """
        Check if context fits within token limit.
        
        Returns:
            True if context fits within limit
        """
        current_tokens = self.context.get_total_tokens()
        return current_tokens <= self.max_tokens
    
    def is_near_token_limit(self, threshold: float = 0.7) -> bool:
        """
        Check if context is near token limit.
        
        Args:
            threshold: Percentage threshold (0.7 = 70%)
            
        Returns:
            True if context is near the limit
        """
        current_tokens = self.context.get_total_tokens()
        return current_tokens >= (self.max_tokens * threshold)
    
    def get_token_usage(self) -> Dict[str, int]:
        """Get current token usage information."""
        total_tokens = self.context.get_total_tokens()
        return {
            "total_tokens": total_tokens,
            "max_tokens": self.max_tokens,
            "tokens_remaining": self.max_tokens - total_tokens,
            "over_limit": total_tokens > self.max_tokens,
            "token_usage_percentage": (total_tokens / self.max_tokens) * 100
        }
    
    def add_message_to_queue(self, message: Message):
        """Add a message to the FIFO queue."""
        self.context.fifo_queue.add_message(message)
    
    def flush_fifo_queue(self):
        """
        Flush all messages from the FIFO queue.
        Used after saving a topic to clear the conversation queue.
        """
        self.context.fifo_queue.clear()
        self.logger.info("FIFO queue flushed")
    
    def save_current_topic(self, topic_summary: str, topic_key_facts: List[str] = None) -> str:
        """
        Save the current conversation as a topic and flush the FIFO queue.
        
        This method is designed to be called by the LLM via function calling.
        
        Args:
            topic_summary: Summary of the current conversation topic
            topic_key_facts: List of key facts from the conversation
            
        Returns:
            Topic ID that was created
        """
        import uuid
        import time
        
        # Generate topic ID
        topic_id = f"topic_{int(time.time())}_{str(uuid.uuid4())[:8]}"
        
        # Get messages from FIFO queue
        queue_messages = self.context.fifo_queue.get_messages()
        
        # Create a simple mean embedding (or zero embedding if no messages)
        if queue_messages:
            # For now, just use a zero embedding as placeholder
            # In a real implementation, you would calculate the mean embedding from message embeddings
            mean_embedding = np.zeros(384)  # all-MiniLM-L6-v2 has 384 dimensions
        else:
            mean_embedding = np.zeros(384)
        
        # Create topic with correct parameters
        topic = Topic(
            id=topic_id,
            summary=topic_summary,
            key_facts=topic_key_facts or [],
            mean_embedding=mean_embedding,
            original_messages=queue_messages.copy(),
            timestamp=time.time(),
            message_count=len(queue_messages)
        )
        
        # Add topic to working context
        self.context.working_context.add_topic(topic)
        
        # Flush the FIFO queue
        self.flush_fifo_queue()
        
        self.logger.info(f"Saved current conversation as topic '{topic_id}' and flushed FIFO queue")
        return topic_id
    
    def evict_oldest_topic(self) -> Optional[str]:
        """
        Evict the oldest topic from working context.
        
        This method is designed to be called by the LLM via function calling.
        
        Returns:
            ID of evicted topic, or None if no topics to evict
        """
        evicted_topic = self.context.working_context.evict_oldest_topic()
        
        if evicted_topic:
            self.logger.info(f"Evicted oldest topic: {evicted_topic.id}")
            return evicted_topic.id
        else:
            self.logger.info("No topics available to evict")
            return None
    
    def get_working_context_topics(self) -> List[Dict[str, Any]]:
        """
        Get list of topics in working context.
        
        Returns:
            List of topic summaries with key facts
        """
        topics = self.context.working_context.get_topics()
        return [
            {
                "id": topic.id,
                "summary": topic.summary,
                "key_facts": topic.key_facts,
                "key_facts_count": len(topic.key_facts),
                "message_count": topic.message_count,
                "timestamp": topic.timestamp,
                "created_at": topic._format_timestamp()
            }
            for topic in topics
        ]
    
    def get_all_key_facts(self) -> List[Dict[str, Any]]:
        """
        Get all key facts from working context topics.
        
        Returns:
            List of all key facts with topic context
        """
        return self.context.working_context.get_all_key_facts()
    
    def search_key_facts(self, search_term: str) -> List[Dict[str, Any]]:
        """
        Search for key facts containing a specific term.
        
        Args:
            search_term: Term to search for in key facts
            
        Returns:
            List of matching facts with topic context
        """
        return self.context.working_context.search_key_facts(search_term)
    
    def update_context(self, key: str, value: Any):
        """Update working context."""
        self.context.working_context.update_context(key, value)
    
    def remove_context(self, key: str):
        """Remove a context key."""
        self.context.working_context.remove_context(key)
    
    def get_messages_for_llm(self, block_order: Optional[List[str]] = None) -> List[Message]:
        """
        Get the final list of messages for LLM.
        
        Returns:
            List of messages ready for LLM consumption
        """
        return self.context.to_messages(block_order)
    
    def get_context_summary(self) -> Dict[str, Any]:
        """Get a summary of the current context state."""
        token_usage = self.get_token_usage()
        return {
            **token_usage,
            "block_token_counts": self.context.get_block_token_counts(),
            "total_messages": len(self.context.to_messages()),
            "queue_size": self.context.fifo_queue.get_size(),
            "within_limit": self.check_token_limit(),
            "near_limit": self.is_near_token_limit(),
            "topics_count": len(self.context.working_context.get_topics())
        } 