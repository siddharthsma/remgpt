"""
Working context block for LLM Context management.
"""

from typing import List, Dict, Any, Optional
from collections import deque
from ..base_block import BaseBlock
from ...types import Message, SystemMessage
from ...summarization import Topic
import logging


class WorkingContextBlock(BaseBlock):
    """Block containing working context data as a FIFO queue of Topics."""
    
    def __init__(
        self, 
        initial_data: Dict[str, Any] = None, 
        name: str = "working_context",
        max_topics: int = 10,
        token_eviction_threshold: float = 0.8,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize working context block.
        
        Args:
            initial_data: Initial context data (legacy support)
            name: Block name
            max_topics: Maximum number of topics to keep
            token_eviction_threshold: Token usage threshold for eviction (0.8 = 80%)
            logger: Optional logger instance
        """
        super().__init__(name, is_read_only=False, logger=logger)
        
        # Legacy context data (for backwards compatibility)
        self.context_data = initial_data or {}
        
        # Topic FIFO queue
        self.topics: deque[Topic] = deque(maxlen=max_topics)
        self.max_topics = max_topics
        self.token_eviction_threshold = token_eviction_threshold
        
        # Statistics
        self.total_topics_processed = 0
        self.total_topics_evicted = 0
    
    def add_topic(self, topic: Topic):
        """
        Add a topic to the working context.
        
        Args:
            topic: Topic to add
        """
        # Update topic token count if not set
        if topic.token_count == 0 and self._token_counter:
            topic_message = topic.to_message()
            topic.token_count = self._token_counter.count_message_tokens(topic_message)
        
        # Add to queue
        self.topics.append(topic)
        self.total_topics_processed += 1
        
        self.logger.info(f"Added topic '{topic.id}' to working context. Queue size: {len(self.topics)}")
    
    def evict_oldest_topic(self) -> Optional[Topic]:
        """
        Evict the oldest topic from the queue.
        
        Returns:
            Evicted topic if any, None if queue is empty
        """
        if self.topics:
            evicted_topic = self.topics.popleft()
            self.total_topics_evicted += 1
            self.logger.info(f"Evicted oldest topic '{evicted_topic.id}' from working context")
            return evicted_topic
        return None
    
    def check_and_evict_for_tokens(self, max_tokens: int) -> List[Topic]:
        """
        Check token usage and evict topics if necessary.
        
        Args:
            max_tokens: Maximum allowed tokens in the context
            
        Returns:
            List of evicted topics
        """
        evicted_topics = []
        
        # Calculate current token usage
        current_tokens = self.get_token_count()
        threshold_tokens = int(max_tokens * self.token_eviction_threshold)
        
        # Evict topics until we're below threshold
        while current_tokens >= threshold_tokens and self.topics:
            evicted_topic = self.evict_oldest_topic()
            if evicted_topic:
                evicted_topics.append(evicted_topic)
                current_tokens = self.get_token_count()
                self.logger.info(
                    f"Evicted topic for token management. "
                    f"Tokens: {current_tokens}/{max_tokens} "
                    f"(threshold: {threshold_tokens})"
                )
        
        return evicted_topics
    
    def get_topics(self) -> List[Topic]:
        """Get all topics in the queue (oldest to newest)."""
        return list(self.topics)
    
    def clear_topics(self):
        """Clear all topics from the queue."""
        cleared_count = len(self.topics)
        self.topics.clear()
        self.logger.info(f"Cleared {cleared_count} topics from working context")
    
    def update_context(self, key: str, value: Any):
        """
        Update legacy context data.
        
        Args:
            key: Context key
            value: Context value
        """
        self.context_data[key] = value
        self.logger.debug(f"Updated context: {key} = {value}")
    
    def remove_context(self, key: str):
        """
        Remove legacy context data.
        
        Args:
            key: Context key to remove
        """
        if key in self.context_data:
            del self.context_data[key]
            self.logger.debug(f"Removed context key: {key}")
    
    def get_context_data(self) -> Dict[str, Any]:
        """Get legacy context data."""
        return self.context_data.copy()
    
    def to_messages(self) -> List[Message]:
        """
        Convert working context to list of messages.
        
        Returns:
            List of system messages for topics and context data
        """
        messages = []
        
        # Add topic messages (oldest to newest)
        for topic in self.topics:
            topic_message = topic.to_message()
            messages.append(topic_message)
        
        # Add legacy context data if any
        if self.context_data:
            context_content = "Working Context:\n"
            for key, value in self.context_data.items():
                context_content += f"- {key}: {value}\n"
            
            context_message = SystemMessage(content=context_content.strip())
            messages.append(context_message)
        
        return messages
    
    def get_token_count(self) -> int:
        """Get total token count for this block."""
        if not self._token_counter:
            return 0
        
        total_tokens = 0
        
        # Count tokens from topics
        for topic in self.topics:
            if topic.token_count > 0:
                total_tokens += topic.token_count
            else:
                # Calculate on demand if not cached
                topic_message = topic.to_message()
                topic_tokens = self._token_counter.count_message_tokens(topic_message)
                topic.token_count = topic_tokens
                total_tokens += topic_tokens
        
        # Count tokens from legacy context data
        if self.context_data:
            context_content = "Working Context:\n"
            for key, value in self.context_data.items():
                context_content += f"- {key}: {value}\n"
            # Use encoding directly for string content
            total_tokens += len(self._token_counter.encoding.encode(context_content.strip()))
        
        return total_tokens
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get working context statistics."""
        return {
            "current_topics": len(self.topics),
            "max_topics": self.max_topics,
            "total_topics_processed": self.total_topics_processed,
            "total_topics_evicted": self.total_topics_evicted,
            "legacy_context_keys": len(self.context_data),
            "token_count": self.get_token_count(),
            "token_eviction_threshold": self.token_eviction_threshold,
            "topics": [
                {
                    "id": topic.id,
                    "summary": topic.summary[:100] + "..." if len(topic.summary) > 100 else topic.summary,
                    "message_count": topic.message_count,
                    "token_count": topic.token_count,
                    "timestamp": topic.timestamp
                }
                for topic in self.topics
            ]
        } 