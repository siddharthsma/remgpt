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
from .topic_similarity_service import TopicSimilarityService
from ..storage.vector_database import VectorDatabase

if TYPE_CHECKING:
    from ..llm import BaseLLMClient


class LLMContextManager:
    """
    Manages LLM context including token limits, message queues, and working context.
    Enhanced with intelligent topic management and similarity-based recall.
    """
    
    def __init__(
        self, 
        max_tokens: int, 
        vector_database: Optional[VectorDatabase] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize context manager.
        
        Args:
            max_tokens: Maximum tokens allowed in context
            vector_database: Optional vector database for topic storage and retrieval
            logger: Optional logger instance
        """
        self.max_tokens = max_tokens
        self.vector_database = vector_database
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize token counter and context
        self.token_counter = TokenCounter()
        self.context = LLMContext(self.token_counter, logger=self.logger)
        
        # Initialize topic similarity service
        self.topic_similarity_service = TopicSimilarityService(logger=self.logger)
        
        self.logger.info(f"LLMContextManager initialized with max_tokens={max_tokens}")
    
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
        Save the current conversation as a topic with intelligent similarity checking.
        
        This method now includes:
        1. Check for similar topics in working context (for updating)
        2. Create proper embeddings for similarity comparison
        3. Update existing topics if high similarity is found
        4. Create new topics if no similar topics exist
        
        Args:
            topic_summary: Summary of the current conversation topic
            topic_key_facts: List of key facts from the conversation
            
        Returns:
            Topic ID that was created or updated
        """
        import uuid
        import time
        
        # Get messages from FIFO queue
        queue_messages = self.context.fifo_queue.get_messages()
        
        if not queue_messages:
            self.logger.warning("No messages in FIFO queue to save as topic")
            return ""
        
        # Create embedding for current conversation
        conversation_embedding = self.topic_similarity_service.create_embedding_from_messages(queue_messages)
        
        # Check for similar topics in working context for potential updating
        working_topics = self.context.working_context.get_topics()
        similar_topic_result = self.topic_similarity_service.find_similar_topic_in_working_context(
            conversation_embedding, working_topics
        )
        
        if similar_topic_result:
            similar_topic, similarity = similar_topic_result
            
            # Check if similarity is high enough to update existing topic
            if self.topic_similarity_service.should_update_existing_topic(
                conversation_embedding, similar_topic
            ):
                # Update existing topic
                updated_topic = self.topic_similarity_service.merge_topics(
                    similar_topic, topic_summary, topic_key_facts or [], queue_messages
                )
                
                # Flush the FIFO queue
                self.flush_fifo_queue()
                
                self.logger.info(
                    f"Updated existing topic '{updated_topic.id}' with new conversation "
                    f"(similarity: {similarity:.3f})"
                )
                return updated_topic.id
        
        # No similar topic found or similarity too low - create new topic
        topic_id = f"topic_{int(time.time())}_{str(uuid.uuid4())[:8]}"
        
        # Create topic with proper embedding
        topic = Topic(
            id=topic_id,
            summary=topic_summary,
            key_facts=topic_key_facts or [],
            mean_embedding=conversation_embedding,
            original_messages=queue_messages.copy(),
            timestamp=time.time(),
            message_count=len(queue_messages)
        )
        
        # Add topic to working context
        self.context.working_context.add_topic(topic)
        
        # Flush the FIFO queue
        self.flush_fifo_queue()
        
        self.logger.info(f"Saved current conversation as new topic '{topic_id}' and flushed FIFO queue")
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
    
    def update_topic(self, topic_id: str, additional_summary: str, additional_key_facts: List[str] = None) -> str:
        """
        Update an existing topic with additional information.
        
        This method is designed to be called by the LLM via function calling.
        
        Args:
            topic_id: ID of the topic to update
            additional_summary: Additional summary to merge
            additional_key_facts: Additional key facts to add
            
        Returns:
            Updated topic ID
        """
        # Get messages from FIFO queue
        queue_messages = self.context.fifo_queue.get_messages()
        
        # Find the topic in working context
        working_topics = self.context.working_context.get_topics()
        target_topic = None
        
        for topic in working_topics:
            if topic.id == topic_id:
                target_topic = topic
                break
        
        if not target_topic:
            self.logger.error(f"Topic {topic_id} not found in working context")
            return ""
        
        # Update the topic using similarity service
        updated_topic = self.topic_similarity_service.merge_topics(
            target_topic, additional_summary, additional_key_facts or [], queue_messages
        )
        
        # Flush the FIFO queue
        self.flush_fifo_queue()
        
        self.logger.info(f"Updated topic '{topic_id}' with additional information")
        return updated_topic.id
    
    async def recall_similar_topic(self, user_message: str) -> Optional[str]:
        """
        Recall a similar topic from vector database and load it into working context.
        
        This method is designed to be called by the LLM via function calling.
        
        Args:
            user_message: The user message to find similar topics for
            
        Returns:
            ID of recalled topic, or None if no similar topic found
        """
        if not self.vector_database:
            self.logger.debug("No vector database configured for topic recall")
            return None
        
        # Create embedding for user message
        message_embedding = self.topic_similarity_service.create_embedding_from_text(user_message)
        
        # First check if similar topic already exists in working context
        working_topics = self.context.working_context.get_topics()
        similar_in_working = self.topic_similarity_service.find_similar_topic_in_working_context(
            message_embedding, working_topics
        )
        
        if similar_in_working:
            topic, similarity = similar_in_working
            self.logger.info(f"Similar topic already in working context: {topic.id} (similarity: {similarity:.3f})")
            return topic.id
        
        # Search vector database for similar topics
        similar_topic_result = await self.topic_similarity_service.find_similar_topic_in_vector_database(
            message_embedding, self.vector_database
        )
        
        if similar_topic_result:
            similar_topic, similarity = similar_topic_result
            
            # Add the similar topic to working context
            self.context.working_context.add_topic(similar_topic)
            
            self.logger.info(
                f"Recalled similar topic from vector database: {similar_topic.id} "
                f"(similarity: {similarity:.3f})"
            )
            return similar_topic.id
        
        self.logger.debug("No similar topics found for recall")
        return None 