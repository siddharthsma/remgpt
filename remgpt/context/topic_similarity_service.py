"""
Topic similarity service for intelligent topic management.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
from sentence_transformers import SentenceTransformer

from ..summarization.topic import Topic
from ..types import Message
from ..storage.vector_database import VectorDatabase


class TopicSimilarityService:
    """Service for calculating topic similarities and managing topic updates."""
    
    def __init__(
        self, 
        model_name: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.7,
        update_threshold: float = 0.8,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize topic similarity service.
        
        Args:
            model_name: Sentence transformer model name
            similarity_threshold: Threshold for considering topics similar for recall
            update_threshold: Higher threshold for considering topics similar enough to update
            logger: Optional logger instance
        """
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self.update_threshold = update_threshold
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize sentence transformer
        self.model = SentenceTransformer(model_name)
        self.logger.info(f"Loaded sentence transformer model for topic similarity: {model_name}")
    
    def create_embedding_from_messages(self, messages: List[Message]) -> np.ndarray:
        """
        Create an embedding from a list of messages.
        
        Args:
            messages: List of messages to embed
            
        Returns:
            Mean embedding of all messages
        """
        if not messages:
            return np.zeros(384)  # Return zero vector for empty messages
        
        embeddings = []
        for message in messages:
            # Extract text content from message
            if isinstance(message.content, str):
                text = message.content
            elif isinstance(message.content, list):
                # Extract text from structured content
                text_parts = []
                for item in message.content:
                    if hasattr(item, 'text'):
                        text_parts.append(item.text)
                    elif isinstance(item, dict) and 'text' in item:
                        text_parts.append(item['text'])
                text = " ".join(text_parts)
            else:
                text = str(message.content) if message.content else ""
            
            if text.strip():  # Only process non-empty text
                embedding = self.model.encode(text, convert_to_numpy=True)
                embeddings.append(embedding)
        
        if not embeddings:
            return np.zeros(384)
        
        # Return mean embedding
        return np.mean(embeddings, axis=0)
    
    def create_embedding_from_text(self, text: str) -> np.ndarray:
        """
        Create an embedding from text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        if not text.strip():
            return np.zeros(384)
        
        return self.model.encode(text, convert_to_numpy=True)
    
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Cosine similarity score
        """
        # Normalize embeddings
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Calculate cosine similarity
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        return float(similarity)
    
    def find_similar_topic_in_working_context(
        self, 
        conversation_embedding: np.ndarray, 
        working_topics: List[Topic]
    ) -> Optional[Tuple[Topic, float]]:
        """
        Find the most similar topic in working context.
        
        Args:
            conversation_embedding: Embedding of current conversation
            working_topics: List of topics in working context
            
        Returns:
            Tuple of (most_similar_topic, similarity_score) or None if no similar topic found
        """
        if not working_topics:
            return None
        
        best_match = None
        best_similarity = 0.0
        
        for topic in working_topics:
            similarity = self.calculate_similarity(conversation_embedding, topic.mean_embedding)
            if similarity > best_similarity and similarity >= self.similarity_threshold:
                best_similarity = similarity
                best_match = topic
        
        if best_match:
            self.logger.info(f"Found similar topic in working context: {best_match.id} (similarity: {best_similarity:.3f})")
            return best_match, best_similarity
        
        return None
    
    def should_update_existing_topic(
        self, 
        conversation_embedding: np.ndarray, 
        existing_topic: Topic
    ) -> bool:
        """
        Determine if a conversation is similar enough to update an existing topic.
        
        Args:
            conversation_embedding: Embedding of current conversation
            existing_topic: Existing topic to compare against
            
        Returns:
            True if topic should be updated, False if new topic should be created
        """
        similarity = self.calculate_similarity(conversation_embedding, existing_topic.mean_embedding)
        should_update = similarity >= self.update_threshold
        
        self.logger.info(
            f"Topic update check: similarity={similarity:.3f}, "
            f"threshold={self.update_threshold}, should_update={should_update}"
        )
        
        return should_update
    
    async def find_similar_topic_in_vector_database(
        self, 
        message_embedding: np.ndarray, 
        vector_database: Optional[VectorDatabase],
        limit: int = 5
    ) -> Optional[Tuple[Topic, float]]:
        """
        Find similar topics in vector database.
        
        Args:
            message_embedding: Embedding of new message
            vector_database: Vector database to search
            limit: Maximum number of results to consider
            
        Returns:
            Tuple of (best_matching_topic, similarity_score) or None
        """
        if not vector_database:
            self.logger.debug("No vector database configured for topic search")
            return None
        
        try:
            # Search for similar topics
            similar_topics = await vector_database.search_similar_topics(
                query_embedding=message_embedding,
                limit=limit,
                score_threshold=self.similarity_threshold
            )
            
            if not similar_topics:
                self.logger.debug("No similar topics found in vector database")
                return None
            
            # Return the best match (highest similarity)
            best_topic, best_similarity = similar_topics[0]
            self.logger.info(
                f"Found similar topic in vector database: {best_topic.id} "
                f"(similarity: {best_similarity:.3f})"
            )
            
            return best_topic, best_similarity
            
        except Exception as e:
            self.logger.error(f"Error searching vector database: {e}")
            return None
    
    def merge_topics(
        self, 
        existing_topic: Topic, 
        new_summary: str, 
        new_key_facts: List[str],
        new_messages: List[Message]
    ) -> Topic:
        """
        Merge new conversation data into an existing topic.
        
        Args:
            existing_topic: Topic to update
            new_summary: Summary of new conversation
            new_key_facts: Key facts from new conversation
            new_messages: Messages from new conversation
            
        Returns:
            Updated topic
        """
        # Create embedding for new conversation
        new_embedding = self.create_embedding_from_messages(new_messages)
        
        # Calculate updated mean embedding (weighted average)
        total_messages = existing_topic.message_count + len(new_messages)
        if total_messages > 0:
            weight_existing = existing_topic.message_count / total_messages
            weight_new = len(new_messages) / total_messages
            
            updated_embedding = (
                existing_topic.mean_embedding * weight_existing +
                new_embedding * weight_new
            )
        else:
            updated_embedding = existing_topic.mean_embedding
        
        # Merge key facts (avoid duplicates)
        all_key_facts = existing_topic.key_facts.copy()
        for fact in new_key_facts:
            if fact not in all_key_facts:
                all_key_facts.append(fact)
        
        # Create updated summary
        updated_summary = f"{existing_topic.summary}. Additional discussion: {new_summary}"
        
        # Update topic fields
        existing_topic.summary = updated_summary
        existing_topic.key_facts = all_key_facts
        existing_topic.mean_embedding = updated_embedding
        existing_topic.message_count = total_messages
        existing_topic.original_messages.extend(new_messages)
        
        self.logger.info(f"Merged conversation into existing topic: {existing_topic.id}")
        return existing_topic 