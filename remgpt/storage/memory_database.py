"""
In-memory vector database implementation.
"""

import logging
import numpy as np
from typing import List, Tuple, Optional
from .vector_database import VectorDatabase
from ..summarization.topic import Topic


class InMemoryVectorDatabase(VectorDatabase):
    """In-memory vector database implementation for testing and development."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize in-memory vector database."""
        self.topics: List[Topic] = []
        self.logger = logger or logging.getLogger(__name__)
    
    async def store_topic(self, topic: Topic) -> bool:
        """Store a topic in memory."""
        try:
            # Remove existing topic with same ID if it exists
            self.topics = [t for t in self.topics if t.id != topic.id]
            
            # Add new topic
            self.topics.append(topic)
            
            self.logger.info(f"Stored topic {topic.id} in memory")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store topic in memory: {e}")
            return False
    
    async def search_similar_topics(
        self, 
        query_embedding: np.ndarray, 
        limit: int = 5,
        score_threshold: float = 0.7
    ) -> List[Tuple[Topic, float]]:
        """Search for similar topics in memory."""
        try:
            results = []
            
            for topic in self.topics:
                similarity = self._cosine_similarity(query_embedding, topic.mean_embedding)
                
                if similarity >= score_threshold:
                    results.append((topic, similarity))
            
            # Sort by similarity (highest first) and limit results
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:limit]
            
        except Exception as e:
            self.logger.error(f"Failed to search topics in memory: {e}")
            return []
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return np.dot(a, b) / (norm_a * norm_b)
    
    async def list_topics(
        self, 
        limit: int = 100,
        offset: int = 0
    ) -> List[Topic]:
        """List topics from memory."""
        try:
            # Sort by timestamp (newest first) and apply pagination
            sorted_topics = sorted(self.topics, key=lambda t: t.timestamp, reverse=True)
            return sorted_topics[offset:offset + limit]
            
        except Exception as e:
            self.logger.error(f"Failed to list topics from memory: {e}")
            return [] 