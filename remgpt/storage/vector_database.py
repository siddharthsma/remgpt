"""
Abstract base class for vector databases.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
from ..summarization.topic import Topic


class VectorDatabase(ABC):
    """Abstract base class for vector databases."""
    
    @abstractmethod
    async def store_topic(self, topic: Topic) -> bool:
        """
        Store a topic in the vector database.
        
        Args:
            topic: Topic to store
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def search_similar_topics(
        self, 
        query_embedding: np.ndarray, 
        limit: int = 5,
        score_threshold: float = 0.7
    ) -> List[Tuple[Topic, float]]:
        """
        Search for topics similar to the query embedding.
        
        Args:
            query_embedding: Query vector
            limit: Maximum number of results
            score_threshold: Minimum similarity score
            
        Returns:
            List of (topic, similarity_score) tuples
        """
        pass
    
    @abstractmethod
    async def list_topics(
        self, 
        limit: int = 100,
        offset: int = 0
    ) -> List[Topic]:
        """
        List topics in the database.
        
        Args:
            limit: Maximum number of topics to return
            offset: Number of topics to skip
            
        Returns:
            List of topics
        """
        pass 