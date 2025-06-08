"""
QDrant vector database implementation.
"""

import logging
import numpy as np
from typing import List, Tuple, Optional
from .vector_database import VectorDatabase
from ..summarization.topic import Topic


class QdrantVectorDatabase(VectorDatabase):
    """QDrant vector database implementation."""
    
    def __init__(
        self,
        url: str = "http://localhost:6333",
        collection_name: str = "remgpt_topics",
        vector_size: int = 384,  # Default for all-MiniLM-L6-v2
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize QDrant vector database.
        
        Args:
            url: QDrant server URL
            collection_name: Name of the collection
            vector_size: Size of the embedding vectors
            logger: Optional logger instance
        """
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams, PointStruct
            from qdrant_client.http import models
            
            self.client = QdrantClient(url=url)
            self.collection_name = collection_name
            self.vector_size = vector_size
            self.logger = logger or logging.getLogger(__name__)
            
            # Store models for later use
            self.Distance = Distance
            self.VectorParams = VectorParams
            self.PointStruct = PointStruct
            self.models = models
            
            # Ensure collection exists
            self._ensure_collection_exists()
            
        except ImportError:
            raise ImportError(
                "qdrant-client is required for QDrant vector database. "
                "Install with: pip install qdrant-client"
            )
    
    def _ensure_collection_exists(self):
        """Ensure the collection exists, create if it doesn't."""
        try:
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=self.VectorParams(
                        size=self.vector_size,
                        distance=self.Distance.COSINE
                    )
                )
                self.logger.info(f"Created QDrant collection: {self.collection_name}")
            else:
                self.logger.info(f"Using existing QDrant collection: {self.collection_name}")
                
        except Exception as e:
            self.logger.error(f"Failed to ensure collection exists: {e}")
            raise
    
    async def store_topic(self, topic: Topic) -> bool:
        """Store a topic in QDrant."""
        try:
            point = self.PointStruct(
                id=hash(topic.id),  # Use hash of topic ID as point ID
                vector=topic.mean_embedding.tolist(),
                payload={
                    "topic_id": topic.id,
                    "summary": topic.summary,
                    "key_facts": topic.key_facts,
                    "timestamp": topic.timestamp,
                    "message_count": topic.message_count,
                    "token_count": topic.token_count,
                    "metadata": topic.metadata or {}
                }
            )
            
            self.client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )
            
            self.logger.info(f"Stored topic {topic.id} in QDrant")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store topic in QDrant: {e}")
            return False
    
    async def search_similar_topics(
        self, 
        query_embedding: np.ndarray, 
        limit: int = 5,
        score_threshold: float = 0.7
    ) -> List[Tuple[Topic, float]]:
        """Search for similar topics in QDrant."""
        try:
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.tolist(),
                limit=limit,
                score_threshold=score_threshold
            )
            
            results = []
            for hit in search_result:
                payload = hit.payload
                
                # Reconstruct topic from payload
                topic = Topic(
                    id=payload["topic_id"],
                    summary=payload["summary"],
                    key_facts=payload["key_facts"],
                    mean_embedding=query_embedding,  # We don't store the full embedding
                    original_messages=[],  # We don't store original messages
                    timestamp=payload["timestamp"],
                    message_count=payload["message_count"],
                    token_count=payload.get("token_count", 0),
                    metadata=payload.get("metadata", {})
                )
                
                results.append((topic, hit.score))
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to search topics in QDrant: {e}")
            return []
    
    async def list_topics(
        self, 
        limit: int = 100,
        offset: int = 0
    ) -> List[Topic]:
        """List topics from QDrant."""
        try:
            # Use scroll to get topics
            scroll_result = self.client.scroll(
                collection_name=self.collection_name,
                limit=limit,
                offset=offset
            )
            
            topics = []
            for point in scroll_result[0]:  # scroll returns (points, next_page_offset)
                payload = point.payload
                
                # Reconstruct topic from payload
                topic = Topic(
                    id=payload["topic_id"],
                    summary=payload["summary"],
                    key_facts=payload["key_facts"],
                    mean_embedding=np.array(point.vector),
                    original_messages=[],  # We don't store original messages
                    timestamp=payload["timestamp"],
                    message_count=payload["message_count"],
                    token_count=payload.get("token_count", 0),
                    metadata=payload.get("metadata", {})
                )
                
                topics.append(topic)
            
            return topics
            
        except Exception as e:
            self.logger.error(f"Failed to list topics from QDrant: {e}")
            return [] 