"""
Topic drift detector using sentence embeddings and Page-Hinkley test.
"""

import numpy as np
from typing import List, Optional, Tuple
from collections import deque
import logging

from sentence_transformers import SentenceTransformer
from ..core.types import Message
from .embedding_result import EmbeddingResult
from .page_hinkley_test import PageHinkleyTest


class TopicDriftDetector:
    """
    Detects topic drift in message streams using sentence embeddings and Page-Hinkley test.
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.7,
        drift_threshold: float = 0.5,
        alpha: float = 0.05,
        window_size: int = 10,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize topic drift detector.
        
        Args:
            model_name: Sentence transformer model name
            similarity_threshold: Threshold for considering messages similar
            drift_threshold: Page-Hinkley threshold for drift detection
            alpha: Significance level for Page-Hinkley test
            window_size: Number of recent embeddings to keep for comparison
            logger: Optional logger instance
        """
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self.window_size = window_size
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize sentence transformer
        self.model = SentenceTransformer(model_name)
        self.logger.info(f"Loaded sentence transformer model: {model_name}")
        
        # Initialize Page-Hinkley test
        self.ph_test = PageHinkleyTest(threshold=drift_threshold, alpha=alpha)
        
        # Keep recent embeddings for comparison
        self.recent_embeddings: deque = deque(maxlen=window_size)
        self.recent_similarities: deque = deque(maxlen=window_size)
        
    def create_embedding(self, message: Message) -> EmbeddingResult:
        """
        Create embedding for a message.
        
        Args:
            message: Message to embed
            
        Returns:
            EmbeddingResult containing the embedding and metadata
        """
        import time
        
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
        
        # Create embedding
        embedding = self.model.encode(text, convert_to_numpy=True)
        
        result = EmbeddingResult(
            embedding=embedding,
            message_id=getattr(message, 'id', f"msg_{time.time()}"),
            timestamp=time.time()
        )
        
        self.logger.debug(f"Created embedding for message: {text[:50]}...")
        return result
    
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Cosine similarity between embeddings
        """
        # Normalize embeddings
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Calculate cosine similarity
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        return float(similarity)
    
    def detect_drift(self, message: Message) -> Tuple[bool, EmbeddingResult, float]:
        """
        Detect if the new message represents a topic drift.
        
        Enhanced algorithm:
        1. First check if similarity is above a minimum threshold (same topic)
        2. Only use Page-Hinkley for borderline cases
        3. Require sustained low similarity to detect drift
        
        Args:
            message: New message to analyze
            
        Returns:
            Tuple of (drift_detected, embedding_result, similarity_score)
        """
        # Create embedding for new message
        embedding_result = self.create_embedding(message)
        
        # If this is the first message, no drift can be detected
        if len(self.recent_embeddings) == 0:
            self.recent_embeddings.append(embedding_result.embedding)
            self.logger.info("First message processed, no drift detection possible")
            return False, embedding_result, 1.0
        
        # Calculate similarity with recent messages
        similarities = []
        for recent_embedding in self.recent_embeddings:
            similarity = self.calculate_similarity(
                embedding_result.embedding, 
                recent_embedding
            )
            similarities.append(similarity)
        
        # Use mean similarity with recent messages
        mean_similarity = np.mean(similarities)
        self.recent_similarities.append(mean_similarity)
        
        # ENHANCED DRIFT DETECTION LOGIC
        
        # Stage 1: High similarity = definitely same topic
        if mean_similarity >= self.similarity_threshold:
            drift_detected = False
            self.logger.info(
                f"High similarity detected ({mean_similarity:.3f} >= {self.similarity_threshold}), "
                f"no drift - same topic"
            )
        
        # Stage 2: Very low similarity = potential different topic 
        elif mean_similarity <= 0.25:  # Lowered from 0.3 to be less sensitive
            # Require more sustained evidence for drift
            recent_similarities = list(self.recent_similarities)[-4:]  # Look at last 4 messages
            very_low_count = sum(1 for s in recent_similarities if s <= 0.3)
            avg_recent_similarity = np.mean(recent_similarities) if recent_similarities else mean_similarity
            
            # Require both: multiple low scores AND low average
            drift_detected = very_low_count >= 3 and avg_recent_similarity < 0.28
            
            if drift_detected:
                self.logger.info(
                    f"Strong evidence of topic drift: current={mean_similarity:.3f}, "
                    f"low_count={very_low_count}/4, avg_recent={avg_recent_similarity:.3f}"
                )
            else:
                self.logger.info(
                    f"Low similarity ({mean_similarity:.3f}) but insufficient evidence: "
                    f"low_count={very_low_count}/4, avg_recent={avg_recent_similarity:.3f}"
                )
        
        # Stage 3: Medium similarity = use Page-Hinkley for borderline cases
        else:
            # Only use Page-Hinkley for ambiguous cases (0.25 - 0.6 range)
            ph_drift = self.ph_test.add_sample(mean_similarity)
            
            # Much stricter constraints for medium similarities
            very_low_similarities = [s for s in self.recent_similarities if s < 0.35]
            sustained_drift = (
                len(very_low_similarities) >= min(4, len(self.recent_similarities) // 2) and
                np.mean(self.recent_similarities) < 0.4
            )
            
            drift_detected = ph_drift and sustained_drift
            
            self.logger.info(
                f"Borderline similarity ({mean_similarity:.3f}), "
                f"PH_test={ph_drift}, low_similarities={len(very_low_similarities)}, "
                f"avg_similarity={np.mean(self.recent_similarities):.3f}, "
                f"final_drift={drift_detected}"
            )
        
        # Add current embedding to recent embeddings
        self.recent_embeddings.append(embedding_result.embedding)
        
        # Update drift counter if detected
        if drift_detected:
            self.logger.info(f"TOPIC DRIFT CONFIRMED: similarity={mean_similarity:.3f}")
        
        return drift_detected, embedding_result, mean_similarity
    
    def get_mean_embedding(self) -> Optional[np.ndarray]:
        """
        Get the mean embedding of recent messages.
        
        Returns:
            Mean embedding or None if no embeddings available
        """
        if len(self.recent_embeddings) == 0:
            return None
        
        embeddings_array = np.array(list(self.recent_embeddings))
        return np.mean(embeddings_array, axis=0)
    
    def reset(self):
        """Reset the drift detector state."""
        self.ph_test.reset()
        self.recent_embeddings.clear()
        self.recent_similarities.clear()
        self.logger.info("Topic drift detector reset")
    
    def get_statistics(self) -> dict:
        """Get current drift detection statistics."""
        return {
            "n_messages": len(self.recent_embeddings),
            "recent_similarities": list(self.recent_similarities),
            "mean_recent_similarity": np.mean(self.recent_similarities) if self.recent_similarities else 0.0,
            "ph_cumulative_sum": self.ph_test.cumulative_sum,
            "ph_min_cumulative_sum": self.ph_test.min_cumulative_sum,
            "ph_n_samples": self.ph_test.n_samples
        } 