"""
Topic data structure for context management.
"""

import time
from typing import List, Dict, Any
from dataclasses import dataclass, field
import numpy as np

from ..types import Message, SystemMessage


@dataclass
class Topic:
    """
    Represents a summarized topic with metadata.
    
    This is what gets stored in the WorkingContextBlock as a FIFO queue item.
    """
    
    id: str
    summary: str
    key_facts: List[str]
    mean_embedding: np.ndarray
    original_messages: List[Message]
    timestamp: float
    message_count: int
    token_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_message(self) -> SystemMessage:
        """
        Convert topic to a system message for context.
        
        Returns:
            SystemMessage containing the topic summary
        """
        content_parts = [f"Topic: {self.summary}"]
        
        if self.key_facts:
            content_parts.append("Key facts:")
            for fact in self.key_facts:
                content_parts.append(f"- {fact}")
        
        content = "\n".join(content_parts)
        
        return SystemMessage(content=content)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert topic to dictionary for storage."""
        return {
            "id": self.id,
            "summary": self.summary,
            "key_facts": self.key_facts,
            "mean_embedding": self.mean_embedding.tolist(),
            "timestamp": self.timestamp,
            "message_count": self.message_count,
            "token_count": self.token_count,
            "metadata": self.metadata,
            # Store original message content for reference (not embeddings)
            "original_message_content": [
                {
                    "role": msg.role.value,
                    "content": str(msg.content)[:500] + "..." if len(str(msg.content)) > 500 else str(msg.content)
                }
                for msg in self.original_messages[:10]  # Limit to first 10 messages
            ]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Topic":
        """Create topic from dictionary."""
        # Note: This doesn't restore original_messages, just metadata
        return cls(
            id=data["id"],
            summary=data["summary"],
            key_facts=data["key_facts"],
            mean_embedding=np.array(data["mean_embedding"]),
            original_messages=[],  # Not restored from storage
            timestamp=data["timestamp"],
            message_count=data["message_count"],
            token_count=data.get("token_count", 0),
            metadata=data.get("metadata", {})
        ) 