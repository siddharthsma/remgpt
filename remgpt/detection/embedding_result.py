"""
Embedding result data structure.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class EmbeddingResult:
    """Result of message embedding."""
    embedding: np.ndarray
    message_id: str
    timestamp: float 