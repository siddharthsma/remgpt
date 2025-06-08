"""
Stream event data structure.
"""

from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class StreamEvent:
    """Event emitted during streaming."""
    type: str
    data: Dict[str, Any]
    timestamp: float 