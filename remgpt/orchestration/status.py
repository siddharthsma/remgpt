"""
Orchestrator status enumeration.
"""

from enum import Enum


class OrchestratorStatus(str, Enum):
    """Status of the orchestrator."""
    IDLE = "idle"
    PROCESSING = "processing"
    ERROR = "error" 