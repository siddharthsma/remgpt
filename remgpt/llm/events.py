from enum import Enum
from typing import Any, Dict, Optional, Union
from dataclasses import dataclass


class EventType(str, Enum):
    # LLM Client Events
    TEXT_MESSAGE_START = "TEXT_MESSAGE_START"
    TEXT_MESSAGE_CONTENT = "TEXT_MESSAGE_CONTENT"
    TEXT_MESSAGE_END = "TEXT_MESSAGE_END"
    TOOL_CALL_START = "TOOL_CALL_START"
    TOOL_CALL_ARGS = "TOOL_CALL_ARGS"
    TOOL_CALL_END = "TOOL_CALL_END"
    STATE_SNAPSHOT = "STATE_SNAPSHOT"
    STATE_DELTA = "STATE_DELTA"
    MESSAGES_SNAPSHOT = "MESSAGES_SNAPSHOT"
    RAW = "RAW"
    CUSTOM = "CUSTOM"
    RUN_STARTED = "RUN_STARTED"
    RUN_FINISHED = "RUN_FINISHED"
    RUN_ERROR = "RUN_ERROR"
    STEP_STARTED = "STEP_STARTED"
    STEP_FINISHED = "STEP_FINISHED"


@dataclass
class Event:
    """Represents an event emitted during processing."""
    
    type: Union[EventType, str]  # Allow both EventType enum and string for flexibility
    data: Optional[Dict[str, Any]] = None
    content: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_name: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    raw_data: Optional[Any] = None
    timestamp: Optional[float] = None
    
    def __post_init__(self):
        """Validate event data consistency."""
        # Convert string types to EventType if possible
        if isinstance(self.type, str):
            try:
                self.type = EventType(self.type)
            except ValueError:
                # Allow custom string types for flexibility
                pass
        
        # Validate LLM client specific events
        if self.type in [EventType.TOOL_CALL_START, EventType.TOOL_CALL_ARGS, EventType.TOOL_CALL_END]:
            if not self.tool_call_id:
                raise ValueError(f"tool_call_id is required for {self.type}")
                
        if self.type == EventType.TOOL_CALL_START and not self.tool_name:
            raise ValueError("tool_name is required for TOOL_CALL_START events")
            
        if self.type == EventType.TOOL_CALL_ARGS and self.tool_args is None:
            raise ValueError("tool_args is required for TOOL_CALL_ARGS events")
            
        if self.type == EventType.RUN_ERROR and not self.error:
            raise ValueError("error is required for RUN_ERROR events") 