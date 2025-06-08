# Library imports
from pydantic import BaseModel, Field
from typing import List, Union, Optional, Literal
from enum import Enum

class MessageRole(str, Enum):
    """Enumeration of possible message roles."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"
    TOOL = "tool"

class ContentType(str, Enum):
    """Enumeration of content types."""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"

class ImageDetail(str, Enum):
    """Image detail level for vision models."""
    LOW = "low"
    AUTO = "auto"
    HIGH = "high"

class ImageContent(BaseModel):
    """Image content block."""
    type: Literal["image"] = "image"
    image_url: dict = Field(..., description="Image URL object with 'url' and optional 'detail'")
    
    class Config:
        json_schema_extra = {
            "example": {
                "type": "image",
                "image_url": {
                    "url": "https://example.com/image.jpg",
                    "detail": "auto"
                }
            }
        }

class TextContent(BaseModel):
    """Text content block."""
    type: Literal["text"] = "text"
    text: str = Field(..., description="The text content")
    
    class Config:
        json_schema_extra = {
            "example": {
                "type": "text",
                "text": "Hello, how can I help you?"
            }
        }

class ToolCall(BaseModel):
    """Tool/function call made by the assistant."""
    id: str = Field(..., description="Unique identifier for the tool call")
    type: Literal["function"] = "function"
    function: dict = Field(..., description="Function name and arguments")
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "call_123",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"location": "San Francisco"}'
                }
            }
        }

class Message(BaseModel):
    """Base message model for LLM conversations."""
    role: MessageRole = Field(..., description="The role of the message sender")
    content: Union[
        str, 
        List[Union[TextContent, ImageContent]], 
        None
    ] = Field(None, description="The content of the message")
    name: Optional[str] = Field(None, description="Optional name of the participant")
    tool_calls: Optional[List[ToolCall]] = Field(None, description="Tool calls made by assistant")
    tool_call_id: Optional[str] = Field(None, description="ID of the tool call this message is responding to")
    
    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "role": "user",
                    "content": "Hello, how are you?"
                },
                {
                    "role": "assistant", 
                    "content": "I'm doing well, thank you! How can I help you today?"
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What's in this image?"
                        },
                        {
                            "type": "image", 
                            "image_url": {
                                "url": "https://example.com/image.jpg"
                            }
                        }
                    ]
                }
            ]
        }

class SystemMessage(Message):
    """System message for setting context and instructions."""
    role: Literal[MessageRole.SYSTEM] = MessageRole.SYSTEM
    content: str = Field(..., description="System instructions")

class UserMessage(Message):
    """User message in the conversation."""
    role: Literal[MessageRole.USER] = MessageRole.USER

class AssistantMessage(Message):
    """Assistant message in the conversation."""
    role: Literal[MessageRole.ASSISTANT] = MessageRole.ASSISTANT

class ToolMessage(Message):
    """Tool/function response message."""
    role: Literal[MessageRole.TOOL] = MessageRole.TOOL
    content: str = Field(..., description="Tool response content")
    tool_call_id: str = Field(..., description="ID of the tool call this responds to")
