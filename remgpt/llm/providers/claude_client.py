import json
import uuid
import asyncio
from typing import List, Dict, Any, AsyncGenerator, Optional
from ..base import BaseLLMClient
from ..events import Event, EventType


class ClaudeClient(BaseLLMClient):
    """Claude (Anthropic) LLM client implementation."""
    
    # Only Claude models that support function calling
    SUPPORTED_MODELS = [
        "claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022",
        "claude-3-opus-20240229", "claude-3-sonnet-20240229"
    ]
    
    def __init__(self, model_name: str, api_key: str, **kwargs):
        """Initialize Claude client."""
        super().__init__(model_name, **kwargs)
        self.api_key = api_key
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the Anthropic client."""
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)
        except ImportError:
            raise ImportError("Anthropic package not installed. Install with: pip install anthropic")
    
    def _convert_openai_tools_to_claude(self, openai_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert OpenAI tool format to Claude tool format."""
        claude_tools = []
        for tool in openai_tools:
            if tool.get("type") == "function" and "function" in tool:
                func = tool["function"]
                claude_tool = {
                    "name": func["name"],
                    "description": func["description"],
                    "input_schema": func.get("parameters", {})
                }
                claude_tools.append(claude_tool)
        return claude_tools

    async def generate_stream(self, messages: List[Dict[str, Any]], **kwargs) -> AsyncGenerator[Event, None]:
        """Generate streaming response from Claude."""
        if not self.validate_messages(messages):
            yield Event(type=EventType.RUN_ERROR, error="Invalid message format")
            return
            
        # Emit run started event
        yield Event(type=EventType.RUN_STARTED)
        
        try:
            # Format messages for Anthropic
            formatted_messages = self.format_messages(messages)
            
            # Extract system message if present
            system_message = None
            if formatted_messages and formatted_messages[0]["role"] == "system":
                system_message = formatted_messages[0]["content"]
                formatted_messages = formatted_messages[1:]
            
            # Set up generation parameters
            params = {
                "model": self.model_name,
                "messages": formatted_messages,
                "max_tokens": kwargs.get("max_tokens", 4096),
                "stream": True
            }
            
            if system_message:
                params["system"] = system_message
                
            # Add tools if available (convert from OpenAI format)
            if self.supports_tools() and "tools" in kwargs and kwargs["tools"]:
                claude_tools = self._convert_openai_tools_to_claude(kwargs["tools"])
                params["tools"] = claude_tools
            
            # Add other parameters
            if "temperature" in kwargs:
                params["temperature"] = kwargs["temperature"]
            
            # Create streaming message
            stream = self.client.messages.create(**params)
            
            current_content = ""
            current_tool_calls = {}
            
            for event in stream:
                if event.type == "message_start":
                    continue
                    
                elif event.type == "content_block_start":
                    if event.content_block.type == "text":
                        yield Event(type=EventType.TEXT_MESSAGE_START)
                    elif event.content_block.type == "tool_use":
                        tool_call_id = event.content_block.id
                        tool_name = event.content_block.name
                        
                        current_tool_calls[tool_call_id] = {
                            "name": tool_name,
                            "args": {}
                        }
                        
                        yield Event(
                            type=EventType.TOOL_CALL_START,
                            tool_call_id=tool_call_id,
                            tool_name=tool_name
                        )
                
                elif event.type == "content_block_delta":
                    if event.delta.type == "text_delta":
                        current_content += event.delta.text
                        yield Event(
                            type=EventType.TEXT_MESSAGE_CONTENT,
                            content=event.delta.text
                        )
                
                elif event.type == "content_block_stop":
                    if hasattr(event, 'content_block'):
                        if event.content_block.type == "text":
                            yield Event(type=EventType.TEXT_MESSAGE_END, content=current_content)
                        elif event.content_block.type == "tool_use":
                            tool_call_id = event.content_block.id
                            tool_args = event.content_block.input
                            
                            yield Event(
                                type=EventType.TOOL_CALL_ARGS,
                                tool_call_id=tool_call_id,
                                tool_args=tool_args
                            )
                            yield Event(
                                type=EventType.TOOL_CALL_END,
                                tool_call_id=tool_call_id
                            )
                
                elif event.type == "message_stop":
                    break
            
            yield Event(type=EventType.RUN_FINISHED)
            
        except Exception as e:
            yield Event(type=EventType.RUN_ERROR, error=str(e))
    
    def send_tool_result(self, tool_call_id: str, result: Any) -> None:
        """Send tool result back to Claude."""
        pass
    
    def supports_tools(self) -> bool:
        """Claude supports tools/function calling."""
        return True
    
    def get_supported_models(self) -> List[str]:
        """Get supported Claude models."""
        return self.SUPPORTED_MODELS.copy()
    
    def format_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format messages for Anthropic API."""
        formatted = []
        
        for message in messages:
            msg_role = message["role"] 
            msg_content = message["content"]
            
            # Handle tool results for Claude
            if msg_role == "tool":
                formatted_msg = {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": message.get("tool_call_id"),
                            "content": msg_content
                        }
                    ]
                }
            else:
                formatted_msg = {
                    "role": msg_role,
                    "content": msg_content
                }
            
            formatted.append(formatted_msg)
        
        return formatted 