import json
import uuid
import asyncio
from typing import List, Dict, Any, AsyncGenerator, Optional
from ..base import BaseLLMClient
from ..events import Event, EventType


class OpenAIClient(BaseLLMClient):
    """OpenAI LLM client implementation."""
    
    SUPPORTED_MODELS = [
        "gpt-4", "gpt-4-turbo", "gpt-4-turbo-preview", 
        "gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4o", "gpt-4o-mini"
    ]
    
    def __init__(self, model_name: str, api_key: str, **kwargs):
        """
        Initialize OpenAI client.
        
        Args:
            model_name: OpenAI model name
            api_key: OpenAI API key
            **kwargs: Additional OpenAI parameters
        """
        super().__init__(model_name, **kwargs)
        self.api_key = api_key
        self.client = None
        self._tool_calls_in_progress = {}
        
        # Initialize OpenAI client
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the OpenAI client."""
        try:
            import openai
            self.client = openai.OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("OpenAI package not installed. Install with: pip install openai")
    
    async def generate_stream(self, messages: List[Dict[str, Any]], **kwargs) -> AsyncGenerator[Event, None]:
        """Generate streaming response from OpenAI."""
        if not self.validate_messages(messages):
            yield Event(type=EventType.RUN_ERROR, error="Invalid message format")
            return
            
        # Emit run started event
        yield Event(type=EventType.RUN_STARTED)
        
        try:
            # Format messages for OpenAI
            formatted_messages = self.format_messages(messages)
            
            # Set up generation parameters
            params = {
                "model": self.model_name,
                "messages": formatted_messages,
                "stream": True,
                **kwargs
            }
            
            # Add tools if available
            if self.supports_tools() and "tools" in kwargs and kwargs["tools"]:
                params["tools"] = kwargs["tools"]
                params["tool_choice"] = "auto"
            
            # Create streaming completion
            stream = self.client.chat.completions.create(**params)
            
            current_tool_calls = {}
            current_content = ""
            
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta:
                    delta = chunk.choices[0].delta
                    
                    # Handle text content
                    if delta.content:
                        if not current_content:
                            yield Event(type=EventType.TEXT_MESSAGE_START)
                        
                        current_content += delta.content
                        yield Event(
                            type=EventType.TEXT_MESSAGE_CONTENT,
                            content=delta.content
                        )
                    
                    # Handle tool calls
                    if delta.tool_calls:
                        for i, tool_call in enumerate(delta.tool_calls):
                            # Use index as the key since OpenAI sends ID only in first chunk
                            call_key = tool_call.id if tool_call.id else f"index_{i}"
                            
                            # If we have an ID, use it; otherwise find the existing call by index
                            if not tool_call.id and len(current_tool_calls) > i:
                                # Get the actual call_id from the existing tool calls (by order)
                                call_key = list(current_tool_calls.keys())[i]
                            
                            if call_key not in current_tool_calls:
                                # New tool call - only create if we have a function name
                                if tool_call.function and tool_call.function.name:
                                    current_tool_calls[call_key] = {
                                        "name": tool_call.function.name,
                                        "args": ""
                                    }
                                    
                                    yield Event(
                                        type=EventType.TOOL_CALL_START,
                                        tool_call_id=call_key,
                                        tool_name=tool_call.function.name
                                    )
                            
                            # Accumulate arguments
                            if tool_call.function and tool_call.function.arguments:
                                current_tool_calls[call_key]["args"] += tool_call.function.arguments
            
            # End text message if we had content
            if current_content:
                yield Event(type=EventType.TEXT_MESSAGE_END, content=current_content)
            
            # Process completed tool calls
            for call_id, call_info in current_tool_calls.items():
                try:
                    args = json.loads(call_info["args"]) if call_info["args"] else {}
                    yield Event(
                        type=EventType.TOOL_CALL_ARGS,
                        tool_call_id=call_id,
                        tool_args=args
                    )
                    yield Event(
                        type=EventType.TOOL_CALL_END,
                        tool_call_id=call_id
                    )
                except json.JSONDecodeError:
                    yield Event(
                        type=EventType.RUN_ERROR,
                        error=f"Invalid tool arguments for call {call_id}: {call_info['args']}"
                    )
            
            yield Event(type=EventType.RUN_FINISHED)
            
        except Exception as e:
            yield Event(type=EventType.RUN_ERROR, error=str(e))
    
    def send_tool_result(self, tool_call_id: str, result: Any) -> None:
        """Send tool result back to OpenAI (stored for next generation)."""
        # Store tool result for next message generation
        # This would typically be handled by the orchestrator adding tool results to messages
        pass
    
    def supports_tools(self) -> bool:
        """OpenAI supports tools/function calling."""
        return True
    
    def get_supported_models(self) -> List[str]:
        """Get supported OpenAI models."""
        return self.SUPPORTED_MODELS.copy()
    
    def format_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format messages for OpenAI API."""
        formatted = []
        
        for message in messages:
            formatted_msg = {
                "role": message["role"],
                "content": message["content"]
            }
            
            # Add tool call information if present
            if "tool_calls" in message:
                formatted_msg["tool_calls"] = message["tool_calls"]
            
            if "tool_call_id" in message:
                formatted_msg["tool_call_id"] = message["tool_call_id"]
            
            formatted.append(formatted_msg)
        
        return formatted 