import json
import uuid
import asyncio
from typing import List, Dict, Any, AsyncGenerator, Optional
from ..base import BaseLLMClient
from ..events import Event, EventType


class GeminiClient(BaseLLMClient):
    """Google Gemini LLM client implementation."""
    
    # Only Gemini models that support function calling
    SUPPORTED_MODELS = [
        "gemini-1.5-pro", "gemini-1.5-flash"
    ]
    
    def __init__(self, model_name: str, api_key: str, **kwargs):
        """Initialize Gemini client."""
        super().__init__(model_name, **kwargs)
        self.api_key = api_key
        self.client = None
        self.model = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the Google AI client."""
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)
        except ImportError:
            raise ImportError("Google AI package not installed. Install with: pip install google-generativeai")
    
    def _convert_openai_tools_to_gemini(self, openai_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert OpenAI tool format to Gemini tool format."""
        gemini_tools = []
        for tool in openai_tools:
            if tool.get("type") == "function" and "function" in tool:
                func = tool["function"]
                gemini_tool = {
                    "function_declarations": [{
                        "name": func["name"],
                        "description": func["description"],
                        "parameters": func.get("parameters", {})
                    }]
                }
                gemini_tools.append(gemini_tool)
        return gemini_tools

    async def generate_stream(self, messages: List[Dict[str, Any]], **kwargs) -> AsyncGenerator[Event, None]:
        """Generate streaming response from Gemini."""
        if not self.validate_messages(messages):
            yield Event(type=EventType.RUN_ERROR, error="Invalid message format")
            return
            
        # Emit run started event
        yield Event(type=EventType.RUN_STARTED)
        
        try:
            # Format messages for Gemini
            formatted_messages = self.format_messages(messages)
            
            # Gemini uses a different conversation format
            history = []
            current_message = None
            
            for msg in formatted_messages[:-1]:  # All but the last message go to history
                if msg["role"] == "system":
                    continue
                    
                role = "user" if msg["role"] in ["user", "tool"] else "model"
                history.append({
                    "role": role,
                    "parts": [msg["content"]]
                })
            
            # Last message is the current prompt
            if formatted_messages:
                current_message = formatted_messages[-1]["content"]
            
            # Set up generation parameters
            generation_config = {}
            if "temperature" in kwargs:
                generation_config["temperature"] = kwargs["temperature"]
            if "max_tokens" in kwargs:
                generation_config["max_output_tokens"] = kwargs["max_tokens"]
            
            # Add tools if available (convert from OpenAI format)
            tools = None
            if self.supports_tools() and "tools" in kwargs and kwargs["tools"]:
                tools = self._convert_openai_tools_to_gemini(kwargs["tools"])
            
            # Start chat with history
            chat = self.model.start_chat(history=history)
            
            # Generate streaming response
            if tools:
                response = chat.send_message(
                    current_message,
                    generation_config=generation_config,
                    tools=tools,
                    stream=True
                )
            else:
                response = chat.send_message(
                    current_message,
                    generation_config=generation_config,
                    stream=True
                )
            
            yield Event(type=EventType.TEXT_MESSAGE_START)
            
            current_content = ""
            for chunk in response:
                if chunk.text:
                    current_content += chunk.text
                    yield Event(
                        type=EventType.TEXT_MESSAGE_CONTENT,
                        content=chunk.text
                    )
                    
                # Handle function calls in Gemini format
                if hasattr(chunk, 'function_calls') and chunk.function_calls:
                    for func_call in chunk.function_calls:
                        tool_call_id = str(uuid.uuid4())
                        yield Event(
                            type=EventType.TOOL_CALL_START,
                            tool_call_id=tool_call_id,
                            tool_name=func_call.name
                        )
                        yield Event(
                            type=EventType.TOOL_CALL_ARGS,
                            tool_call_id=tool_call_id,
                            tool_args=dict(func_call.args)
                        )
                        yield Event(
                            type=EventType.TOOL_CALL_END,
                            tool_call_id=tool_call_id
                        )
            
            if current_content:
                yield Event(type=EventType.TEXT_MESSAGE_END, content=current_content)
            yield Event(type=EventType.RUN_FINISHED)
            
        except Exception as e:
            yield Event(type=EventType.RUN_ERROR, error=str(e))
    
    def send_tool_result(self, tool_call_id: str, result: Any) -> None:
        """Send tool result back to Gemini."""
        pass
    
    def supports_tools(self) -> bool:
        """Gemini supports function calling on supported models."""
        return self.model_name in self.SUPPORTED_MODELS
    
    def get_supported_models(self) -> List[str]:
        """Get supported Gemini models."""
        return self.SUPPORTED_MODELS.copy()
    
    def format_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format messages for Gemini API."""
        formatted = []
        
        for message in messages:
            # Gemini handles roles differently
            role = message["role"]
            if role == "assistant":
                role = "model"
            elif role == "tool":
                role = "user"  # Tool results are sent as user messages
            
            formatted_msg = {
                "role": role,
                "content": message["content"]
            }
            
            formatted.append(formatted_msg)
        
        return formatted 