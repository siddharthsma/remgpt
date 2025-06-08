"""
Mock LLM Client for testing purposes.
"""

import asyncio
import time
from typing import List, Dict, Any, AsyncGenerator
from ..base import BaseLLMClient
from ..events import Event, EventType


class MockLLMClient(BaseLLMClient):
    """Mock LLM client for testing and development."""
    
    def __init__(self, model_name: str = "mock-model", **kwargs):
        """
        Initialize the mock LLM client.
        
        Args:
            model_name: Mock model name
            **kwargs: Additional configuration options
        """
        super().__init__(model_name, **kwargs)
        self.response_delay = kwargs.get("response_delay", 0.1)
        self.simulate_streaming = kwargs.get("simulate_streaming", True)
        self.multi_turn_responses = []
        self.current_turn = 0
        
    async def generate_stream(self, messages: List[Dict[str, Any]], **kwargs) -> AsyncGenerator[Event, None]:
        """
        Generate mock streaming response with function calling capability.
        
        Args:
            messages: List of conversation messages
            **kwargs: Generation parameters (tools, etc.)
            
        Yields:
            Event: Mock events simulating LLM behavior
        """
        # Start the run
        yield Event(
            type=EventType.RUN_STARTED,
            data={
                "model": self.model_name,
                "message_count": len(messages),
                "has_tools": "tools" in kwargs
            },
            timestamp=time.time()
        )
        
        # Simulate processing delay
        if self.response_delay > 0:
            await asyncio.sleep(self.response_delay)
        
        # Check if we have configured multi-turn responses
        if self.multi_turn_responses and self.current_turn < len(self.multi_turn_responses):
            # Use configured response for this turn
            current_response = self.multi_turn_responses[self.current_turn]
            response_content = current_response.get("content", "")
            function_calls_needed = current_response.get("tool_calls", [])
            self.current_turn += 1
        else:
            # Fallback to original behavior
            function_calls_needed = self._analyze_messages_for_function_calls(messages)
            response_content = ""
        
        # Handle function calls if needed
        if function_calls_needed:
            for i, func_call in enumerate(function_calls_needed):
                # Handle both old format (dict with name/args) and new format (dict with id/name/args)
                if "id" in func_call:
                    call_id = func_call["id"]
                    tool_name = func_call["name"]
                    tool_args = func_call.get("args", {})
                else:
                    call_id = f"mock_call_{i+1}"
                    tool_name = func_call["name"]
                    tool_args = func_call.get("args", {})
                
                # Tool call start
                yield Event(
                    type=EventType.TOOL_CALL_START,
                    tool_call_id=call_id,
                    tool_name=tool_name,
                    timestamp=time.time()
                )
                
                # Tool call args
                yield Event(
                    type=EventType.TOOL_CALL_ARGS,
                    tool_call_id=call_id,
                    tool_args=tool_args,
                    timestamp=time.time()
                )
                
                # Tool call end
                yield Event(
                    type=EventType.TOOL_CALL_END,
                    tool_call_id=call_id,
                    timestamp=time.time()
                )
                
                # Simulate processing delay
                if self.response_delay > 0:
                    await asyncio.sleep(self.response_delay * 0.5)
        
        # Generate mock response content if not already set
        if not response_content:
            if function_calls_needed:
                function_names = [fc.get("name", "unknown") for fc in function_calls_needed]
                response_content = f"I've handled the context management tasks: {', '.join(function_names)}. How can I help you now?"
            else:
                response_content = "I understand your message. How can I assist you today?"
        
        # Stream the text response
        if self.simulate_streaming:
            words = response_content.split()
            current_content = ""
            
            for i, word in enumerate(words):
                if i == 0:
                    # First word - start text message
                    yield Event(
                        type=EventType.TEXT_MESSAGE_START,
                        timestamp=time.time()
                    )
                
                current_content += word
                if i < len(words) - 1:
                    current_content += " "
                
                # Stream content
                yield Event(
                    type=EventType.TEXT_MESSAGE_CONTENT,
                    content=word if i == 0 else " " + word,
                    timestamp=time.time()
                )
                
                # Simulate typing delay
                if self.response_delay > 0:
                    await asyncio.sleep(self.response_delay * 0.1)
        else:
            # Non-streaming response
            yield Event(
                type=EventType.TEXT_MESSAGE_START,
                timestamp=time.time()
            )
            
            yield Event(
                type=EventType.TEXT_MESSAGE_CONTENT,
                content=response_content,
                timestamp=time.time()
            )
        
        # Finish the run
        yield Event(
            type=EventType.RUN_FINISHED,
            data={
                "finish_reason": "stop",
                "content": response_content,
                "function_calls_made": len(function_calls_needed)
            },
            timestamp=time.time()
        )
    
    def _analyze_messages_for_function_calls(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Analyze messages to determine what function calls should be made.
        
        Args:
            messages: List of conversation messages
            
        Returns:
            List of function calls that should be made
        """
        function_calls = []
        
        # Check for topic drift warning
        has_drift_warning = any(
            "TOPIC DRIFT DETECTED" in str(msg.get("content", ""))
            for msg in messages
        )
        
        # Check for token limit warning
        has_token_warning = any(
            "APPROACHING TOKEN LIMIT" in str(msg.get("content", ""))
            for msg in messages
        )
        
        if has_drift_warning:
            function_calls.append({
                "name": "save_current_topic",
                "args": {
                    "topic_summary": "Previous conversation topic that needs to be saved",
                    "topic_key_facts": ["Important fact 1", "Important fact 2"]
                }
            })
        
        if has_token_warning:
            function_calls.append({
                "name": "evict_oldest_topic",
                "args": {}
            })
        
        return function_calls
    
    async def send_tool_result(self, tool_call_id: str, result: Any) -> None:
        """
        Mock implementation of sending tool result back to LLM.
        
        Args:
            tool_call_id: ID of the tool call
            result: Result of the tool execution
        """
        # Mock implementation - in a real client this would continue the conversation
        pass
    
    def supports_tools(self) -> bool:
        """
        Mock LLM supports tools.
        
        Returns:
            bool: Always True for mock client
        """
        return True
    
    def get_supported_models(self) -> List[str]:
        """
        Get list of mock models.
        
        Returns:
            List[str]: List of mock model names
        """
        return ["mock-model", "mock-gpt-4", "mock-claude-3"]
    
    def set_response_delay(self, delay: float):
        """
        Set response delay for controlling mock timing.
        
        Args:
            delay: Delay in seconds between events
        """
        self.response_delay = delay
    
    def set_streaming_mode(self, streaming: bool):
        """
        Enable or disable streaming simulation.
        
        Args:
            streaming: Whether to simulate streaming responses
        """
        self.simulate_streaming = streaming
    
    def configure_multi_turn_responses(self, responses: List[Dict[str, Any]]):
        """
        Configure multi-turn responses for testing.
        
        Args:
            responses: List of response dictionaries with 'content' and 'tool_calls'
        """
        self.multi_turn_responses = responses
        self.current_turn = 0 