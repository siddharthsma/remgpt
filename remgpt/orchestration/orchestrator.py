"""
Conversation orchestrator for managing LLM interactions and context updates.
Updated to use the unified Event system, with OrchestratorStatus removed.
"""

import asyncio
import json
import logging
from typing import AsyncGenerator, Dict, Any, Optional, Callable, Awaitable

from ..context import LLMContextManager
from ..types import Message, UserMessage, AssistantMessage, ToolMessage
from ..detection import TopicDriftDetector, EmbeddingResult
from ..summarization import TopicSummarizer, Topic
from ..storage import VectorDatabase, InMemoryVectorDatabase
from ..llm import BaseLLMClient, Event, EventType
from ..tools import ToolExecutor


class ConversationOrchestrator:
    """
    Orchestrates conversation flow between messages, context management, and LLM.
    
    This class:
    - Receives incoming messages
    - Detects topic drift using embeddings and Page-Hinkley test
    - Summarizes messages when drift is detected
    - Stores topic summaries in vector database
    - Updates context via ContextManager
    - Calls LLM for responses
    - Handles tool/function calls
    - Manages token-based eviction
    """
    
    def __init__(
        self,
        context_manager: LLMContextManager,
        llm_client: Optional[BaseLLMClient] = None,
        tool_executor: Optional[ToolExecutor] = None,
        vector_database: Optional[VectorDatabase] = None,
        drift_detection_config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the orchestrator.
        
        Args:
            context_manager: The context manager instance
            llm_client: LLM client instance (BaseLLMClient)
            tool_executor: Tool executor for handling tool calls
            vector_database: Vector database for storing topics
            drift_detection_config: Configuration for drift detection
            logger: Optional logger instance
        """
        self.context_manager = context_manager
        self.llm_client = llm_client
        self.tool_executor = tool_executor or ToolExecutor()
        self.logger = logger or logging.getLogger(__name__)
        
        # Tool/function registry
        self.tool_handlers: Dict[str, Callable] = {}
        
        # Topic drift detection (always enabled)
        self.vector_db = vector_database or InMemoryVectorDatabase(logger=self.logger)
        self._initialize_topic_components(drift_detection_config or {})
        
        # Statistics
        self.topics_created = 0
        self.drift_detections = 0
        self.messages_processed = 0
    
    def _initialize_topic_components(self, config: Dict[str, Any]):
        """Initialize topic drift detection components."""
        self.drift_detector = TopicDriftDetector(
            model_name=config.get("model_name", "all-MiniLM-L6-v2"),
            similarity_threshold=config.get("similarity_threshold", 0.7),
            drift_threshold=config.get("drift_threshold", 0.5),
            alpha=config.get("alpha", 0.05),
            window_size=config.get("window_size", 10),
            logger=self.logger
        )
        
        self.topic_summarizer = TopicSummarizer(
            llm_client=self.llm_client,
            max_summary_length=config.get("max_summary_length", 200),
            max_key_facts=config.get("max_key_facts", 5),
            logger=self.logger
        )
        
        self.logger.info("Topic drift detection components initialized")
    
    def register_tool_handler(self, tool_name: str, handler: Callable):
        """Register a handler for a specific tool/function."""
        self.tool_handlers[tool_name] = handler
        self.logger.info(f"Registered tool handler: {tool_name}")
    
    async def process_message(self, message: Message) -> AsyncGenerator[Event, None]:
        """
        Process an incoming message and yield streaming events.
        
        Args:
            message: The incoming message to process
            
        Yields:
            Event: Events from LLM processing and orchestration
        """
        import time
        
        start_time = time.time()
        self.messages_processed += 1
        
        try:
            # Step 1: Add user message to FIFO queue
            self.context_manager.add_message_to_queue(message)
            self.logger.info(f"Added {message.role.value} message to FIFO queue")
            
            # Step 2: Check for topic drift
            drift_detected = False
            if message.role.value == "user":
                drift_detected = await self._detect_topic_drift(message)
                
                if drift_detected:
                    drift_warning = AssistantMessage(
                        content="TOPIC DRIFT DETECTED: The conversation has shifted to a new topic. I should save the current conversation topic before continuing."
                    )
                    self.context_manager.add_message_to_queue(drift_warning)
                    self.logger.info("Added topic drift warning to FIFO queue")
            
            # Step 3: Check token usage
            token_warning_added = False
            if self.context_manager.is_near_token_limit(threshold=0.7):
                token_warning = AssistantMessage(
                    content="APPROACHING TOKEN LIMIT: My context is getting full (70%+ of limit). I should consider evicting an old topic to make room for new content."
                )
                self.context_manager.add_message_to_queue(token_warning)
                token_warning_added = True
                self.logger.info("Added token limit warning to FIFO queue")
            
            # Step 4: Get messages for LLM
            llm_messages = self.context_manager.get_messages_for_llm()
            
            # Log context summary
            context_summary = self.context_manager.get_context_summary()
            self.logger.info(
                f"Context: {context_summary['total_tokens']} tokens, "
                f"{context_summary['topics_count']} topics, "
                f"queue: {context_summary['queue_size']}, "
                f"drift: {drift_detected}, "
                f"token_warning: {token_warning_added}"
            )
            
            # Step 5: Call LLM
            if self.llm_client:
                async for event in self._call_llm_with_functions(llm_messages, drift_detected or token_warning_added):
                    yield event
            else:
                # No LLM client provided - cannot process without one
                yield Event(
                    type=EventType.RUN_ERROR,
                    error="No LLM client provided to orchestrator",
                    data={"error_type": "MissingLLMClient"},
                    timestamp=time.time()
                )
            
        except Exception as e:
            self.logger.error(f"Error in process_message: {e}")
            yield Event(
                type=EventType.RUN_ERROR,
                error=str(e),
                data={"error_type": type(e).__name__},
                timestamp=time.time()
            )
    
    async def _detect_topic_drift(self, message: Message) -> bool:
        """Detect topic drift for a message."""
        try:
            self.logger.debug(f"Analyzing message for topic drift: {str(message.content)[:50]}...")
            
            embedding_result = self.drift_detector.create_embedding(message)
            
            # EmbeddingResult doesn't have success/error attributes - it just returns the result or raises exception
            drift_result = self.drift_detector.detect_drift(embedding_result.embedding)
            
            if drift_result.drift_detected:
                self.logger.info(f"Topic drift detected: {drift_result}")
                self.drift_detections += 1
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error in topic drift detection: {e}")
            return False
    
    async def _call_llm_with_functions(self, messages: list, function_calling_enabled: bool) -> AsyncGenerator[Event, None]:
        """Call the LLM with messages and handle streaming response with tool execution."""
        import time
        
        yield Event(
            type=EventType.RUN_STARTED,
            data={
                "message_count": len(messages),
                "function_calling": function_calling_enabled,
                "client_type": "real"
            },
            timestamp=time.time()
        )
        
        try:
            # Convert messages to proper format for LLM client
            formatted_messages = self._format_messages_for_llm(messages)
            
            # Prepare generation parameters
            generation_params = {}
            if function_calling_enabled and self.tool_executor.get_registered_tools():
                generation_params["tools"] = self.tool_executor.get_tool_schemas()
            
            # Track conversation state for potential tool calls
            pending_tool_calls = {}
            assistant_content = ""
            
            # Stream LLM response and handle tool calls
            async for event in self.llm_client.generate_stream(formatted_messages, **generation_params):
                
                if event.type == EventType.RUN_STARTED:
                    yield Event(
                        type=EventType.TEXT_MESSAGE_START,
                        timestamp=time.time()
                    )
                
                elif event.type == EventType.TEXT_MESSAGE_CONTENT:
                    assistant_content += event.content
                    yield event  # Pass through the original event
                
                elif event.type == EventType.TOOL_CALL_START:
                    pending_tool_calls[event.tool_call_id] = {
                        "name": event.tool_name,
                        "args": None
                    }
                    yield event  # Pass through the original event
                
                elif event.type == EventType.TOOL_CALL_ARGS:
                    if event.tool_call_id in pending_tool_calls:
                        pending_tool_calls[event.tool_call_id]["args"] = event.tool_args
                    yield event  # Pass through the original event
                
                elif event.type == EventType.TOOL_CALL_END:
                    # Execute the tool
                    if event.tool_call_id in pending_tool_calls:
                        tool_call = pending_tool_calls[event.tool_call_id]
                        try:
                            # Execute tool via tool executor
                            result = await self.tool_executor.execute_tool(
                                tool_call_id=event.tool_call_id,
                                tool_name=tool_call["name"],
                                tool_args=tool_call["args"] or {}
                            )
                            
                            yield Event(
                                type=EventType.CUSTOM,
                                data={
                                    "function_name": tool_call["name"],
                                    "result": result,
                                    "call_id": event.tool_call_id,
                                    "success": True,
                                    "event_subtype": "tool_result"
                                },
                                timestamp=time.time()
                            )
                            
                            # Add tool result to messages for continued conversation
                            formatted_messages.append({
                                "role": "tool",
                                "content": json.dumps(result),
                                "tool_call_id": event.tool_call_id
                            })
                            
                        except Exception as tool_error:
                            self.logger.error(f"Tool execution error: {tool_error}")
                            yield Event(
                                type=EventType.CUSTOM,
                                data={
                                    "function_name": tool_call["name"],
                                    "error": str(tool_error),
                                    "call_id": event.tool_call_id,
                                    "success": False,
                                    "event_subtype": "tool_result"
                                },
                                timestamp=time.time()
                            )
                            
                            # Add error result to messages
                            formatted_messages.append({
                                "role": "tool",
                                "content": json.dumps({"error": str(tool_error)}),
                                "tool_call_id": event.tool_call_id
                            })
                        
                        # Remove from pending
                        del pending_tool_calls[event.tool_call_id]
                
                elif event.type == EventType.RUN_FINISHED:
                    # Add final assistant message to context
                    if assistant_content.strip():
                        assistant_message = AssistantMessage(content=assistant_content.strip())
                        self.context_manager.add_message_to_queue(assistant_message)
                        
                        yield Event(
                            type=EventType.TEXT_MESSAGE_END,
                            content=assistant_message.content,
                            data={
                                "message": {
                                    "role": assistant_message.role.value,
                                    "content": assistant_message.content
                                },
                                "tool_calls_executed": len(pending_tool_calls) == 0
                            },
                            timestamp=time.time()
                        )
                
                elif event.type == EventType.RUN_ERROR:
                    yield event  # Pass through the original event
            
        except Exception as e:
            self.logger.error(f"Error in LLM call: {e}")
            yield Event(
                type=EventType.RUN_ERROR,
                error=str(e),
                timestamp=time.time()
            )
    
    def _register_context_management_functions(self):
        """Register context management functions that the LLM can call."""
        from ..tools import BaseTool
        
        class SaveTopicTool(BaseTool):
            def __init__(self, context_manager):
                super().__init__("save_current_topic", "Save the current conversation topic")
                self.context_manager = context_manager
            
            async def execute(self, topic_summary: str, topic_key_facts: list = None) -> dict:
                topic_id = self.context_manager.save_current_topic(topic_summary, topic_key_facts)
                return {"topic_id": topic_id, "summary": topic_summary}
            
            def get_schema(self) -> dict:
                return {
                    "type": "function",
                    "function": {
                        "name": "save_current_topic",
                        "description": "Save the current conversation topic",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "topic_summary": {"type": "string"},
                                "topic_key_facts": {"type": "array", "items": {"type": "string"}}
                            },
                            "required": ["topic_summary"]
                        }
                    }
                }
        
        class EvictTopicTool(BaseTool):
            def __init__(self, context_manager):
                super().__init__("evict_oldest_topic", "Evict the oldest topic from context")
                self.context_manager = context_manager
            
            async def execute(self) -> dict:
                topic_id = self.context_manager.evict_oldest_topic()
                return {"evicted_topic_id": topic_id}
            
            def get_schema(self) -> dict:
                return {
                    "type": "function", 
                    "function": {
                        "name": "evict_oldest_topic",
                        "description": "Evict the oldest topic from context",
                        "parameters": {
                            "type": "object",
                            "properties": {},
                            "required": []
                        }
                    }
                }
        
        # Register tools with the tool executor
        save_tool = SaveTopicTool(self.context_manager)
        evict_tool = EvictTopicTool(self.context_manager)
        
        self.tool_executor.register_tool(save_tool)
        self.tool_executor.register_tool(evict_tool)
        
        self.logger.info("Registered context management tools with tool executor")
    
    def _get_drift_statistics(self) -> Dict[str, Any]:
        """Get topic drift detection statistics."""
        stats = {
            "topics_created": self.topics_created,
            "drift_detections": self.drift_detections,
            "messages_processed": self.messages_processed
        }
        
        if self.drift_detector:
            stats.update(self.drift_detector.get_statistics())
        
        return stats
    
    def get_status(self) -> Dict[str, Any]:
        """Get current orchestrator status."""
        status_data = {
            "context_summary": self.context_manager.get_context_summary(),
            "registered_tools": list(self.tool_handlers.keys()),
            "topic_drift": self._get_drift_statistics()
        }
        
        if hasattr(self.context_manager.context, 'working_context'):
            status_data["working_context"] = self.context_manager.context.working_context.get_statistics()
        
        return status_data

    def _format_messages_for_llm(self, messages: list) -> list:
        """Format messages from context manager for LLM client."""
        formatted_messages = []
        
        for msg in messages:
            if hasattr(msg, 'role') and hasattr(msg, 'content'):
                # Message object from types
                formatted_msg = {
                    "role": msg.role.value if hasattr(msg.role, 'value') else str(msg.role),
                    "content": str(msg.content)
                }
                
                # Add tool call information if present
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    formatted_msg["tool_calls"] = msg.tool_calls
                
                if hasattr(msg, 'tool_call_id') and msg.tool_call_id:
                    formatted_msg["tool_call_id"] = msg.tool_call_id
                    
                formatted_messages.append(formatted_msg)
                
            elif isinstance(msg, dict):
                # Already formatted message
                formatted_messages.append(msg)
                
            else:
                # String or other content
                formatted_messages.append({
                    "role": "user",
                    "content": str(msg)
                })
        
        return formatted_messages 