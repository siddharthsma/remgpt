"""
Conversation orchestrator for managing LLM interactions and context updates.
Updated to use the unified Event system, with OrchestratorStatus removed.
"""

import asyncio
import json
import logging
from typing import AsyncGenerator, Dict, Any, Optional, Callable, Awaitable, List

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
        logger: Optional[logging.Logger] = None,
        # Remote tool configuration
        mcp_servers: Optional[List[str]] = None,
        a2a_agents: Optional[List[str]] = None
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
            mcp_servers: List of MCP server URLs or paths
            a2a_agents: List of A2A agent base URLs
        """
        self.context_manager = context_manager
        self.llm_client = llm_client
        self.tool_executor = tool_executor or ToolExecutor()
        self.logger = logger or logging.getLogger(__name__)
        
        # Automatically sync context manager with LLM client for accurate token counting
        if self.llm_client:
            self.context_manager.sync_with_llm_client(self.llm_client)
        
        # Remote tool configuration (optional)
        self.mcp_servers = mcp_servers or []
        self.a2a_agents = a2a_agents or []
        self.remote_tool_manager = None
        
        # Topic drift detection (always enabled)
        self.vector_db = vector_database or InMemoryVectorDatabase(logger=self.logger)
        self._initialize_topic_components(drift_detection_config or {})
        
        # Statistics
        self.topics_created = 0
        self.drift_detections = 0
        self.messages_processed = 0
        
        # Register context management functions for the LLM to use
        self._register_context_management_functions()
    
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
    
    async def initialize_remote_tools(self):
        """Initialize remote MCP and A2A tools."""
        if not self.mcp_servers and not self.a2a_agents:
            return
        
        try:
            from ..tools.remote import RemoteToolManager
            self.remote_tool_manager = RemoteToolManager()
            
            # Initialize MCP servers
            if self.mcp_servers:
                self.logger.info(f"Connecting to {len(self.mcp_servers)} MCP servers...")
                mcp_tools = await self.remote_tool_manager.add_mcp_servers(self.mcp_servers)
                
                # Register MCP tools with the tool executor
                for tool in mcp_tools:
                    self.tool_executor.register_tool(tool)
                
                self.logger.info(f"Registered {len(mcp_tools)} MCP tools")
            
            # Initialize A2A agents
            if self.a2a_agents:
                self.logger.info(f"Connecting to {len(self.a2a_agents)} A2A agents...")
                a2a_tools = await self.remote_tool_manager.add_a2a_agents(self.a2a_agents)
                
                # Register A2A tools with the tool executor
                for tool in a2a_tools:
                    self.tool_executor.register_tool(tool)
                
                self.logger.info(f"Registered {len(a2a_tools)} A2A tools")
                
        except ImportError as e:
            self.logger.warning(f"Remote tools not available: {e}")
        except Exception as e:
            self.logger.error(f"Failed to initialize remote tools: {e}")
    
    async def cleanup_remote_tools(self):
        """Clean up remote tool connections."""
        if self.remote_tool_manager:
            await self.remote_tool_manager.cleanup()
            self.remote_tool_manager = None
    
    def register_tool_handler(self, tool_name: str, handler: Callable):
        """
        Register a handler for a specific tool/function.
        Note: This is deprecated, use tool_executor.register_tool() instead.
        """
        # Create a simple tool wrapper for backward compatibility
        from ..tools import BaseTool
        
        class LegacyTool(BaseTool):
            def __init__(self, name, handler):
                super().__init__(name, f"Legacy tool: {name}")
                self.handler = handler
                
            async def execute(self, **kwargs):
                if asyncio.iscoroutinefunction(self.handler):
                    return await self.handler(**kwargs)
                else:
                    return self.handler(**kwargs)
                    
            def get_schema(self):
                return {
                    "type": "function",
                    "function": {
                        "name": self.name,
                        "description": self.description,
                        "parameters": {"type": "object", "properties": {}}
                    }
                }
        
        legacy_tool = LegacyTool(tool_name, handler)
        self.tool_executor.register_tool(legacy_tool)
        self.logger.info(f"Registered legacy tool handler: {tool_name}")
    
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
            
            # Step 2: Check for topic drift and handle topic recall
            drift_detected = False
            topic_recalled = False
            if message.role.value == "user":
                drift_detected = await self._detect_topic_drift(message)
                
                if drift_detected:
                    # First, try to recall a similar topic for the new message
                    try:
                        recalled_topic_id = await self.context_manager.recall_similar_topic(message.content)
                        if recalled_topic_id:
                            topic_recalled = True
                            self.logger.info(f"Automatically recalled similar topic: {recalled_topic_id}")
                    except Exception as e:
                        self.logger.error(f"Error during automatic topic recall: {e}")
                    
                    # Enhance user message with appropriate instructions
                    original_content = message.content
                    if topic_recalled:
                        enhanced_content = f"{original_content}\n\n[SYSTEM INSTRUCTION: Topic drift detected and similar topic recalled. Before responding, you must call save_current_topic to save the previous conversation, then continue with the recalled topic context.]"
                    else:
                        enhanced_content = f"{original_content}\n\n[SYSTEM INSTRUCTION: Topic drift detected. Before responding to this new topic, you must call save_current_topic to save the previous conversation topic. You may also call recall_similar_topic to check if this topic has been discussed before.]"
                    
                    # Update the message content
                    message.content = enhanced_content
                    self.logger.info("Enhanced user message with topic drift and recall instructions")
            
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
            # Extract text content from message
            if hasattr(message, 'content'):
                if isinstance(message.content, str):
                    text_content = message.content
                elif isinstance(message.content, list):
                    # Handle rich content - extract text parts
                    text_parts = []
                    for item in message.content:
                        if isinstance(item, dict) and item.get('type') == 'text':
                            text_parts.append(item.get('text', ''))
                        elif isinstance(item, str):
                            text_parts.append(item)
                    text_content = ' '.join(text_parts)
                else:
                    text_content = str(message.content)
            else:
                text_content = str(message)
            
            self.logger.debug(f"Analyzing message for topic drift: {text_content[:50]}...")
            
            # Create a simple message-like object for the drift detector
            class SimpleMessage:
                def __init__(self, content):
                    self.content = content
            
            simple_message = SimpleMessage(text_content)
            
            # Use detect_drift which handles both embedding creation and drift detection
            drift_detected, embedding_result, similarity = self.drift_detector.detect_drift(simple_message)
            
            if drift_detected:
                self.logger.info(f"Topic drift detected: similarity={similarity:.3f}")
                self.drift_detections += 1
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error in topic drift detection: {e}")
            return False
    
    async def _call_llm_with_functions(self, messages: list, function_calling_enabled: bool, max_turns: int = 5) -> AsyncGenerator[Event, None]:
        """
        Call the LLM with messages and handle streaming response with tool execution.
        Supports multi-turn conversations with recursive tool calling.
        
        Args:
            messages: Messages to send to LLM
            function_calling_enabled: Whether function calling is enabled
            max_turns: Maximum number of conversation turns to prevent infinite loops
        """
        import time
        
        yield Event(
            type=EventType.RUN_STARTED,
            data={
                "message_count": len(messages),
                "function_calling": function_calling_enabled,
                "client_type": "real",
                "max_turns": max_turns
            },
            timestamp=time.time()
        )
        
        # Recursive function to handle multi-turn conversations
        async for event in self._execute_conversation_turn(messages, function_calling_enabled, max_turns):
            yield event
    
    async def _execute_conversation_turn(self, messages: list, function_calling_enabled: bool, turns_remaining: int) -> AsyncGenerator[Event, None]:
        """Execute a single conversation turn with tool call handling."""
        import time
        
        if turns_remaining <= 0:
            self.logger.warning("Maximum conversation turns reached, ending conversation")
            yield Event(
                type=EventType.RUN_ERROR,
                error="Maximum conversation turns reached",
                data={"turns_remaining": turns_remaining},
                timestamp=time.time()
            )
            return
        
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
            has_tool_calls = False
            
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
                    has_tool_calls = True
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
                            tool_message = {
                                "role": "tool",
                                "content": json.dumps(result),
                                "tool_call_id": event.tool_call_id
                            }
                            formatted_messages.append(tool_message)
                            
                            # Log successful tool execution
                            self.logger.info(f"Successfully executed tool {tool_call['name']}: {result}")
                            
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
                            error_message = {
                                "role": "tool",
                                "content": json.dumps({"error": str(tool_error)}),
                                "tool_call_id": event.tool_call_id
                            }
                            formatted_messages.append(error_message)
                            
                            # Log tool execution error
                            self.logger.error(f"Tool execution failed for {tool_call['name']}: {tool_error}")
                        
                        # Remove from pending
                        del pending_tool_calls[event.tool_call_id]
                
                elif event.type == EventType.RUN_FINISHED:
                    # Check if we need to continue the conversation
                    has_tool_results = any(msg.get("role") == "tool" for msg in formatted_messages[-10:])
                    should_continue = has_tool_calls or has_tool_results
                    
                    if should_continue and turns_remaining > 1:
                        # Add assistant message to conversation if we have content
                        if assistant_content.strip():
                            assistant_message = {
                                "role": "assistant",
                                "content": assistant_content.strip()
                            }
                            formatted_messages.append(assistant_message)
                            self.logger.info(f"Added assistant response to conversation: {assistant_content[:50]}...")
                        
                        # Check for post-tool warnings (topic drift and token limits)
                        await self._check_post_tool_warnings(formatted_messages)
                        
                        # Continue conversation with updated context
                        self.logger.info(f"Continuing conversation (turns remaining: {turns_remaining - 1})")
                        async for follow_up_event in self._execute_conversation_turn(
                            formatted_messages, 
                            function_calling_enabled, 
                            turns_remaining - 1
                        ):
                            yield follow_up_event
                        return  # Don't process further, recursive call handles it
                    
                    # Final turn - add assistant message to context and end
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
                                "tool_calls_executed": len(pending_tool_calls) == 0,
                                "conversation_turns": 5 - turns_remaining + 1,
                                "final_turn": True
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
    
    async def _check_post_tool_warnings(self, formatted_messages: list) -> None:
        """
        Check for topic drift and token limit warnings after tool execution.
        Adds warning messages to the conversation if needed.
        """
        import time
        from ..types import AssistantMessage
        
        try:
            # Check for topic drift on recent assistant messages
            recent_assistant_messages = [
                msg for msg in formatted_messages[-5:] 
                if msg.get("role") == "assistant" and msg.get("content", "").strip()
            ]
            
            if recent_assistant_messages:
                # Create a temporary message for drift detection
                last_assistant_content = recent_assistant_messages[-1]["content"]
                temp_message = AssistantMessage(content=last_assistant_content)
                
                # Check for topic drift
                drift_detected = await self._detect_topic_drift(temp_message)
                if drift_detected:
                    drift_warning = {
                        "role": "assistant", 
                        "content": "TOPIC DRIFT DETECTED: The conversation topic has changed significantly after the tool execution. I should consider saving the current context."
                    }
                    formatted_messages.append(drift_warning)
                    self.logger.info("Added post-tool topic drift warning")
            
            # Check token usage (this needs to be estimated from formatted_messages)
            estimated_tokens = self._estimate_token_count(formatted_messages)
            max_tokens = getattr(self.context_manager.context, 'max_tokens', 4000)
            if estimated_tokens > max_tokens * 0.7:
                token_warning = {
                    "role": "assistant",
                    "content": f"TOKEN LIMIT WARNING: Context is getting full after tool execution ({estimated_tokens} tokens). I should consider evicting old topics to make room."
                }
                formatted_messages.append(token_warning)
                self.logger.info(f"Added post-tool token limit warning: {estimated_tokens} tokens")
                
        except Exception as e:
            self.logger.error(f"Error checking post-tool warnings: {e}")
    
    def _estimate_token_count(self, formatted_messages: list) -> int:
        """Estimate token count for formatted messages."""
        total_chars = sum(len(str(msg.get("content", ""))) for msg in formatted_messages)
        # Rough estimation: ~4 characters per token
        return total_chars // 4
    
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
        
        class UpdateTopicTool(BaseTool):
            def __init__(self, context_manager):
                super().__init__("update_topic", "Update an existing topic with additional information")
                self.context_manager = context_manager
            
            async def execute(self, topic_id: str, additional_summary: str, additional_key_facts: list = None) -> dict:
                updated_id = self.context_manager.update_topic(topic_id, additional_summary, additional_key_facts)
                return {"topic_id": updated_id, "status": "updated"}
            
            def get_schema(self) -> dict:
                return {
                    "type": "function",
                    "function": {
                        "name": "update_topic",
                        "description": "Update an existing topic with additional information instead of creating a new topic",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "topic_id": {"type": "string", "description": "ID of the existing topic to update"},
                                "additional_summary": {"type": "string", "description": "Additional summary to merge"},
                                "additional_key_facts": {"type": "array", "items": {"type": "string"}, "description": "Additional key facts"}
                            },
                            "required": ["topic_id", "additional_summary"]
                        }
                    }
                }
        
        class RecallTopicTool(BaseTool):
            def __init__(self, context_manager):
                super().__init__("recall_similar_topic", "Recall a similar topic from memory")
                self.context_manager = context_manager
            
            async def execute(self, **kwargs) -> dict:
                user_message = kwargs.get("user_message", "")
                recalled_id = await self.context_manager.recall_similar_topic(user_message)
                return {"topic_id": recalled_id, "status": "recalled" if recalled_id else "none_found"}
            
            def get_schema(self) -> dict:
                return {
                    "type": "function",
                    "function": {
                        "name": "recall_similar_topic",
                        "description": "Search for and recall a similar topic from long-term memory based on user message content",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "user_message": {"type": "string", "description": "The user message to find similar topics for"}
                            },
                            "required": ["user_message"]
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
        update_tool = UpdateTopicTool(self.context_manager)
        recall_tool = RecallTopicTool(self.context_manager)
        evict_tool = EvictTopicTool(self.context_manager)
        
        self.tool_executor.register_tool(save_tool)
        self.tool_executor.register_tool(update_tool)
        self.tool_executor.register_tool(recall_tool)
        self.tool_executor.register_tool(evict_tool)
        
        self.logger.info("Registered enhanced context management tools with tool executor")
    
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
            "status": "active",  # Fixed: Add status field required by API
            "context_summary": self.context_manager.get_context_summary(),
            "registered_tools": list(self.tool_executor.get_registered_tools()),  # Fixed: Use tool_executor
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