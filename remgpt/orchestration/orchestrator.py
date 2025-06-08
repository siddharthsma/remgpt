"""
Conversation orchestrator for managing LLM interactions and context updates.
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
from .status import OrchestratorStatus
from .events import StreamEvent


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
        llm_client: Optional[Callable] = None,
        vector_database: Optional[VectorDatabase] = None,
        drift_detection_config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the orchestrator.
        
        Args:
            context_manager: The context manager instance
            llm_client: LLM client function (async callable)
            vector_database: Vector database for storing topics
            drift_detection_config: Configuration for drift detection
            logger: Optional logger instance
        """
        self.context_manager = context_manager
        self.llm_client = llm_client
        self.logger = logger or logging.getLogger(__name__)
        self.status = OrchestratorStatus.IDLE
        
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
    
    async def process_message(self, message: Message) -> AsyncGenerator[StreamEvent, None]:
        """
        Process an incoming message and yield streaming events.
        
        New Algorithm:
        1. Add user message to FIFO queue
        2. Check for topic drift - if detected, add warning message to queue
        3. Check token usage - if near limit, add warning message to queue  
        4. Construct system message from all blocks except FIFO queue and working context
        5. Call LLM with function calling enabled for context management
        6. Handle any function calls from LLM
        7. Stream back LLM events only
        
        Args:
            message: The incoming message to process
            
        Yields:
            StreamEvent: Events from LLM processing only
        """
        import time
        
        self.status = OrchestratorStatus.PROCESSING
        start_time = time.time()
        self.messages_processed += 1
        
        try:
            # Step 1: Add user message to FIFO queue FIRST
            self.context_manager.add_message_to_queue(message)
            self.logger.info(f"Added {message.role.value} message to FIFO queue")
            
            # Step 2: Check for topic drift (only for user messages)
            drift_detected = False
            if message.role.value == "user":
                drift_detected = await self._detect_topic_drift(message)
                
                # If drift detected, add warning message to FIFO queue
                if drift_detected:
                    drift_warning = AssistantMessage(
                        content="TOPIC DRIFT DETECTED: The conversation has shifted to a new topic. I should save the current conversation topic before continuing."
                    )
                    self.context_manager.add_message_to_queue(drift_warning)
                    self.logger.info("Added topic drift warning to FIFO queue")
            
            # Step 3: Check token usage and add warning if near limit
            token_warning_added = False
            if self.context_manager.is_near_token_limit(threshold=0.7):
                token_warning = AssistantMessage(
                    content="APPROACHING TOKEN LIMIT: My context is getting full (70%+ of limit). I should consider evicting an old topic to make room for new content."
                )
                self.context_manager.add_message_to_queue(token_warning)
                token_warning_added = True
                self.logger.info("Added token limit warning to FIFO queue")
            
            # Step 4: Get messages for LLM (includes system messages from blocks)
            # The context manager will construct system messages from:
            # - SystemInstructionsBlock
            # - MemoryInstructionsBlock  
            # - ToolsDefinitionsBlock
            # Plus conversation messages from FIFO queue and working context topics
            llm_messages = self.context_manager.get_messages_for_llm()
            
            # Log context summary for debugging
            context_summary = self.context_manager.get_context_summary()
            self.logger.info(
                f"Context: {context_summary['total_tokens']} tokens, "
                f"{context_summary['topics_count']} topics, "
                f"queue: {context_summary['queue_size']}, "
                f"drift: {drift_detected}, "
                f"token_warning: {token_warning_added}"
            )
            
            # Step 5: Call LLM with function calling enabled - ONLY YIELD LLM EVENTS
            if self.llm_client:
                async for event in self._call_llm_with_functions(llm_messages, drift_detected or token_warning_added):
                    yield event
            else:
                # Mock LLM with function calling capability - ONLY YIELD LLM EVENTS
                async for event in self._mock_llm_with_functions(llm_messages, drift_detected, token_warning_added):
                    yield event
            
        except Exception as e:
            self.status = OrchestratorStatus.ERROR
            self.logger.error(f"Error in process_message: {e}")
            # Only yield error events that would come from LLM
            yield StreamEvent(
                type="llm_error",
                data={"error": str(e), "error_type": type(e).__name__},
                timestamp=time.time()
            )
        finally:
            self.status = OrchestratorStatus.IDLE
    
    async def _detect_topic_drift(self, message: Message) -> bool:
        """
        Detect topic drift for a message.
        
        Args:
            message: The message to analyze for topic drift
            
        Returns:
            bool: True if drift is detected, False otherwise
        """
        try:
            self.logger.debug(f"Analyzing message for topic drift: {str(message.content)[:50]}...")
            
            # Get message embedding using the correct API
            embedding_result = self.drift_detector.create_embedding(message)
            self.logger.debug(f"Created embedding of size {len(embedding_result.embedding)}")
            
            # Check for drift using the correct API
            drift_detected, embedding_result, similarity_score = self.drift_detector.detect_drift(message)
            
            if drift_detected:
                self.drift_detections += 1
                self.logger.info(f"Topic drift detected with similarity score: {similarity_score}")
            else:
                self.logger.debug(f"No topic drift detected (similarity: {similarity_score})")
            
            return drift_detected
            
        except Exception as e:
            self.logger.error(f"Error in topic drift detection: {e}")
            return False
    
    async def _call_llm_with_functions(self, messages: list, function_calling_enabled: bool) -> AsyncGenerator[StreamEvent, None]:
        """
        Call the LLM with messages and handle streaming response.
        
        Args:
            messages: List of messages for the LLM
            function_calling_enabled: Whether function calling is enabled
            
        Yields:
            StreamEvent: Events from LLM processing
        """
        import time
        
        # This is a placeholder for actual LLM integration with function calling
        # In practice, you would:
        # 1. Prepare function definitions for context management
        # 2. Call the LLM API with function calling enabled
        # 3. Handle function calls and responses
        # 4. Stream back the results
        
        yield StreamEvent(
            type="llm_call_start",
            data={
                "message_count": len(messages),
                "function_calling": function_calling_enabled,
                "client_type": "real"
            },
            timestamp=time.time()
        )
        
        try:
            # Actual LLM call would go here
            # For now, this is a placeholder
            yield StreamEvent(
                type="llm_response_chunk",
                data={"content": "Real LLM integration not implemented yet."},
                timestamp=time.time()
            )
            
        except Exception as e:
            yield StreamEvent(
                type="llm_error",
                data={"error": str(e)},
                timestamp=time.time()
            )
    
    async def _mock_llm_with_functions(self, messages: list, drift_detected: bool, token_warning_added: bool) -> AsyncGenerator[StreamEvent, None]:
        """
        Mock LLM with function calling capability for testing.
        
        Args:
            messages: List of messages for the LLM
            drift_detected: Whether topic drift was detected
            token_warning_added: Whether token limit warning was added
            
        Yields:
            StreamEvent: Mock LLM events
        """
        import time
        
        yield StreamEvent(
            type="llm_call_start",
            data={
                "message_count": len(messages),
                "function_calling": True,
                "client_type": "mock"
            },
            timestamp=time.time()
        )
        
        # Register context management functions
        self._register_context_management_functions()
        
        # Mock LLM behavior based on context warnings
        function_calls_made = []
        
        # Check for topic drift warning in messages
        has_drift_warning = any(
            hasattr(msg, 'content') and 
            "TOPIC DRIFT DETECTED" in str(msg.content) 
            for msg in messages
        )
        
        # Check for token limit warning in messages  
        has_token_warning = any(
            hasattr(msg, 'content') and 
            "APPROACHING TOKEN LIMIT" in str(msg.content)
            for msg in messages
        )
        
        if has_drift_warning:
            # Mock LLM decides to save current topic
            yield StreamEvent(
                type="llm_function_call",
                data={
                    "function_name": "save_current_topic",
                    "arguments": {
                        "topic_summary": "Previous conversation topic that needs to be saved",
                        "topic_key_facts": ["Important fact 1", "Important fact 2"]
                    }
                },
                timestamp=time.time()
            )
            
            # Execute the function call
            topic_id = self.context_manager.save_current_topic(
                "Previous conversation topic that needs to be saved",
                ["Important fact 1", "Important fact 2"]
            )
            
            function_calls_made.append({
                "function": "save_current_topic", 
                "result": topic_id
            })
            
            yield StreamEvent(
                type="llm_function_result",
                data={
                    "function_name": "save_current_topic",
                    "result": topic_id,
                    "success": True
                },
                timestamp=time.time()
            )
        
        if has_token_warning:
            # Mock LLM decides to evict oldest topic
            yield StreamEvent(
                type="llm_function_call",
                data={
                    "function_name": "evict_oldest_topic",
                    "arguments": {}
                },
                timestamp=time.time()
            )
            
            # Execute the function call
            evicted_topic_id = self.context_manager.evict_oldest_topic()
            
            function_calls_made.append({
                "function": "evict_oldest_topic",
                "result": evicted_topic_id
            })
            
            yield StreamEvent(
                type="llm_function_result",
                data={
                    "function_name": "evict_oldest_topic", 
                    "result": evicted_topic_id,
                    "success": True
                },
                timestamp=time.time()
            )
        
        # Mock LLM response after handling context management
        if function_calls_made:
            response_content = f"I've handled the context management tasks: {', '.join([fc['function'] for fc in function_calls_made])}. How can I help you now?"
        else:
            response_content = "I understand your message. How can I assist you today?"
        
        yield StreamEvent(
            type="llm_response_start",
            data={},
            timestamp=time.time()
        )
        
        yield StreamEvent(
            type="llm_response_chunk",
            data={"content": response_content},
            timestamp=time.time()
        )
        
        # Add assistant response to context
        assistant_message = AssistantMessage(content=response_content)
        self.context_manager.add_message_to_queue(assistant_message)
        
        yield StreamEvent(
            type="llm_response_complete",
            data={
                "message": assistant_message.dict(),
                "function_calls_made": function_calls_made
            },
            timestamp=time.time()
        )
    
    def _register_context_management_functions(self):
        """Register context management functions that the LLM can call."""
        
        # Register save_current_topic function
        async def save_topic_handler(topic_summary: str, topic_key_facts: list = None):
            """Handler for save_current_topic function calls."""
            return self.context_manager.save_current_topic(topic_summary, topic_key_facts)
        
        # Register evict_oldest_topic function
        async def evict_topic_handler():
            """Handler for evict_oldest_topic function calls."""
            return self.context_manager.evict_oldest_topic()
        
        self.register_tool_handler("save_current_topic", save_topic_handler)
        self.register_tool_handler("evict_oldest_topic", evict_topic_handler)
        
        self.logger.info("Registered context management functions for LLM")
    
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
            "status": self.status.value,
            "context_summary": self.context_manager.get_context_summary(),
            "registered_tools": list(self.tool_handlers.keys()),
            "topic_drift": self._get_drift_statistics()
        }
        
        # Add working context statistics
        if hasattr(self.context_manager.context, 'working_context'):
            status_data["working_context"] = self.context_manager.context.working_context.get_statistics()
        
        return status_data 