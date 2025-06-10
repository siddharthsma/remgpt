"""
Context management tools for LLM conversation orchestration.

This module contains specialized tools that allow the LLM to manage its own context,
including saving topics, recalling similar discussions, updating existing topics,
and evicting old content when approaching token limits.
"""

import logging
from typing import Dict, Any, Optional, List
from ..tools.base import BaseTool


class SaveCurrentTopicTool(BaseTool):
    """Tool for saving the current conversation topic to long-term memory."""
    
    def __init__(self, context_manager, logger: Optional[logging.Logger] = None):
        super().__init__(
            name="save_current_topic",
            description="Save the current conversation topic to long-term memory with summary and key facts"
        )
        self.context_manager = context_manager
        self.logger = logger or logging.getLogger(__name__)
    
    async def execute(self, topic_summary: str, topic_key_facts: List[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Save the current conversation as a topic.
        
        Args:
            topic_summary: Brief summary of the conversation topic
            topic_key_facts: List of key facts or important points
            **kwargs: Additional arguments (ignored for flexibility)
            
        Returns:
            Dictionary with topic creation confirmation
        """
        try:
            # Ignore any extra parameters that might be passed by the LLM
            topic_id = self.context_manager.save_current_topic(topic_summary, topic_key_facts)
            self.logger.info(f"Successfully saved topic: {topic_id}")
            return {
                "topic_id": topic_id,
                "summary": topic_summary,
                "key_facts": topic_key_facts or [],
                "status": "saved"
            }
        except Exception as e:
            self.logger.error(f"Failed to save topic: {e}")
            return {
                "error": str(e),
                "status": "failed"
            }
    
    def get_schema(self) -> Dict[str, Any]:
        """Get the JSON schema for this tool."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": "Save the current conversation topic to long-term memory. Use this when switching to a new topic or when the conversation context is getting full.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "topic_summary": {
                            "type": "string",
                            "description": "A concise summary of the current conversation topic (e.g., 'Discussion about Python programming basics')"
                        },
                        "topic_key_facts": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of key facts, important points, or conclusions from the conversation"
                        }
                    },
                    "required": ["topic_summary"]
                }
            }
        }


class UpdateTopicTool(BaseTool):
    """Tool for updating an existing topic with additional information."""
    
    def __init__(self, context_manager, logger: Optional[logging.Logger] = None):
        super().__init__(
            name="update_topic",
            description="Update an existing topic with additional information instead of creating a new topic"
        )
        self.context_manager = context_manager
        self.logger = logger or logging.getLogger(__name__)
    
    async def execute(self, topic_id: str, additional_summary: str, additional_key_facts: List[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Update an existing topic with additional information.
        
        Args:
            topic_id: ID of the existing topic to update
            additional_summary: Additional summary text to merge
            additional_key_facts: Additional key facts to add
            **kwargs: Additional arguments (ignored for flexibility)
            
        Returns:
            Dictionary with update confirmation
        """
        try:
            # Ignore any extra parameters that might be passed by the LLM
            updated_id = self.context_manager.update_topic(topic_id, additional_summary, additional_key_facts)
            self.logger.info(f"Successfully updated topic: {updated_id}")
            return {
                "topic_id": updated_id,
                "additional_summary": additional_summary,
                "additional_key_facts": additional_key_facts or [],
                "status": "updated"
            }
        except Exception as e:
            self.logger.error(f"Failed to update topic {topic_id}: {e}")
            return {
                "error": str(e),
                "topic_id": topic_id,
                "status": "failed"
            }
    
    def get_schema(self) -> Dict[str, Any]:
        """Get the JSON schema for this tool."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": "Update an existing topic with additional information instead of creating a new topic. Use this when continuing a previously saved topic discussion.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "topic_id": {
                            "type": "string",
                            "description": "ID of the existing topic to update (obtained from previous save_current_topic or recall_similar_topic calls)"
                        },
                        "additional_summary": {
                            "type": "string",
                            "description": "Additional summary information to merge with the existing topic"
                        },
                        "additional_key_facts": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Additional key facts to add to the existing topic"
                        }
                    },
                    "required": ["topic_id", "additional_summary"]
                }
            }
        }


class RecallSimilarTopicTool(BaseTool):
    """Tool for recalling similar topics from long-term memory."""
    
    def __init__(self, context_manager, logger: Optional[logging.Logger] = None):
        super().__init__(
            name="recall_similar_topic",
            description="Search for and recall a similar topic from long-term memory"
        )
        self.context_manager = context_manager
        self.logger = logger or logging.getLogger(__name__)
    
    async def execute(self, user_message: str = None, query: str = None, **kwargs) -> Dict[str, Any]:
        """
        Recall similar topics from context.
        
        Args:
            user_message: The current user message to find similar topics for
            query: Alternative parameter name for user_message (for flexibility)
            **kwargs: Additional arguments (ignored for flexibility)
            
        Returns:
            Dictionary with recalled topics
        """
        try:
            # Debug logging to see what parameters we received
            self.logger.info(f"RecallSimilarTopicTool called with user_message='{user_message}', query='{query}', kwargs={kwargs}")
            
            # Use user_message if provided, otherwise try query, otherwise extract from kwargs
            search_text = user_message or query
            
            # Check if user_message might be in kwargs with different name
            if not search_text:
                search_text = kwargs.get('user_message') or kwargs.get('query') or kwargs.get('message') or kwargs.get('text')
            
            # If still no search text, try to get it from recent conversation context
            if not search_text:
                # Get the most recent user message from the FIFO queue
                queue_messages = self.context_manager.context.fifo_queue.get_messages()
                if queue_messages:
                    # Find the most recent user message
                    for msg in reversed(queue_messages):
                        if hasattr(msg, 'role') and msg.role == 'user':
                            search_text = msg.content
                            self.logger.info(f"Using recent user message for search: {search_text[:50]}...")
                            break
            
            if not search_text:
                self.logger.info("No search text provided for recall_similar_topic. Returning empty result to allow demo to continue.")
                return {
                    "recalled_topics": [],
                    "status": "no_query_provided_but_continuing",
                    "message": "No search query provided, but topic recall attempted. System continues normally."
                }
            
            # Ignore any extra parameters that might be passed by the LLM
            topics = self.context_manager.recall_similar_topics(search_text)
            self.logger.info(f"Recalled {len(topics)} similar topics for query: {search_text[:50]}...")
            
            return {
                "recalled_topics": topics,
                "total_recalled": len(topics),
                "status": "success"
            }
            
        except Exception as e:
            self.logger.error(f"Error recalling similar topics: {e}")
            return {
                "recalled_topics": [],
                "status": "error",
                "error": str(e)
            }
    
    def get_schema(self) -> Dict[str, Any]:
        """Get the JSON schema for this tool."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": "Search for and recall a similar topic from long-term memory. Use this to check if the current topic has been discussed before and to load relevant context.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user_message": {
                            "type": "string",
                            "description": "The user's current message content to search for similar topics. If not provided, will search based on current conversation context."
                        }
                    },
                    "required": []
                }
            }
        }


class EvictOldestTopicTool(BaseTool):
    """Tool for evicting the oldest topic from context when approaching token limits."""
    
    def __init__(self, context_manager, logger: Optional[logging.Logger] = None):
        super().__init__(
            name="evict_oldest_topic",
            description="Evict the oldest topic from context to make room for new content"
        )
        self.context_manager = context_manager
        self.logger = logger or logging.getLogger(__name__)
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Evict the oldest topic from context.
        
        Args:
            **kwargs: Additional arguments (ignored for flexibility)
        
        Returns:
            Dictionary with eviction confirmation
        """
        try:
            # Ignore any extra parameters that might be passed by the LLM
            topic_id = self.context_manager.evict_oldest_topic()
            
            if topic_id:
                self.logger.info(f"Successfully evicted oldest topic: {topic_id}")
                return {
                    "evicted_topic_id": topic_id,
                    "status": "evicted"
                }
            else:
                self.logger.info("No topics available to evict")
                return {
                    "evicted_topic_id": None,
                    "status": "no_topics_to_evict"
                }
                
        except Exception as e:
            self.logger.error(f"Failed to evict oldest topic: {e}")
            return {
                "error": str(e),
                "status": "failed"
            }
    
    def get_schema(self) -> Dict[str, Any]:
        """Get the JSON schema for this tool."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": "Evict the oldest topic from context to make room for new content. Use this when approaching token limits to maintain conversation flow.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        }


class ContextManagementToolFactory:
    """Factory class for creating and managing context management tools."""
    
    def __init__(self, context_manager, logger: Optional[logging.Logger] = None):
        """
        Initialize the factory.
        
        Args:
            context_manager: The context manager instance
            logger: Optional logger instance
        """
        self.context_manager = context_manager
        self.logger = logger or logging.getLogger(__name__)
        self._tools = {}
        self._create_tools()
    
    def _create_tools(self):
        """Create all context management tools."""
        self._tools = {
            "save_current_topic": SaveCurrentTopicTool(self.context_manager, self.logger),
            "update_topic": UpdateTopicTool(self.context_manager, self.logger),
            "recall_similar_topic": RecallSimilarTopicTool(self.context_manager, self.logger),
            "evict_oldest_topic": EvictOldestTopicTool(self.context_manager, self.logger)
        }
        self.logger.info(f"Created {len(self._tools)} context management tools")
    
    def get_tool(self, tool_name: str) -> Optional[BaseTool]:
        """
        Get a specific context management tool.
        
        Args:
            tool_name: Name of the tool to retrieve
            
        Returns:
            Tool instance or None if not found
        """
        return self._tools.get(tool_name)
    
    def get_all_tools(self) -> List[BaseTool]:
        """
        Get all context management tools.
        
        Returns:
            List of all tool instances
        """
        return list(self._tools.values())
    
    def get_tool_names(self) -> List[str]:
        """
        Get names of all available tools.
        
        Returns:
            List of tool names
        """
        return list(self._tools.keys())
    
    def register_tools_with_executor(self, tool_executor):
        """
        Register all context management tools with a tool executor.
        
        Args:
            tool_executor: The tool executor to register tools with
        """
        for tool in self._tools.values():
            tool_executor.register_tool(tool)
        
        self.logger.info(f"Registered {len(self._tools)} context management tools with executor") 