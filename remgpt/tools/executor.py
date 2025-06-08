import asyncio
import logging
from typing import Dict, Any, Optional, List, Callable
from .base import BaseTool


logger = logging.getLogger(__name__)


class ToolExecutor:
    """Handles execution of tools called by LLMs."""
    
    def __init__(self):
        """Initialize the tool executor."""
        self._tools: Dict[str, BaseTool] = {}
        self._tool_results: Dict[str, Any] = {}
        
    def register_tool(self, tool: BaseTool) -> None:
        """
        Register a tool for execution.
        
        Args:
            tool: Tool instance to register
        """
        if not isinstance(tool, BaseTool):
            raise TypeError("Tool must inherit from BaseTool")
        
        self._tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")
    
    def unregister_tool(self, tool_name: str) -> None:
        """
        Unregister a tool.
        
        Args:
            tool_name: Name of tool to unregister
        """
        if tool_name in self._tools:
            del self._tools[tool_name]
            logger.info(f"Unregistered tool: {tool_name}")
    
    def get_registered_tools(self) -> List[str]:
        """Get list of registered tool names."""
        return list(self._tools.keys())
    
    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """
        Get schemas for all registered tools.
        
        Returns:
            List[Dict[str, Any]]: List of tool schemas
        """
        schemas = []
        for tool in self._tools.values():
            try:
                schema = tool.get_schema()
                schemas.append(schema)
            except Exception as e:
                logger.error(f"Error getting schema for tool {tool.name}: {e}")
        
        return schemas
    
    async def execute_tool(self, tool_call_id: str, tool_name: str, 
                          tool_args: Dict[str, Any]) -> Any:
        """
        Execute a tool and store the result.
        
        Args:
            tool_call_id: Unique ID for this tool call
            tool_name: Name of the tool to execute
            tool_args: Arguments to pass to the tool
            
        Returns:
            Any: Tool execution result
            
        Raises:
            ValueError: If tool is not registered
            Exception: If tool execution fails
        """
        if tool_name not in self._tools:
            error_msg = f"Tool '{tool_name}' not registered. Available tools: {list(self._tools.keys())}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        tool = self._tools[tool_name]
        
        try:
            # Validate arguments
            if not tool.validate_args(tool_args):
                raise ValueError(f"Invalid arguments for tool {tool_name}: {tool_args}")
            
            logger.info(f"Executing tool {tool_name} with call ID {tool_call_id}")
            
            # Execute the tool
            result = await tool.execute(**tool_args)
            
            # Store result
            self._tool_results[tool_call_id] = result
            
            logger.info(f"Tool {tool_name} executed successfully (call ID: {tool_call_id})")
            return result
            
        except Exception as e:
            error_msg = f"Error executing tool {tool_name}: {str(e)}"
            logger.error(error_msg)
            
            # Store error as result
            self._tool_results[tool_call_id] = {"error": error_msg}
            raise
    
    def get_tool_result(self, tool_call_id: str) -> Optional[Any]:
        """
        Get the result of a tool execution.
        
        Args:
            tool_call_id: Tool call ID
            
        Returns:
            Optional[Any]: Tool result if available, None otherwise
        """
        return self._tool_results.get(tool_call_id)
    
    def clear_results(self) -> None:
        """Clear all stored tool results."""
        self._tool_results.clear()
        logger.info("Cleared all tool results")
    
    def has_tool(self, tool_name: str) -> bool:
        """
        Check if a tool is registered.
        
        Args:
            tool_name: Tool name to check
            
        Returns:
            bool: True if tool is registered, False otherwise
        """
        return tool_name in self._tools
    
    async def execute_multiple_tools(self, tool_calls: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Execute multiple tools concurrently.
        
        Args:
            tool_calls: List of tool call dictionaries with keys:
                       'id', 'name', 'args'
        
        Returns:
            Dict[str, Any]: Mapping of tool_call_id to results
        """
        tasks = []
        
        for tool_call in tool_calls:
            task = self.execute_tool(
                tool_call_id=tool_call['id'],
                tool_name=tool_call['name'],
                tool_args=tool_call['args']
            )
            tasks.append(task)
        
        # Execute all tools concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Map results back to tool call IDs
        result_map = {}
        for i, tool_call in enumerate(tool_calls):
            result_map[tool_call['id']] = results[i]
        
        return result_map 