"""
Factory functions for creating orchestrators with remote tool support.
"""

from typing import List, Optional, Dict, Any
import logging
from .orchestrator import ConversationOrchestrator
from ..context import LLMContextManager
from ..llm import BaseLLMClient
from ..tools import ToolExecutor
from ..storage import VectorDatabase


async def create_orchestrator(
    context_manager: LLMContextManager,
    llm_client: Optional[BaseLLMClient] = None,
    tool_executor: Optional[ToolExecutor] = None,
    vector_database: Optional[VectorDatabase] = None,
    drift_detection_config: Optional[Dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None,
    # Remote tool configuration
    mcp_servers: Optional[List[str]] = None,
    a2a_agents: Optional[List[str]] = None,
    auto_initialize_remote_tools: bool = True
) -> ConversationOrchestrator:
    """
    Create a ConversationOrchestrator with optional remote tool support.
    
    Args:
        context_manager: The context manager instance
        llm_client: LLM client instance (BaseLLMClient)
        tool_executor: Tool executor for handling tool calls
        vector_database: Vector database for storing topics
        drift_detection_config: Configuration for drift detection
        logger: Optional logger instance
        mcp_servers: List of MCP server URLs or paths (e.g., ["uvx pymupdf4llm-mcp@latest stdio"])
        a2a_agents: List of A2A agent base URLs (e.g., ["http://localhost:8000"])
        auto_initialize_remote_tools: Whether to automatically initialize remote tools
        
    Returns:
        Configured ConversationOrchestrator with remote tools initialized
        
    Example:
        ```python
        # With MCP servers
        orchestrator = await create_orchestrator(
            context_manager=context_manager,
            llm_client=openai_client,
            mcp_servers=["uvx pymupdf4llm-mcp@latest stdio", "http://localhost:3000/mcp"]
        )
        
        # With A2A agents
        orchestrator = await create_orchestrator(
            context_manager=context_manager,
            llm_client=claude_client,
            a2a_agents=["http://localhost:8000", "http://localhost:8002"]
        )
        
        # With both
        orchestrator = await create_orchestrator(
            context_manager=context_manager,
            llm_client=gemini_client,
            mcp_servers=["uvx weather-mcp@latest stdio"],
            a2a_agents=["http://localhost:8001"]
        )
        ```
    """
    orchestrator = ConversationOrchestrator(
        context_manager=context_manager,
        llm_client=llm_client,
        tool_executor=tool_executor,
        vector_database=vector_database,
        drift_detection_config=drift_detection_config,
        logger=logger,
        mcp_servers=mcp_servers,
        a2a_agents=a2a_agents
    )
    
    # Initialize remote tools if requested
    if auto_initialize_remote_tools and (mcp_servers or a2a_agents):
        await orchestrator.initialize_remote_tools()
    
    return orchestrator


async def create_orchestrator_with_config(config: Dict[str, Any]) -> ConversationOrchestrator:
    """
    Create a ConversationOrchestrator from a configuration dictionary.
    
    Args:
        config: Configuration dictionary with keys:
            - context_manager: LLMContextManager instance (required)
            - llm_client: BaseLLMClient instance (optional)
            - tool_executor: ToolExecutor instance (optional)
            - vector_database: VectorDatabase instance (optional)
            - drift_detection_config: Dict[str, Any] (optional)
            - logger: logging.Logger (optional)
            - mcp_servers: List[str] (optional)
            - a2a_agents: List[str] (optional)
            - auto_initialize_remote_tools: bool (optional, defaults to True)
            
    Returns:
        Configured ConversationOrchestrator
        
    Example:
        ```python
        config = {
            "context_manager": context_manager,
            "llm_client": openai_client,
            "mcp_servers": ["uvx weather-mcp@latest stdio"],
            "a2a_agents": ["http://localhost:8000"],
            "drift_detection_config": {
                "similarity_threshold": 0.8,
                "drift_threshold": 0.6
            }
        }
        
        orchestrator = await create_orchestrator_with_config(config)
        ```
    """
    context_manager = config.get("context_manager")
    if not context_manager:
        raise ValueError("context_manager is required in config")
    
    return await create_orchestrator(
        context_manager=context_manager,
        llm_client=config.get("llm_client"),
        tool_executor=config.get("tool_executor"),
        vector_database=config.get("vector_database"),
        drift_detection_config=config.get("drift_detection_config"),
        logger=config.get("logger"),
        mcp_servers=config.get("mcp_servers"),
        a2a_agents=config.get("a2a_agents"),
        auto_initialize_remote_tools=config.get("auto_initialize_remote_tools", True)
    ) 