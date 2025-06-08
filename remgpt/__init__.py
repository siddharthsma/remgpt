"""
RemGPT - A Python library for RemGPT functionality.
"""

__version__ = "0.1.0"
__author__ = "Siddharth Ambegaonkar"
__email__ = "sid.ambegaonkar@gmail.com"

# Import main classes/functions
from .context import (
    LLMContext, 
    LLMContextManager,
    TokenCounter,
    BaseBlock,
    SystemInstructionsBlock,
    MemoryInstructionsBlock,
    ToolsDefinitionsBlock,
    WorkingContextBlock,
    FIFOQueueBlock,
    create_context_manager,
    create_context_with_config
)
from .orchestration import ConversationOrchestrator
from .detection import TopicDriftDetector, PageHinkleyTest, EmbeddingResult
from .summarization import TopicSummarizer, Topic
from .storage import VectorDatabase, QdrantVectorDatabase, InMemoryVectorDatabase
from .llm import (
    BaseLLMClient,
    LLMClientFactory,
    OpenAIClient,
    ClaudeClient,
    GeminiClient,
    Event,
    EventType
)
from .tools import ToolExecutor, BaseTool
from .types import (
    Message,
    SystemMessage,
    UserMessage,
    AssistantMessage,
    ToolMessage,
    MessageRole,
    ContentType,
    ImageDetail,
    TextContent,
    ImageContent,
    ToolCall
)

__all__ = [
    "LLMContext",
    "LLMContextManager", 
    "TokenCounter",
    "BaseBlock",
    "SystemInstructionsBlock",
    "MemoryInstructionsBlock",
    "ToolsDefinitionsBlock",
    "WorkingContextBlock",
    "FIFOQueueBlock",
    "create_context_manager",
    "create_context_with_config",
    "ConversationOrchestrator",
    "TopicDriftDetector",
    "PageHinkleyTest",
    "EmbeddingResult",
    "TopicSummarizer",
    "Topic",
    "VectorDatabase",
    "QdrantVectorDatabase", 
    "InMemoryVectorDatabase",
    "BaseLLMClient",
    "LLMClientFactory", 
    "OpenAIClient",
    "ClaudeClient",
    "GeminiClient",
    "Event",
    "EventType",
    "ToolExecutor",
    "BaseTool",
    "Message",
    "SystemMessage",
    "UserMessage", 
    "AssistantMessage",
    "ToolMessage",
    "MessageRole",
    "ContentType",
    "ImageDetail",
    "TextContent",
    "ImageContent",
    "ToolCall",
] 