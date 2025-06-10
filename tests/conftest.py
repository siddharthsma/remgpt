"""
Pytest configuration and shared fixtures for RemGPT API tests.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import sys
import os

# Add the project root to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(autouse=True)
def mock_sentence_transformers():
    """
    Mock sentence transformers to avoid loading heavy ML models during tests.
    This fixture is auto-used for all tests.
    """
    with patch('remgpt.detection.topic_drift_detector.SentenceTransformer') as mock_st:
        # Mock the sentence transformer model
        mock_model = MagicMock()
        mock_model.encode.return_value = [0.1, 0.2, 0.3, 0.4]  # Mock embedding
        mock_st.return_value = mock_model
        yield mock_st


@pytest.fixture(autouse=True)
def disable_background_tasks():
    """
    Disable background tasks during testing to avoid interference.
    This fixture is auto-used for all tests.
    """
    with patch('remgpt.api.process_message_queue'):
        yield


@pytest.fixture
def sample_user_message():
    """Create a sample UserMessage for testing."""
    from remgpt.core.types import UserMessage
    return UserMessage(
        content="Hello, this is a test message.",
        name="test_user_123"
    )


@pytest.fixture
def sample_stream_events():
    """Create sample StreamEvent objects for testing."""
    from remgpt.orchestration import StreamEvent
    return [
        StreamEvent(
            type="processing_start",
            data={"message_role": "user"},
            timestamp=1640995200.0
        ),
        StreamEvent(
            type="topic_drift_detected",
            data={"drift_detected": False, "similarity_score": 0.85},
            timestamp=1640995201.0
        ),
        StreamEvent(
            type="llm_response_chunk",
            data={"content": "Hello! How can I help you today?"},
            timestamp=1640995202.0
        ),
        StreamEvent(
            type="processing_complete",
            data={"duration": 2.0, "final_context": {"total_tokens": 150}},
            timestamp=1640995204.0
        )
    ]


@pytest.fixture
def mock_context_manager():
    """Create a mock context manager for testing."""
    mock = MagicMock()
    mock.get_context_summary.return_value = {
        "total_tokens": 100,
        "within_limit": True,
        "block_token_counts": {"system": 20, "memory": 30, "queue": 50},
        "total_messages": 5,
        "queue_size": 3
    }
    mock.check_token_limit.return_value = True
    mock.max_tokens = 4000
    return mock


@pytest.fixture
def auth_headers():
    """Create valid authorization headers for testing."""
    return {"Authorization": "Bearer test_token_12345"}


@pytest.fixture
def invalid_auth_headers():
    """Create invalid authorization headers for testing."""
    return [
        {},  # No auth header
        {"Authorization": "Invalid format"},  # Wrong format
        {"Authorization": "Bearer "},  # Empty token
        {"Authorization": "Bearer short"},  # Too short token
    ]


@pytest.fixture
def sample_context_config():
    """Create sample context configuration data."""
    return {
        "max_tokens": 5000,
        "system_instructions": "You are a helpful AI assistant for testing.",
        "memory_content": "Remember that this is a test environment.",
        "tools": [
            {"name": "test_tool", "description": "A tool for testing"}
        ],
        "model": "gpt-4"
    }


# Test markers for different test categories
pytest.mark.unit = pytest.mark.unit
pytest.mark.integration = pytest.mark.integration
pytest.mark.auth = pytest.mark.auth
pytest.mark.streaming = pytest.mark.streaming
pytest.mark.performance = pytest.mark.performance


@pytest.fixture
def mock_logger():
    """Provide a mock logger for tests."""
    return Mock()


@pytest.fixture
def sample_tool_schema():
    """Provide a sample tool schema for testing."""
    return {
        "name": "sample_tool",
        "description": "A sample tool for testing",
        "parameters": {
            "type": "object",
            "properties": {
                "input": {"type": "string", "description": "Input parameter"},
                "count": {"type": "integer", "minimum": 1, "maximum": 10}
            },
            "required": ["input"]
        }
    }


@pytest.fixture
def sample_tool_schemas():
    """Provide multiple sample tool schemas for testing."""
    return [
        {
            "name": "tool_1",
            "description": "First test tool",
            "parameters": {
                "type": "object",
                "properties": {
                    "param1": {"type": "string"}
                },
                "required": ["param1"]
            }
        },
        {
            "name": "tool_2",
            "description": "Second test tool",
            "parameters": {
                "type": "object",
                "properties": {
                    "param2": {"type": "number"}
                }
            }
        }
    ]


# Pytest configuration for async tests
pytest_plugins = ['pytest_asyncio'] 