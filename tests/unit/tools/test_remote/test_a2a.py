"""
Unit tests for A2AProtocol class.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from remgpt.tools.remote.a2a import A2AProtocol, TaskRequestBuilder


class TestTaskRequestBuilder:
    """Test the TaskRequestBuilder class."""
    
    def test_build_send_task_params_minimal(self):
        """Test building task parameters with minimal required fields."""
        params = TaskRequestBuilder.build_send_task_params(
            task_id="test-123",
            text="Hello, agent!"
        )
        
        assert params["id"] == "test-123"
        assert params["message"]["role"] == "user"
        assert len(params["message"]["parts"]) == 1
        assert params["message"]["parts"][0]["type"] == "text"
        assert params["message"]["parts"][0]["text"] == "Hello, agent!"
        assert "sessionId" in params
    
    def test_build_send_task_params_complete(self):
        """Test building task parameters with all fields."""
        params = TaskRequestBuilder.build_send_task_params(
            task_id="test-456",
            role="agent",
            text="Process this data",
            data={"key": "value"},
            file_uri="file://example.txt",
            session_id="session-123",
            metadata={"source": "test"}
        )
        
        assert params["id"] == "test-456"
        assert params["sessionId"] == "session-123"
        assert params["message"]["role"] == "agent"
        assert params["metadata"] == {"source": "test"}
        
        # Check parts
        parts = params["message"]["parts"]
        assert len(parts) == 3
        
        text_part = next(p for p in parts if p["type"] == "text")
        assert text_part["text"] == "Process this data"
        
        data_part = next(p for p in parts if p["type"] == "data")
        assert data_part["data"] == {"key": "value"}
        
        file_part = next(p for p in parts if p["type"] == "file")
        assert file_part["file"]["uri"] == "file://example.txt"


class TestA2AProtocolInitialization:
    """Test A2AProtocol initialization."""
    
    def test_http_url_initialization(self):
        """Test initialization with HTTP URL."""
        protocol = A2AProtocol("http://localhost:8000")
        assert protocol.agent_url == "http://localhost:8000"
        assert protocol.agent_name == "agent_localhost:8000"
    
    def test_https_url_initialization(self):
        """Test initialization with HTTPS URL."""
        protocol = A2AProtocol("https://api.example.com")
        assert protocol.agent_url == "https://api.example.com"
        assert protocol.agent_name == "agent_api.example.com"
    
    def test_url_with_trailing_slash(self):
        """Test initialization with URL having trailing slash."""
        protocol = A2AProtocol("http://localhost:8000/")
        assert protocol.agent_url == "http://localhost:8000/"
        assert protocol.agent_name == "agent_"
    
    def test_custom_agent_name(self):
        """Test initialization with custom agent name."""
        protocol = A2AProtocol("http://localhost:8000", "my_agent")
        assert protocol.agent_url == "http://localhost:8000"
        assert protocol.agent_name == "my_agent"


class TestA2AProtocolWithoutDependencies:
    """Test A2AProtocol behavior when httpx is not available."""
    
    @patch('remgpt.tools.remote.a2a.HTTPX_AVAILABLE', False)
    def test_import_error_handling(self):
        """Test that ImportError is handled gracefully when httpx is not available."""
        with pytest.raises(ImportError, match="httpx package required"):
            A2AProtocol("http://localhost:8000")


@pytest.fixture
def mock_httpx():
    """Mock httpx for testing."""
    with patch('remgpt.tools.remote.a2a.httpx') as mock_httpx:
        mock_client = AsyncMock()
        mock_httpx.AsyncClient.return_value.__aenter__.return_value = mock_client
        yield {'httpx': mock_httpx, 'client': mock_client}


class TestA2AProtocolBasicOperations:
    """Test A2A protocol basic operations."""
    
    @pytest.mark.asyncio
    async def test_list_tools(self):
        """Test listing available tools."""
        protocol = A2AProtocol("http://localhost:8000")
        
        tools = await protocol.list_tools()
        
        # A2A protocol provides a single 'send_task' tool
        assert len(tools) == 1
        assert tools[0]["name"] == "send_task"
        assert "Send a task" in tools[0]["description"]
        assert "inputSchema" in tools[0]
        assert "text" in tools[0]["inputSchema"]["properties"]
        assert "data" in tools[0]["inputSchema"]["properties"]
        assert "role" in tools[0]["inputSchema"]["properties"]
        assert "file_uri" in tools[0]["inputSchema"]["properties"]
        assert "session_id" in tools[0]["inputSchema"]["properties"]
    
    def test_agent_name_generation(self):
        """Test automatic agent name generation."""
        protocol1 = A2AProtocol("http://localhost:8000")
        protocol2 = A2AProtocol("https://api.example.com/agent")
        protocol3 = A2AProtocol("http://localhost:8000", "custom_name")
        
        assert protocol1.agent_name == "agent_localhost:8000"
        assert protocol2.agent_name == "agent_agent"
        assert protocol3.agent_name == "custom_name"


class TestA2AProtocolToolOperations:
    """Test A2A protocol tool operations."""
    
    @pytest.mark.asyncio
    async def test_call_tool_invalid_name(self):
        """Test calling an invalid tool name."""
        protocol = A2AProtocol("http://localhost:8000")
        
        with pytest.raises(ValueError, match="Unknown A2A tool: invalid_tool"):
            await protocol.call_tool("invalid_tool", {"text": "hello"})
    
    @pytest.mark.asyncio
    async def test_send_task_success(self, mock_httpx):
        """Test successful task sending."""
        protocol = A2AProtocol("http://localhost:8000")
        
        # Mock successful JSON-RPC response
        mock_response = Mock()
        mock_response.json.return_value = {
            "jsonrpc": "2.0",
            "id": "request-123",
            "result": {
                "id": "task-456",
                "sessionId": "session-789",
                "status": {"state": "submitted"},
                "metadata": {"source": "test"}
            }
        }
        mock_response.raise_for_status = Mock()
        mock_httpx['client'].post.return_value = mock_response
        
        # Call send_task
        result = await protocol.send_task("Hello, agent!")
        
        # Verify results
        assert result["task_id"] == "task-456"
        assert result["status"] == "submitted"
        assert result["session_id"] == "session-789"
        assert "response" in result
        
        # Verify HTTP call was made
        mock_httpx['client'].post.assert_called_once()
        call_args = mock_httpx['client'].post.call_args
        assert call_args[0][0] == "http://localhost:8000"
        
        # Verify JSON-RPC structure
        json_payload = call_args[1]["json"]
        assert json_payload["jsonrpc"] == "2.0"
        assert json_payload["method"] == "tasks/send"
        assert "params" in json_payload
        assert "id" in json_payload
    
    @pytest.mark.asyncio
    async def test_call_tool_with_data(self, mock_httpx):
        """Test tool call with additional data."""
        protocol = A2AProtocol("http://localhost:8000")
        
        # Mock successful response
        mock_response = Mock()
        mock_response.json.return_value = {
            "jsonrpc": "2.0",
            "id": "request-123",
            "result": {
                "id": "task-456",
                "status": {"state": "working"}
            }
        }
        mock_response.raise_for_status = Mock()
        mock_httpx['client'].post.return_value = mock_response
        
        # Call tool with complex data
        result = await protocol.call_tool("send_task", {
            "text": "Process this data",
            "data": {"key": "value", "numbers": [1, 2, 3]},
            "role": "agent",
            "file_uri": "file://example.txt",
            "session_id": "custom-session",
            "metadata": {"priority": "high"}
        })
        
        # Verify results
        assert result["task_id"] == "task-456"
        assert result["status"] == "working"
        
        # Verify request structure
        call_args = mock_httpx['client'].post.call_args
        json_payload = call_args[1]["json"]
        params = json_payload["params"]
        
        assert params["sessionId"] == "custom-session"
        assert params["metadata"] == {"priority": "high"}
        assert params["message"]["role"] == "agent"
        
        # Check message parts
        parts = params["message"]["parts"]
        text_part = next(p for p in parts if p["type"] == "text")
        data_part = next(p for p in parts if p["type"] == "data")
        file_part = next(p for p in parts if p["type"] == "file")
        
        assert text_part["text"] == "Process this data"
        assert data_part["data"] == {"key": "value", "numbers": [1, 2, 3]}
        assert file_part["file"]["uri"] == "file://example.txt"
    
    @pytest.mark.asyncio
    async def test_call_tool_error_response(self, mock_httpx):
        """Test handling of JSON-RPC error responses."""
        protocol = A2AProtocol("http://localhost:8000")
        
        # Mock error response
        mock_response = Mock()
        mock_response.json.return_value = {
            "jsonrpc": "2.0",
            "id": "request-123",
            "error": {
                "code": -32001,
                "message": "Task not found"
            }
        }
        mock_response.raise_for_status = Mock()
        mock_httpx['client'].post.return_value = mock_response
        
        with pytest.raises(Exception, match="A2A error \\(-32001\\): Task not found"):
            await protocol.call_tool("send_task", {"text": "hello"})
    
    @pytest.mark.asyncio
    async def test_call_tool_http_error(self, mock_httpx):
        """Test tool call with HTTP error."""
        protocol = A2AProtocol("http://localhost:8000")
        
        # Mock HTTP error
        mock_httpx['client'].post.side_effect = Exception("HTTP 500 Server Error")
        
        with pytest.raises(Exception, match="HTTP 500 Server Error"):
            await protocol.call_tool("send_task", {"text": "hello"})
    
    @pytest.mark.asyncio
    async def test_send_task_direct_method(self, mock_httpx):
        """Test the direct send_task method."""
        protocol = A2AProtocol("http://localhost:8000", "test_agent")
        
        # Mock successful response
        mock_response = Mock()
        mock_response.json.return_value = {
            "jsonrpc": "2.0",
            "id": "request-123",
            "result": {
                "id": "task-789",
                "status": {"state": "completed"}
            }
        }
        mock_response.raise_for_status = Mock()
        mock_httpx['client'].post.return_value = mock_response
        
        # Call send_task directly
        result = await protocol.send_task(
            text="Direct task call",
            role="user",
            data={"test": True}
        )
        
        # Verify results
        assert result["task_id"] == "task-789"
        assert result["status"] == "completed"


class TestA2AProtocolIntegration:
    """Integration tests for A2A protocol."""
    
    @pytest.mark.asyncio
    async def test_full_workflow(self, mock_httpx):
        """Test complete A2A workflow."""
        protocol = A2AProtocol("https://api.example.com", "test_agent")
        
        # Mock successful response
        mock_response = Mock()
        mock_response.json.return_value = {
            "jsonrpc": "2.0",
            "id": "request-123",
            "result": {
                "id": "task-456",
                "sessionId": "session-789",
                "status": {"state": "completed", "message": None},
                "artifacts": [
                    {
                        "name": "result",
                        "parts": [{"type": "text", "text": "Task completed successfully"}]
                    }
                ]
            }
        }
        mock_response.raise_for_status = Mock()
        mock_httpx['client'].post.return_value = mock_response
        
        # Execute workflow
        tools = await protocol.list_tools()
        assert len(tools) == 1
        assert tools[0]["name"] == "send_task"
        
        result = await protocol.call_tool("send_task", {"text": "Complete this task"})
        assert result["task_id"] == "task-456"
        assert result["status"] == "completed"
        assert result["session_id"] == "session-789"
        
        # Verify call was made with proper JSON-RPC structure
        call_args = mock_httpx['client'].post.call_args
        json_payload = call_args[1]["json"]
        assert json_payload["jsonrpc"] == "2.0"
        assert json_payload["method"] == "tasks/send" 