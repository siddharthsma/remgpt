"""
Integration tests for tools module components.
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from remgpt.tools.base import BaseTool
from remgpt.tools.executor import ToolExecutor
from remgpt.tools.remote.base import RemoteToolProtocol
from remgpt.tools.remote.tool import RemoteTool
from remgpt.tools.remote.manager import RemoteToolManager


class LocalTestTool(BaseTool):
    """Local tool for integration testing."""
    
    def __init__(self):
        super().__init__("local_calculator", "Local calculator tool")
        self.execution_count = 0
    
    async def execute(self, operation, a, b):
        """Execute local calculation."""
        self.execution_count += 1
        
        if operation == "add":
            return {"result": a + b, "type": "local"}
        elif operation == "multiply":
            return {"result": a * b, "type": "local"}
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    def get_schema(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "operation": {"type": "string", "enum": ["add", "multiply"]},
                        "a": {"type": "number"},
                        "b": {"type": "number"}
                    },
                    "required": ["operation", "a", "b"]
                }
            }
        }
    
    def validate_args(self, args):
        """Validate arguments."""
        required = {"operation", "a", "b"}
        if not required.issubset(args.keys()):
            return False
        if args["operation"] not in ["add", "multiply"]:
            return False
        return True


class MockRemoteProtocol(RemoteToolProtocol):
    """Mock remote protocol for integration testing."""
    
    def __init__(self, connection_string="mock://remote"):
        self.connection_string = connection_string
        self.is_connected = False
        self.tools_data = [
            {
                "name": "send_task",
                "description": "Send a task to the mock agent",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "Task text"},
                        "data": {"type": "object", "description": "Optional data"},
                        "role": {"type": "string", "enum": ["user", "agent"], "default": "user"}
                    },
                    "required": ["text"]
                }
            }
        ]
    
    async def connect(self):
        self.is_connected = True
    
    async def disconnect(self):
        self.is_connected = False
    
    async def list_tools(self):
        if not self.is_connected:
            raise RuntimeError("Not connected")
        return self.tools_data
    
    async def call_tool(self, name, arguments):
        if not self.is_connected:
            raise RuntimeError("Not connected")
        
        if name == "send_task":
            text = arguments["text"]
            data = arguments.get("data", {})
            role = arguments.get("role", "user")
            
            # Mock task response
            return {
                "task_id": f"task-{hash(text) % 10000}",
                "status": "completed",
                "response": {
                    "text": f"Processed: {text}",
                    "data": data,
                    "role": role
                },
                "type": "remote"
            }
        
        raise ValueError(f"Unknown tool: {name}")


class TestToolsIntegration:
    """Integration tests for tools components."""
    
    @pytest.mark.asyncio
    async def test_local_and_remote_tools_together(self):
        """Test using local and remote tools together."""
        # Setup local tool
        local_tool = LocalTestTool()
        
        # Setup remote tools
        remote_protocol = MockRemoteProtocol()
        await remote_protocol.connect()
        
        # Create remote tool manually
        tools_info = await remote_protocol.list_tools()
        remote_tool = RemoteTool(tools_info[0], remote_protocol, "mock")
        
        # Setup executor with both local and remote tools
        executor = ToolExecutor()
        executor.register_tool(local_tool)
        executor.register_tool(remote_tool)
        
        # Test that we have both tools
        assert len(executor.get_registered_tools()) == 2
        assert executor.has_tool("local_calculator") is True
        assert executor.has_tool("mock_send_task") is True
        
        # Execute local tool
        calc_result = await executor.execute_tool("call_1", "local_calculator", {
            "operation": "add",
            "a": 10,
            "b": 5
        })
        assert calc_result == {"result": 15, "type": "local"}
        
        # Execute remote tool
        task_result = await executor.execute_tool("call_2", "mock_send_task", {
            "text": "Process calculation result: 15",
            "data": {"result": 15}
        })
        assert task_result["type"] == "remote"
        assert task_result["status"] == "completed"
        assert "task_id" in task_result
        
        # Cleanup
        await remote_protocol.disconnect()
    
    @pytest.mark.asyncio
    async def test_tool_chaining_workflow(self):
        """Test chaining local and remote tools in a workflow."""
        # Setup tools
        local_tool = LocalTestTool()
        remote_protocol = MockRemoteProtocol()
        await remote_protocol.connect()
        
        # Create remote tool manually
        tools_info = await remote_protocol.list_tools()
        remote_tool = RemoteTool(tools_info[0], remote_protocol, "mock")
        
        executor = ToolExecutor()
        executor.register_tool(local_tool)
        executor.register_tool(remote_tool)
        
        # Workflow: Calculate locally, then format remotely
        # Step 1: Calculate 10 * 3 = 30
        calc_result = await executor.execute_tool("call_1", "local_calculator", {
            "operation": "multiply",
            "a": 10,
            "b": 3
        })
        calculated_value = calc_result["result"]
        
        # Step 2: Send task with calculation result
        task_result = await executor.execute_tool("call_2", "mock_send_task", {
            "text": f"Process calculation result: {calculated_value}",
            "data": {"calculation": calculated_value, "operation": "multiply"}
        })
        
        # Verify workflow
        assert calculated_value == 30
        assert task_result["status"] == "completed"
        assert task_result["response"]["text"] == f"Processed: Process calculation result: {calculated_value}"
        
        await remote_protocol.disconnect()
    
    @pytest.mark.asyncio
    async def test_tool_schema_generation(self):
        """Test schema generation for mixed tool types."""
        local_tool = LocalTestTool()
        remote_protocol = MockRemoteProtocol()
        await remote_protocol.connect()
        
        # Create remote tool manually
        tools_info = await remote_protocol.list_tools()
        remote_tool = RemoteTool(tools_info[0], remote_protocol, "mock")
        
        executor = ToolExecutor()
        executor.register_tool(local_tool)
        executor.register_tool(remote_tool)
        
        # Get all schemas
        schemas = executor.get_tool_schemas()
        assert len(schemas) == 2
        
        # Verify schema structure
        for schema in schemas:
            assert schema["type"] == "function"
            assert "function" in schema
            assert "name" in schema["function"]
            assert "description" in schema["function"]
            assert "parameters" in schema["function"]
        
        # Verify specific schemas
        schema_names = {schema["function"]["name"] for schema in schemas}
        assert "local_calculator" in schema_names
        assert "mock_send_task" in schema_names
        
        await remote_protocol.disconnect()
    
    @pytest.mark.asyncio
    async def test_error_handling_across_tool_types(self):
        """Test error handling for both local and remote tools."""
        local_tool = LocalTestTool()
        remote_protocol = MockRemoteProtocol()
        await remote_protocol.connect()
        
        # Create remote tool manually
        tools_info = await remote_protocol.list_tools()
        remote_tool = RemoteTool(tools_info[0], remote_protocol, "mock")
        
        executor = ToolExecutor()
        executor.register_tool(local_tool)
        executor.register_tool(remote_tool)
        
        # Test local tool error (validation catches invalid operation)
        with pytest.raises(ValueError, match="Invalid arguments"):
            await executor.execute_tool("call_1", "local_calculator", {
                "operation": "divide",  # Invalid operation
                "a": 10,
                "b": 2
            })
        
        # Test remote tool error
        with pytest.raises(ValueError, match="Unknown tool"):
            await remote_protocol.call_tool("nonexistent_tool", {})
        
        # Test validation errors
        with pytest.raises(ValueError, match="Invalid arguments"):
            await executor.execute_tool("call_2", "local_calculator", {
                "operation": "add",
                "a": 10
                # Missing 'b' parameter
            })
        
        await remote_protocol.disconnect()
    
    @pytest.mark.asyncio
    async def test_tool_replacement_and_updates(self):
        """Test updating tools in the executor."""
        local_tool = LocalTestTool()
        executor = ToolExecutor()
        executor.register_tool(local_tool)
        
        # Initial execution
        result1 = await executor.execute_tool("call_1", "local_calculator", {
            "operation": "add", "a": 5, "b": 3
        })
        assert result1 == {"result": 8, "type": "local"}
        assert local_tool.execution_count == 1
        
        # Replace with a different local tool (same name)
        class UpdatedLocalTool(BaseTool):
            def __init__(self):
                super().__init__("local_calculator", "Updated calculator")
            
            async def execute(self, operation, a, b):
                if operation == "add":
                    return {"result": a + b + 1, "type": "updated"}  # Add 1 for difference
                return {"result": 0, "type": "updated"}
            
            def get_schema(self):
                return {
                    "type": "function",
                    "function": {
                        "name": self.name,
                        "description": self.description,
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "operation": {"type": "string"},
                                "a": {"type": "number"},
                                "b": {"type": "number"}
                            },
                            "required": ["operation", "a", "b"]
                        }
                    }
                }
        
        updated_tool = UpdatedLocalTool()
        executor.register_tool(updated_tool)  # Should replace the old tool
        
        # Execute updated tool
        result2 = await executor.execute_tool("call_2", "local_calculator", {
            "operation": "add", "a": 5, "b": 3
        })
        assert result2 == {"result": 9, "type": "updated"}  # 8 + 1
        
        # Old tool should not have been called again
        assert local_tool.execution_count == 1
    
    @pytest.mark.asyncio
    async def test_concurrent_tool_execution(self):
        """Test executing multiple tools concurrently."""
        import asyncio
        
        local_tool = LocalTestTool()
        remote_protocol = MockRemoteProtocol()
        await remote_protocol.connect()
        
        # Create remote tool manually
        tools_info = await remote_protocol.list_tools()
        remote_tool = RemoteTool(tools_info[0], remote_protocol, "mock")
        
        executor = ToolExecutor()
        executor.register_tool(local_tool)
        executor.register_tool(remote_tool)
        
        # Execute multiple tools concurrently
        tasks = [
            executor.execute_tool("call_1", "local_calculator", {"operation": "add", "a": 1, "b": 2}),
            executor.execute_tool("call_2", "local_calculator", {"operation": "multiply", "a": 3, "b": 4}),
            executor.execute_tool("call_3", "mock_send_task", {"text": "Process number 42", "data": {"value": 42}}),
            executor.execute_tool("call_4", "mock_send_task", {"text": "Process percentage 0.75", "data": {"value": 0.75}})
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Verify results
        assert results[0] == {"result": 3, "type": "local"}
        assert results[1] == {"result": 12, "type": "local"}
        assert results[2]["type"] == "remote"
        assert results[2]["status"] == "completed"
        assert "task_id" in results[2]
        assert results[3]["type"] == "remote"
        assert results[3]["status"] == "completed"
        assert "task_id" in results[3]
        
        # Local tool should have been called twice
        assert local_tool.execution_count == 2
        
        await remote_protocol.disconnect()


class TestToolsModuleIntegration:
    """Test integration with the overall tools module."""
    
    def test_tools_module_imports(self):
        """Test that all tools components can be imported together."""
        from remgpt.tools import BaseTool, ToolExecutor
        from remgpt.tools.remote import RemoteToolProtocol, RemoteTool, RemoteToolManager
        
        # All should be importable
        assert BaseTool is not None
        assert ToolExecutor is not None
        assert RemoteToolProtocol is not None
        assert RemoteTool is not None
        assert RemoteToolManager is not None
    
    @pytest.mark.asyncio
    async def test_tools_factory_integration(self):
        """Test integration with factory patterns."""
        # Create a simple tool executor with mixed tools
        executor = ToolExecutor()
        
        # Add local tool
        local_tool = LocalTestTool()
        executor.register_tool(local_tool)
        
        # Create remote tool manually
        remote_protocol = MockRemoteProtocol()
        await remote_protocol.connect()
        
        tools_info = await remote_protocol.list_tools()
        remote_tool = RemoteTool(tools_info[0], remote_protocol, "mock")
        executor.register_tool(remote_tool)
        
        # Verify integration
        assert len(executor.get_registered_tools()) == 2
        schemas = executor.get_tool_schemas()
        assert len(schemas) == 2
        
        # Test execution
        result = await executor.execute_tool("call_1", "local_calculator", {
            "operation": "add", "a": 10, "b": 20
        })
        assert result["result"] == 30
        
        await remote_protocol.disconnect()
    
    def test_tool_validation_integration(self):
        """Test validation across different tool types."""
        local_tool = LocalTestTool()
        
        # Test local tool validation
        assert local_tool.validate_args({"operation": "add", "a": 1, "b": 2}) is True
        assert local_tool.validate_args({"operation": "invalid", "a": 1, "b": 2}) is False
        assert local_tool.validate_args({"a": 1, "b": 2}) is False  # Missing operation
        
        # Test remote tool validation (when created)
        remote_protocol = MockRemoteProtocol()
        tool_schema = remote_protocol.tools_data[0]
        remote_tool = RemoteTool(tool_schema, remote_protocol, "mock")
        
        # Remote tool should use default validation (always True) unless overridden
        assert remote_tool.validate_args({"number": 42}) is True
        assert remote_tool.validate_args({"number": 42, "format": "currency"}) is True 