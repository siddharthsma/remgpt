"""
Unit tests for BaseTool abstract class.
"""

import pytest
from abc import ABC
from remgpt.tools.base import BaseTool


class ConcreteTool(BaseTool):
    """Concrete implementation for testing."""
    
    def __init__(self, name="test_tool", description="Test tool description"):
        super().__init__(name, description)
        self.execution_count = 0
    
    async def execute(self, **kwargs):
        """Test implementation of execute method."""
        self.execution_count += 1
        return {"result": "test_result", "args": kwargs, "count": self.execution_count}
    
    def get_schema(self):
        """Test implementation of get_schema method."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        }


class TestBaseTool:
    """Test cases for BaseTool abstract class."""
    
    def test_abstract_class(self):
        """Test that BaseTool is abstract and cannot be instantiated."""
        with pytest.raises(TypeError):
            BaseTool("test", "description")
    
    def test_concrete_implementation(self):
        """Test that concrete implementation can be instantiated."""
        tool = ConcreteTool("test_tool", "Test description")
        assert tool.name == "test_tool"
        assert tool.description == "Test description"
    
    def test_name_property(self):
        """Test name property."""
        tool = ConcreteTool("custom_name", "Description")
        assert tool.name == "custom_name"
    
    def test_description_property(self):
        """Test description property."""
        tool = ConcreteTool("tool", "Custom description")
        assert tool.description == "Custom description"
    
    @pytest.mark.asyncio
    async def test_execute_method(self):
        """Test execute method implementation."""
        tool = ConcreteTool()
        
        # Test execution without arguments
        result = await tool.execute()
        assert result["result"] == "test_result"
        assert result["args"] == {}
        assert result["count"] == 1
        
        # Test execution with arguments
        result = await tool.execute(param1="value1", param2="value2")
        assert result["result"] == "test_result"
        assert result["args"] == {"param1": "value1", "param2": "value2"}
        assert result["count"] == 2
    
    def test_get_schema_default(self):
        """Test default get_schema implementation."""
        tool = ConcreteTool("test_tool", "Test description")
        schema = tool.get_schema()
        
        expected_schema = {
            "type": "function",
            "function": {
                "name": "test_tool",
                "description": "Test description",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        }
        
        assert schema == expected_schema
    
    def test_validate_args_default(self):
        """Test default validate_args implementation."""
        tool = ConcreteTool()
        
        # Default implementation should return True for any arguments
        assert tool.validate_args({}) is True
        assert tool.validate_args({"param": "value"}) is True
        assert tool.validate_args({"multiple": "params", "test": 123}) is True


class CustomSchemaTool(BaseTool):
    """Tool with custom schema for testing."""
    
    def __init__(self):
        super().__init__("custom_schema_tool", "Tool with custom schema")
    
    async def execute(self, required_param, optional_param=None):
        return {"required": required_param, "optional": optional_param}
    
    def get_schema(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "required_param": {"type": "string", "description": "Required parameter"},
                        "optional_param": {"type": "string", "description": "Optional parameter"}
                    },
                    "required": ["required_param"]
                }
            }
        }
    
    def validate_args(self, args):
        """Custom validation."""
        if "required_param" not in args:
            return False
        if not isinstance(args["required_param"], str):
            return False
        return True


class TestCustomSchemaTool:
    """Test cases for custom schema implementation."""
    
    def test_custom_schema(self):
        """Test custom schema implementation."""
        tool = CustomSchemaTool()
        schema = tool.get_schema()
        
        assert schema["function"]["name"] == "custom_schema_tool"
        assert "required_param" in schema["function"]["parameters"]["properties"]
        assert "optional_param" in schema["function"]["parameters"]["properties"]
        assert schema["function"]["parameters"]["required"] == ["required_param"]
    
    def test_custom_validation_valid(self):
        """Test custom validation with valid arguments."""
        tool = CustomSchemaTool()
        
        # Valid arguments
        assert tool.validate_args({"required_param": "test"}) is True
        assert tool.validate_args({"required_param": "test", "optional_param": "optional"}) is True
    
    def test_custom_validation_invalid(self):
        """Test custom validation with invalid arguments."""
        tool = CustomSchemaTool()
        
        # Missing required parameter
        assert tool.validate_args({}) is False
        assert tool.validate_args({"optional_param": "optional"}) is False
        
        # Wrong type for required parameter
        assert tool.validate_args({"required_param": 123}) is False
    
    @pytest.mark.asyncio
    async def test_custom_execute(self):
        """Test custom execute implementation."""
        tool = CustomSchemaTool()
        
        # Execute with required parameter only
        result = await tool.execute(required_param="test_value")
        assert result["required"] == "test_value"
        assert result["optional"] is None
        
        # Execute with both parameters
        result = await tool.execute(required_param="test_value", optional_param="optional_value")
        assert result["required"] == "test_value"
        assert result["optional"] == "optional_value" 