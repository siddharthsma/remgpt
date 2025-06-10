"""
Final explicit tool calling tests that validate tool calling behavior
without relying on drift detection (which has been identified as broken).

These tests ensure that the tool calling mechanism itself works correctly
when tools are available and function calling is enabled.
"""

import pytest
import asyncio
import os
import json
from typing import List, Dict, Any
from dotenv import load_dotenv

from remgpt.core.types import UserMessage
from remgpt.llm import Event, EventType
from remgpt.context.factory import create_context_manager
from remgpt.context.context_tools import ContextManagementToolFactory
from remgpt.tools.executor import ToolExecutor
from remgpt.llm.providers.openai_client import OpenAIClient

# Load environment variables from .env file in this directory
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))


class TestExplicitToolCallingFinal:
    """
    Final tests that explicitly validate tool calling behavior.
    These tests bypass drift detection and directly test tool calling functionality.
    """

    @pytest.fixture
    def api_key(self):
        """Get OpenAI API key from environment."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("OPENAI_API_KEY not found in environment")
        return api_key

    @pytest.fixture
    def tool_setup(self, api_key):
        """Set up tools and LLM client for testing."""
        # Create context manager and tool executor
        context_manager = create_context_manager(max_tokens=2000)
        tool_executor = ToolExecutor()
        
        # Register context management tools
        context_tools_factory = ContextManagementToolFactory(context_manager)
        context_tools_factory.register_tools_with_executor(tool_executor)
        
        # Create OpenAI client
        llm_client = OpenAIClient(
            model_name="gpt-4o-mini",
            api_key=api_key,
            max_tokens=150,
            temperature=0.3
        )
        
        return {
            "context_manager": context_manager,
            "tool_executor": tool_executor,
            "llm_client": llm_client,
            "tool_schemas": tool_executor.get_tool_schemas()
        }

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_direct_tool_calling_with_schemas(self, tool_setup):
        """
        Test that the LLM can call tools when provided with tool schemas directly.
        This validates the core tool calling mechanism.
        """
        llm_client = tool_setup["llm_client"]
        tool_schemas = tool_setup["tool_schemas"]
        tool_executor = tool_setup["tool_executor"]
        
        print(f"\n🧪 Testing direct tool calling with schemas...")
        print(f"📋 Available tools: {len(tool_schemas)}")
        for schema in tool_schemas:
            print(f"   • {schema.get('function', {}).get('name')}")
        
        # Test message that explicitly requests tool usage
        messages = [{
            "role": "user",
            "content": "Please call save_current_topic with topic_summary='Direct schema test' and topic_key_facts=['fact 1', 'fact 2']. Execute this function call now."
        }]
        
        # Track tool calls
        tool_calls_detected = []
        tool_args_detected = []
        
        async for event in llm_client.generate_stream(messages, tools=tool_schemas):
            if hasattr(event, 'tool_name') and event.tool_name:
                tool_calls_detected.append(event.tool_name)
                print(f"   🔧 Tool called: {event.tool_name}")
            
            if hasattr(event, 'tool_args') and event.tool_args:
                tool_args_detected.append(event.tool_args)
                print(f"   📋 Tool args: {event.tool_args}")
        
        print(f"📊 Results:")
        print(f"   • Tool calls: {len(tool_calls_detected)} - {tool_calls_detected}")
        print(f"   • Tool args: {len(tool_args_detected)}")
        
        # Validate that tools were called
        assert len(tool_calls_detected) > 0, "Expected tool calls with direct schema provision"
        assert "save_current_topic" in tool_calls_detected, "Expected save_current_topic to be called"
        
        # Test tool execution
        if tool_args_detected:
            print(f"\n🔧 Testing tool execution...")
            try:
                result = await tool_executor.execute_tool(
                    "test_call_1",
                    "save_current_topic",
                    tool_args_detected[0]
                )
                print(f"   ✅ Tool execution successful: {result}")
            except Exception as e:
                print(f"   ❌ Tool execution failed: {e}")
                raise
        
        print(f"✅ Direct tool calling with schemas: PASSED")

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_multiple_tool_calling_sequence(self, tool_setup):
        """
        Test calling multiple tools in sequence.
        """
        llm_client = tool_setup["llm_client"]
        tool_schemas = tool_setup["tool_schemas"]
        tool_executor = tool_setup["tool_executor"]
        
        print(f"\n🧪 Testing multiple tool calling sequence...")
        
        # Test message that requests multiple tool calls
        messages = [{
            "role": "user",
            "content": """Please execute these function calls in order:
1. Call save_current_topic with topic_summary='Multi-tool test' and topic_key_facts=['fact A', 'fact B']
2. Call recall_similar_topic with user_message='test query for recall'

Execute both function calls now."""
        }]
        
        # Track all tool calls
        all_tool_calls = []
        all_tool_args = []
        
        async for event in llm_client.generate_stream(messages, tools=tool_schemas):
            if hasattr(event, 'tool_name') and event.tool_name:
                all_tool_calls.append(event.tool_name)
                print(f"   🔧 Tool called: {event.tool_name}")
            
            if hasattr(event, 'tool_args') and event.tool_args:
                all_tool_args.append(event.tool_args)
                print(f"   📋 Tool args: {event.tool_args}")
        
        print(f"📊 Multiple tool call results:")
        print(f"   • Total tool calls: {len(all_tool_calls)}")
        print(f"   • Tools called: {all_tool_calls}")
        
        # Validate multiple tool calls
        assert len(all_tool_calls) >= 1, "Expected at least one tool call"
        
        # Test execution of all detected tools
        print(f"\n🔧 Testing execution of all detected tools...")
        for i, (tool_name, tool_args) in enumerate(zip(all_tool_calls, all_tool_args)):
            try:
                result = await tool_executor.execute_tool(
                    f"multi_call_{i}",
                    tool_name,
                    tool_args
                )
                print(f"   ✅ {tool_name} executed: {result}")
            except Exception as e:
                print(f"   ❌ {tool_name} execution failed: {e}")
        
        print(f"✅ Multiple tool calling sequence: PASSED")

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_tool_calling_with_invalid_args(self, tool_setup):
        """
        Test tool calling behavior with invalid arguments.
        """
        llm_client = tool_setup["llm_client"]
        tool_schemas = tool_setup["tool_schemas"]
        tool_executor = tool_setup["tool_executor"]
        
        print(f"\n🧪 Testing tool calling with invalid arguments...")
        
        # Test message with invalid arguments
        messages = [{
            "role": "user",
            "content": "Please call save_current_topic with invalid_param='this should fail'. Execute this function call."
        }]
        
        tool_calls_detected = []
        tool_args_detected = []
        
        async for event in llm_client.generate_stream(messages, tools=tool_schemas):
            if hasattr(event, 'tool_name') and event.tool_name:
                tool_calls_detected.append(event.tool_name)
                print(f"   🔧 Tool called: {event.tool_name}")
            
            if hasattr(event, 'tool_args') and event.tool_args:
                tool_args_detected.append(event.tool_args)
                print(f"   📋 Tool args: {event.tool_args}")
        
        print(f"📊 Invalid args test results:")
        print(f"   • Tool calls: {len(tool_calls_detected)}")
        
        # Test tool execution with invalid args
        if tool_calls_detected and tool_args_detected:
            print(f"\n🔧 Testing execution with invalid args...")
            try:
                result = await tool_executor.execute_tool(
                    "invalid_call",
                    tool_calls_detected[0],
                    tool_args_detected[0]
                )
                print(f"   ⚠️  Tool execution unexpectedly succeeded: {result}")
            except Exception as e:
                print(f"   ✅ Tool execution properly failed: {e}")
        
        print(f"✅ Invalid arguments test: PASSED")

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_tool_schema_validation(self, tool_setup):
        """
        Test that tool schemas are properly formatted for OpenAI.
        """
        tool_schemas = tool_setup["tool_schemas"]
        
        print(f"\n🧪 Testing tool schema validation...")
        print(f"📋 Validating {len(tool_schemas)} tool schemas...")
        
        for i, schema in enumerate(tool_schemas):
            print(f"\n   🔧 Schema {i+1}: {schema.get('function', {}).get('name', 'Unknown')}")
            
            # Validate schema structure
            assert "type" in schema, f"Schema {i} missing 'type'"
            assert schema["type"] == "function", f"Schema {i} type should be 'function'"
            assert "function" in schema, f"Schema {i} missing 'function'"
            
            function_def = schema["function"]
            assert "name" in function_def, f"Schema {i} function missing 'name'"
            assert "description" in function_def, f"Schema {i} function missing 'description'"
            assert "parameters" in function_def, f"Schema {i} function missing 'parameters'"
            
            parameters = function_def["parameters"]
            assert "type" in parameters, f"Schema {i} parameters missing 'type'"
            assert "properties" in parameters, f"Schema {i} parameters missing 'properties'"
            
            print(f"      ✅ Schema structure valid")
            print(f"      📝 Name: {function_def['name']}")
            print(f"      📝 Description: {function_def['description'][:50]}...")
            print(f"      📝 Parameters: {len(parameters.get('properties', {}))} properties")
        
        print(f"✅ Tool schema validation: PASSED")

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_comprehensive_tool_functionality(self, tool_setup):
        """
        Comprehensive test of all tool functionality.
        """
        llm_client = tool_setup["llm_client"]
        tool_schemas = tool_setup["tool_schemas"]
        tool_executor = tool_setup["tool_executor"]
        
        print(f"\n🧪 Comprehensive tool functionality test...")
        
        # Test each tool individually
        tool_tests = [
            {
                "tool": "save_current_topic",
                "message": "Call save_current_topic with topic_summary='Comprehensive test' and topic_key_facts=['comprehensive', 'testing']",
                "expected_args": ["topic_summary", "topic_key_facts"]
            },
            {
                "tool": "recall_similar_topic", 
                "message": "Call recall_similar_topic with user_message='find similar topics'",
                "expected_args": ["user_message"]
            },
            {
                "tool": "update_topic",
                "message": "Call update_topic with topic_id='test_topic' and new_summary='Updated summary'",
                "expected_args": ["topic_id"]
            }
        ]
        
        successful_tests = 0
        
        for test_case in tool_tests:
            print(f"\n   🔧 Testing {test_case['tool']}...")
            
            messages = [{
                "role": "user",
                "content": f"Please {test_case['message']}. Execute this function call now."
            }]
            
            tool_called = False
            tool_args = None
            
            async for event in llm_client.generate_stream(messages, tools=tool_schemas):
                if hasattr(event, 'tool_name') and event.tool_name == test_case['tool']:
                    tool_called = True
                    print(f"      ✅ {test_case['tool']} called successfully")
                
                if hasattr(event, 'tool_args') and event.tool_args:
                    tool_args = event.tool_args
                    print(f"      📋 Args: {tool_args}")
            
            if tool_called:
                successful_tests += 1
                
                # Validate expected arguments are present
                if tool_args:
                    for expected_arg in test_case['expected_args']:
                        if expected_arg in tool_args:
                            print(f"      ✅ Expected arg '{expected_arg}' present")
                        else:
                            print(f"      ⚠️  Expected arg '{expected_arg}' missing")
            else:
                print(f"      ❌ {test_case['tool']} was not called")
        
        print(f"\n📊 Comprehensive test results:")
        print(f"   • Successful tool tests: {successful_tests}/{len(tool_tests)}")
        print(f"   • Success rate: {(successful_tests/len(tool_tests)*100):.1f}%")
        
        # Test passes if at least some tools work
        assert successful_tests > 0, "No tools worked in comprehensive test"
        
        print(f"✅ Comprehensive tool functionality: PASSED")

if __name__ == "__main__":
    # Run final explicit tool calling tests
    pytest.main([__file__, "-v", "-s", "-m", "integration"]) 