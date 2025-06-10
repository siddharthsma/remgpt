# RemGPT Test Suite

This directory contains the complete test suite for RemGPT, organized by test type and component area.

## 📁 Directory Structure

```
tests/
├── unit/                          # Unit tests for individual components
│   ├── api/                       # API endpoint and authentication tests
│   ├── llm/                       # LLM client and provider tests
│   ├── context/                   # Context management and topic detection tests
│   └── tools/                     # Tool execution and management tests
├── integration/                   # Integration tests for system behavior
├── conftest.py                    # Global pytest configuration
└── README.md                      # This file
```

## 🧪 Test Categories

### Unit Tests (`tests/unit/`)

Unit tests focus on testing individual components in isolation with mocked dependencies.

#### API Tests (`tests/unit/api/`)
- **`test_api.py`**: FastAPI endpoint testing, SSE formatting, error handling
- **`test_authentication.py`**: Authentication mechanism validation, token handling

#### LLM Tests (`tests/unit/llm/`)
- **`test_llm_client.py`**: OpenAI client functionality, streaming, error handling

#### Context Tests (`tests/unit/context/`)
- **`test_summarization.py`**: Topic summarization and content processing
- **`test_topic_detection.py`**: Topic drift detection algorithms and thresholds

#### Tools Tests (`tests/unit/tools/`)
- **`test_base.py`**: Base tool functionality and interfaces
- **`test_executor.py`**: Tool execution coordination and lifecycle
- **`test_integration.py`**: Tool system integration
- **`test_remote/`**: Remote tool protocols (MCP, A2A)
  - **`test_mcp.py`**: Model Context Protocol implementation
  - **`test_a2a.py`**: Agent-to-Agent communication
  - **`test_manager.py`**: Remote tool management
  - **`test_tool.py`**: Remote tool execution
  - **`test_init.py`**: Remote tool initialization
  - **`test_remote_base.py`**: Base remote tool functionality

### Integration Tests (`tests/integration/`)

Integration tests validate end-to-end system behavior with real components and API calls.

#### Core Integration Tests
- **`test_always_available_tools.py`** ✅ **CURRENT**: Tests that tools are available immediately without drift detection
- **`test_explicit_tool_calling_final.py`** ✅ **CURRENT**: Comprehensive tool calling validation
- **`test_end_to_end_system.py`**: Complete system integration testing
- **`test_comprehensive_system_integration.py`**: System behavior validation based on demo analysis
- **`test_integration.py`**: General integration test cases

#### Test Infrastructure
- **`conftest.py`**: Integration test fixtures and configuration
- **`run_tests.py`**: Test runner with environment setup
- **`.env`**: Environment configuration for integration tests (not in repo)
- **`README.md`**: Integration test documentation

## 🚀 Running Tests

### Prerequisites
```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables for integration tests
cp tests/integration/.env.example tests/integration/.env
# Edit .env file with your OpenAI API key
```

### Run All Tests
```bash
# Run the complete test suite
pytest

# Run with verbose output
pytest -v

# Run with coverage
pytest --cov=remgpt
```

### Run Specific Test Categories
```bash
# Unit tests only
pytest tests/unit/

# Integration tests only  
pytest tests/integration/

# Specific component tests
pytest tests/unit/api/
pytest tests/unit/tools/
```

### Run Individual Test Files
```bash
# Test always-available tools (key functionality)
pytest tests/integration/test_always_available_tools.py -v

# Test tool calling validation
pytest tests/integration/test_explicit_tool_calling_final.py -v

# Test API endpoints
pytest tests/unit/api/test_api.py -v
```

### Run Tests by Markers
```bash
# Integration tests only
pytest -m integration

# Authentication tests only
pytest -m auth

# Tool-related tests only
pytest -m tools
```

## 📋 Test Markers

- `@pytest.mark.integration`: Integration tests requiring API keys
- `@pytest.mark.auth`: Authentication-related tests
- `@pytest.mark.streaming`: Streaming functionality tests
- `@pytest.mark.tools`: Tool execution and management tests
- `@pytest.mark.asyncio`: Asynchronous test functions

## 🔧 Key Test Files

### Critical Tests (Must Pass)
1. **`test_always_available_tools.py`**: Validates that tools work without drift detection
2. **`test_explicit_tool_calling_final.py`**: Comprehensive tool calling validation
3. **`test_api.py`**: Core API functionality
4. **`test_executor.py`**: Tool execution system

### Development Tests  
- **`test_end_to_end_system.py`**: Full system integration
- **`test_comprehensive_system_integration.py`**: System behavior analysis

## 🛠️ Adding New Tests

### Unit Tests
1. Create test file in appropriate `tests/unit/[component]/` directory
2. Use mocked dependencies for isolation
3. Follow naming convention: `test_[component_name].py`

### Integration Tests
1. Add test file to `tests/integration/`
2. Use real components with API keys
3. Add `@pytest.mark.integration` marker
4. Document any external dependencies

### Test Structure Template
```python
"""
Test description and purpose.
"""

import pytest
from remgpt.component import ComponentToTest

class TestComponentName:
    """Test class for specific component."""
    
    @pytest.fixture
    def setup_component(self):
        """Fixture for test setup."""
        return ComponentToTest()
    
    def test_basic_functionality(self, setup_component):
        """Test basic component functionality."""
        # Test implementation
        assert setup_component.method() == expected_result
    
    @pytest.mark.asyncio
    async def test_async_functionality(self, setup_component):
        """Test asynchronous component functionality."""
        result = await setup_component.async_method()
        assert result is not None
```

## 🔍 Troubleshooting

### Common Issues

1. **Missing API Key**: Integration tests require `OPENAI_API_KEY` in `tests/integration/.env`
2. **Import Errors**: Ensure RemGPT is installed in development mode: `pip install -e .`
3. **Async Test Failures**: Use `@pytest.mark.asyncio` for async test functions
4. **Tool Registration**: Ensure tools are properly registered in test fixtures

### Environment Setup
```bash
# For development
pip install -e .
pip install pytest pytest-asyncio pytest-cov

# For integration tests
echo "OPENAI_API_KEY=your_key_here" > tests/integration/.env
```

## 📊 Test Metrics

- **Unit Tests**: ~15 files, focuses on component isolation
- **Integration Tests**: ~6 files, validates end-to-end behavior
- **Coverage Target**: 85%+ for core functionality
- **Performance**: Integration tests should complete within 5 minutes

## 🎯 Current Focus Areas

1. **Tool Availability**: Ensuring tools work without drift detection
2. **Integration Stability**: Reliable end-to-end functionality
3. **API Robustness**: Comprehensive endpoint testing
4. **Remote Tools**: MCP and A2A protocol implementation

---

*This test suite ensures RemGPT's reliability and functionality across all components and integration scenarios.* 