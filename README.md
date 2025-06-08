# RemGPT - Intelligent Conversation Management System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

RemGPT is a sophisticated AI conversation orchestrator with intelligent context management, topic drift detection, and self-managing memory through a clean streaming API.

## 🌟 Key Features

- **Self-Managing Memory**: AI automatically decides when to save or evict topics
- **Multi-Provider LLM Support**: OpenAI, Claude, Gemini with seamless switching
- **Remote Tool Integration**: MCP and Agent-to-Agent protocols
- **Topic Drift Detection**: 82.9% accuracy using statistical analysis
- **Clean Streaming API**: Server-sent events with Bearer token authentication

## 🚀 Quick Start

### Installation
```bash
pip install "numpy<2.0"  # Ensure compatibility
pip install remgpt
```

### Basic Usage
```python
from remgpt import LLMClientFactory, ToolExecutor, ConversationOrchestrator
from remgpt.tools import BaseTool

# Create LLM client
factory = LLMClientFactory()
client = factory.create_client(
    provider="openai",
    model_name="gpt-4",
    api_key="your-api-key"
)

# Set up tools
class CalculatorTool(BaseTool):
    def __init__(self):
        super().__init__("calculator", "Perform arithmetic calculations")
    
    async def execute(self, operation: str, a: float, b: float) -> dict:
        return {"result": a + b if operation == "add" else a * b}
    
    def get_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "Perform arithmetic calculations",
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

executor = ToolExecutor()
executor.register_tool(CalculatorTool())

# Process messages with streaming
messages = [{"role": "user", "content": "Calculate 2 + 3"}]

async for event in client.generate_stream(messages, tools=executor.get_tool_schemas()):
    if event.type == "TEXT_MESSAGE_CONTENT":
        print(event.content)
```

### Full Orchestrator with Remote Tools
```python
from remgpt import create_orchestrator, create_context_manager

# Create context manager
context_manager = create_context_manager(
    max_tokens=4000,
    system_instructions="You are a helpful assistant."
)

# Create orchestrator with remote tools
orchestrator = await create_orchestrator(
    context_manager=context_manager,
    mcp_servers=["uvx weather-mcp@latest stdio"],
    a2a_agents=["http://localhost:8000"]
)

# Process messages
async for event in orchestrator.process_message({"role": "user", "content": "Hello!"}):
    if event.type == "TEXT_MESSAGE_CONTENT":
        print(event.content)
```

## 🌐 API Server

### Start Server
```bash
uvicorn remgpt.api:app --host 0.0.0.0 --port 8000
```

### REST API
```bash
curl -X POST "http://localhost:8000/messages/stream" \
  -H "Authorization: Bearer your_token_here" \
  -H "Content-Type: application/json" \
  -d '{"content": "Hello, how are you?"}'
```

## 🏗️ Architecture

### Core Modules
```
remgpt/
├── api.py              # FastAPI web interface
├── context/            # Context management system
├── llm/                # LLM client abstractions (OpenAI, Claude, Gemini)
├── tools/              # Tool execution system
│   └── remote/         # Remote tool protocols (MCP, A2A)
├── orchestration/      # Conversation orchestration
├── detection/          # Topic drift detection
├── summarization/      # Topic summarization
└── storage/            # Vector database abstractions
```

### Design Principles
- **Single Responsibility**: Each class has its own file
- **Protocol Abstraction**: Unified interfaces for different providers
- **Graceful Degradation**: Works without optional dependencies
- **Factory Patterns**: Easy configuration and instantiation

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run specific test categories
pytest tests/test_llm_client.py          # LLM client tests
pytest tests/test_tools/                 # Tool system tests
pytest tests/test_context/               # Context management tests
```

## 📦 Remote Tools

### MCP (Model Context Protocol)
```python
from remgpt.tools.remote import MCPProtocol

# Connect to MCP server
protocol = MCPProtocol()
await protocol.connect("uvx weather-mcp@latest stdio")
```

### Agent-to-Agent (A2A)
```python
from remgpt.tools.remote import A2AProtocol

# Connect to A2A agent
protocol = A2AProtocol("http://localhost:8000")
await protocol.call_tool("send_task", {"text": "Hello, agent!"})
```

## 🔧 Development

### Setup
```bash
git clone https://github.com/yourusername/remgpt.git
cd remgpt
pip install -e .
pip install -r requirements-test.txt
```

### Key Development Features
- **40+ Tests**: Unit and integration tests with real models
- **Performance Validated**: <1 second embedding generation
- **Modular Architecture**: Easy to extend and maintain
- **Clean Imports**: Conditional exports based on availability

## 📄 License

MIT License - see LICENSE file for details. 