# RemGPT - Intelligent Conversation Management System

RemGPT is a sophisticated AI conversation orchestrator that provides intelligent context management, topic drift detection, and self-managing memory through a clean streaming API.

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [API Usage](#api-usage)
- [Algorithm Implementation](#algorithm-implementation)
- [Topic Drift Detection](#topic-drift-detection)
- [Testing](#testing)
- [Development](#development)
- [License](#license)

## Overview

RemGPT implements a novel conversation orchestration algorithm that allows AI assistants to intelligently manage their own context through function calling. The system automatically detects topic changes, preserves important conversation history, and manages memory limitations without manual intervention.

**Core Design Principle**: The orchestrator handles all context management internally and only streams LLM events to the API, creating a clean separation between internal processing and external interface.

## Key Features

### 🧠 Intelligent Context Management
- **Self-Managing Memory**: AI automatically decides when to save or evict topics
- **Topic Preservation**: Important conversations are summarized and retained
- **Adaptive Token Management**: Handles context limits intelligently

### 🔄 Advanced Topic Drift Detection
- **Real-time Detection**: Uses sentence embeddings and Page-Hinkley statistical test
- **High Accuracy**: Comprehensive testing shows 82.9% drift detection rate
- **False Positive Prevention**: Sophisticated algorithms prevent unnecessary triggers

### 🚀 Clean Streaming API
- **Server-Sent Events**: Real-time streaming responses
- **Bearer Token Authentication**: Ready for OAuth/JWT integration  
- **Simple Interface**: Only LLM events exposed to clients

### 🧪 Comprehensive Testing
- **38+ Tests**: Unit and integration tests with real SentenceTransformer models
- **Performance Validated**: <1 second embedding generation
- **Edge Case Coverage**: Handles negative similarities, empty content, and extreme topic changes

## Installation

### Prerequisites
```bash
# Ensure compatible NumPy version
pip install "numpy<2.0"

# Install RemGPT
pip install remgpt
```

### Development Setup
```bash
# Clone repository
git clone https://github.com/yourusername/remgpt.git
cd remgpt

# Install dependencies
pip install -e .
pip install -r requirements-test.txt

# Run tests
pytest tests/
```

## API Usage

### Basic Authentication
```bash
curl -X POST "http://localhost:8000/messages/stream" \
  -H "Authorization: Bearer your_token_here" \
  -H "Content-Type: application/json" \
  -d '{"content": "Hello, how are you today?"}'
```

### JavaScript/TypeScript Client
```javascript
async function sendMessage(content, token) {
  const response = await fetch('http://localhost:8000/messages/stream', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${token}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ content })
  });

  const reader = response.body.getReader();
  const decoder = new TextDecoder();

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    const chunk = decoder.decode(value);
    const lines = chunk.split('\n');
    
    for (const line of lines) {
      if (line.startsWith('data: ')) {
        const eventData = JSON.parse(line.substring(6));
        
        switch (eventData.type) {
          case 'llm_response_chunk':
            console.log('Response:', eventData.data.content);
            break;
          case 'llm_function_call':
            console.log('AI called function:', eventData.data.function_name);
            break;
        }
      }
    }
  }
}
```

### Python Client
```python
import aiohttp
import asyncio
import json

async def stream_message(content, token):
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(
            'http://localhost:8000/messages/stream',
            headers=headers,
            json={'content': content}
        ) as response:
            
            async for line in response.content:
                line = line.decode('utf-8').strip()
                if line.startswith('data: '):
                    event_data = json.loads(line[6:])
                    if event_data['type'] == 'llm_response_chunk':
                        print(event_data['data'].get('content', ''))

# Usage
asyncio.run(stream_message("What is AI?", "your_token"))
```

## Algorithm Implementation

### Message Processing Pipeline

```
User Message → FIFO Queue → Drift Detection → Context Warnings → LLM with Functions → Stream Events
```

#### Key Components:

1. **Internal Processing** (No API Events):
   - Topic drift detection using sentence embeddings
   - Context token usage monitoring (70% threshold)
   - Warning message insertion for LLM guidance
   - System message construction from multiple blocks

2. **LLM Function Calls** (Streamed to API):
   - `save_current_topic()`: Summarizes and preserves current conversation
   - `evict_oldest_topic()`: Removes oldest topic to free memory
   - All function calls visible as streaming events

3. **Event Streaming** (API Surface):
   - `llm_call_start`: Processing begins
   - `llm_function_call`: Context management function called
   - `llm_response_chunk`: Streaming response content
   - `llm_response_complete`: Response finished

### Example Conversation Flow

```
User: "Let's discuss machine learning algorithms"
→ No drift detected (first message)
→ LLM responds about ML algorithms

User: "Actually, let's talk about cooking instead"
→ Topic drift detected internally
→ Warning added: "TOPIC DRIFT DETECTED"
→ LLM sees warning and calls save_current_topic()
→ ML conversation saved as topic
→ LLM responds about cooking

[Many cooking messages...]
→ Context reaches 70% token limit
→ Warning added: "APPROACHING TOKEN LIMIT"
→ LLM calls evict_oldest_topic()
→ ML topic evicted to make space
```

## Topic Drift Detection

### Advanced Statistical Testing

RemGPT uses a sophisticated approach to detect topic changes:

- **Sentence Embeddings**: Real semantic understanding via SentenceTransformer
- **Page-Hinkley Test**: Statistical change detection with configurable sensitivity
- **Cosine Similarity**: Robust similarity calculations between message embeddings

### Performance Metrics

After fixing a critical bug in the Page-Hinkley implementation:

| Scenario | Before Fix | After Fix | Improvement |
|----------|------------|-----------|-------------|
| Similar Topics | 0.0% | 0.0% | ✅ Perfect (no false positives) |
| Gradual Shift | 60.0% | 80.0% | +20% |
| Professional Shift | 0.0% | 100.0% | +100% |
| Sudden Change | 0.0% | 100.0% | +100% |
| Topic Jumps | 0.0% | 100.0% | +100% |

**Overall Improvement**: 8.6% → 82.9% drift detection rate

### Critical Bug Fix

**The Problem**: Original implementation used running mean for deviation calculation:
```python
# BUGGY: Always zero for single samples
deviation = self.mean_estimate - similarity
```

**The Solution**: Use baseline similarity approach:
```python
# FIXED: Non-zero deviation for single samples  
deviation = self.baseline_similarity - similarity
```

This fix enables immediate detection of extreme topic changes.

## Testing

### Comprehensive Test Suite

**38+ Tests Total**:
- **Unit Tests (14)**: Page-Hinkley algorithm, data structures
- **Integration Tests (25)**: Real SentenceTransformer model validation
- **Edge Case Tests**: Single sample drift, negative similarity, zero deviation prevention
- **Performance Tests**: <1 second embedding generation

### Running Tests

```bash
# All tests
pytest tests/test_topic_detection.py

# Fast tests only (no model loading)
pytest tests/test_topic_detection.py -m "not slow"

# Using test runner
python scripts/run_real_tests.py --verbose

# With coverage
python scripts/run_real_tests.py --coverage
```

### Test Categories

- `@pytest.mark.slow`: Tests requiring model download (~90MB)
- `@pytest.mark.edge_case`: Boundary condition tests
- `@pytest.mark.bug_prevention`: Regression prevention tests
- `@pytest.mark.topic_detection`: All drift detection tests

### Performance Validation

- **Model Loading**: ~10 seconds (all-MiniLM-L6-v2)
- **Embedding Speed**: <1 second per message
- **Topic Clustering**: Clear semantic separation across domains
- **Memory Usage**: Efficient with configurable window sizes

### Topic Drift Demo

Experience the topic detection system in action:

```bash
# Interactive demonstration
python scripts/topic_drift_demo.py
```

Features multiple conversation scenarios:
- Similar Topics (no false positives)
- Gradual topic transitions
- Sudden topic changes
- Topic jumps between unrelated domains

## Development

### Project Structure

```
remgpt/
├── detection/           # Topic drift detection
│   ├── detector.py     # Main TopicDriftDetector
│   └── page_hinkley_test.py  # Statistical test implementation
├── context/            # Context management
│   ├── manager.py      # LLM context manager
│   ├── llm_context.py  # Context data structures
│   └── blocks/         # Context building blocks
├── orchestration/      # Main orchestrator
│   └── orchestrator.py # Conversation orchestration
├── types/              # Data types and models
└── scripts/            # Utility scripts
    ├── run_real_tests.py    # Test runner
    ├── topic_drift_demo.py  # Interactive demo
    └── build_and_upload.py  # Package deployment
```

### Configuration

Key parameters for topic drift detection:

```python
detector = TopicDriftDetector(
    model_name="all-MiniLM-L6-v2",  # SentenceTransformer model
    similarity_threshold=0.6,        # Topic similarity threshold
    drift_threshold=0.25,           # Page-Hinkley drift sensitivity
    alpha=0.1,                      # Statistical significance level
    window_size=5                   # Recent messages to consider
)
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Add comprehensive tests
4. Ensure all tests pass: `pytest tests/`
5. Submit a pull request

### Requirements

**Core Dependencies**:
- `sentence-transformers`: Semantic embeddings
- `numpy`: Numerical computing
- `fastapi`: Web framework
- `pydantic`: Data validation

**Test Dependencies**:
- `pytest`: Test framework
- `pytest-asyncio`: Async test support
- `pytest-cov`: Coverage reporting

### First Time Setup

The first test run will download the SentenceTransformer model:

```bash
🚀 Running topic detection tests...
📋 This will download the SentenceTransformer model (~90MB) if not already cached
```

Model is cached locally for subsequent runs.

### Development Workflow

```bash
# Quick validation (no model download)
python scripts/run_real_tests.py --fast

# Full validation before commit
python scripts/run_real_tests.py --verbose

# Demo topic detection
python scripts/topic_drift_demo.py

# Performance analysis
python scripts/run_real_tests.py --benchmark
```

## Authentication

### Current (Development)
```bash
# Any Bearer token works in development
Authorization: Bearer dev_token_123
```

### Future (OAuth/JWT)
```bash
# JWT token from OAuth provider
Authorization: Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9...
```

User identity extracted from JWT claims (`sub`, `email`, `preferred_username`).

## Error Handling

### Missing Authentication
```json
{
  "detail": "Missing Authorization header. Expected: Bearer <token>"
}
```

### Invalid Token Format
```json
{
  "detail": "Invalid authorization header format. Expected: Bearer <token>"
}
```

## License

MIT License - see LICENSE file for details.

---

**RemGPT** - Building intelligent conversation systems with advanced topic detection and self-managing context. 