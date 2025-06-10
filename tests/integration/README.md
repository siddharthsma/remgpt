# Integration Tests

These are end-to-end integration tests that use **real OpenAI API calls** to verify complete system behavior.

## Setup

1. **Create a `.env` file** in this directory (`tests/integration/.env`) with your OpenAI API key:
   ```bash
   OPENAI_API_KEY=your_actual_openai_api_key_here
   ```

2. **Install dependencies** (if not already installed):
   ```bash
   pip install python-dotenv pytest-asyncio
   ```

## Running Tests

### Run all integration tests:
```bash
pytest tests/integration/ -v -m integration
```

### Run specific test:
```bash
pytest tests/integration/test_end_to_end_system.py::TestEndToEndSystem::test_real_system_statistics_tracking -v -s
```

### Run with detailed output:
```bash
pytest tests/integration/ -v -s -m integration
```

## Test Coverage

These tests verify:

- ✅ **Real System Statistics**: Token counting, message processing, topic creation with actual API calls
- ✅ **Real Drift Detection**: Sentence transformer embeddings and actual similarity calculations  
- ✅ **Real Tool Calling**: Context management tools triggered by actual drift detection
- ✅ **Real Context Management**: Token limits and context eviction with real token usage

## Test Features

- **No Mocks**: Uses real OpenAI client, real sentence transformers, real token counting
- **Exact Demo Behavior**: Tests the same sequences from `comprehensive_memory_demo.py`
- **Statistics Verification**: Ensures statistics are tracked correctly (not showing 0)
- **Sensitivity Validation**: Confirms drift detection improvements work with real embeddings

## Cost Considerations

These tests make real OpenAI API calls. Estimated cost per full test run:
- ~10-15 API calls with `gpt-4o-mini`
- ~1500-2000 tokens total
- Cost: ~$0.01-0.02 USD per test run

## Environment Variables

- `OPENAI_API_KEY`: Required. Your OpenAI API key
- `LOG_LEVEL`: Optional. Set to `DEBUG` for detailed logging

## Notes

- Tests use `gpt-4o-mini` with low temperature (0.3) for predictable responses
- Max tokens per response limited to 150 to control costs
- Tests may take 30-60 seconds due to real API calls
- Some drift detection behavior may vary slightly due to real LLM responses 