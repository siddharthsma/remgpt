# Scripts Directory

This directory contains utility scripts for development, testing, and deployment of RemGPT.

## Scripts Overview

### 🧪 Testing & Development

#### `run_real_tests.py`
**Purpose**: Convenient test runner for topic detection tests with real SentenceTransformer models.

```bash
# Run all topic detection tests with verbose output
python scripts/run_real_tests.py --verbose

# Run only fast tests (no model loading)
python scripts/run_real_tests.py --fast

# Run with coverage reporting
python scripts/run_real_tests.py --coverage

# Run performance benchmarks
python scripts/run_real_tests.py --benchmark
```

**Features**:
- Handles SentenceTransformer model download (~90MB first run)
- Provides convenient CLI options for different test scenarios
- Includes performance timing and coverage reporting
- Filters test categories (fast vs slow tests)

#### `topic_drift_demo.py`
**Purpose**: Interactive demonstration of topic drift detection capabilities.

```bash
# Run the demo
python scripts/topic_drift_demo.py
```

**Features**:
- Tests multiple conversation scenarios (Similar Topics, Topic Jumps, etc.)
- Shows real-time drift detection with similarity scores
- Demonstrates Page-Hinkley statistical test in action
- Performance metrics and detailed analysis
- Uses real SentenceTransformer embeddings

**Demo Scenarios**:
- **Similar Topics**: Programming-related messages (should not drift)
- **Gradual Shift**: Slow topic transition testing
- **Professional Shift**: Career change conversation
- **Sudden Change**: Abrupt topic switching
- **Topic Jumps**: Multiple unrelated topics
- **Related Topics**: AI to Machine Learning transition
- **Topic Returns**: Returning to previous topics

### 🚀 Deployment & Distribution

#### `build_and_upload.py`
**Purpose**: Automated package building and PyPI upload script.

```bash
# Build and upload to PyPI
python scripts/build_and_upload.py
```

**Features**:
- Cleans previous builds
- Builds source distribution and wheel
- Uploads to PyPI using twine
- Handles authentication and error checking

**Prerequisites**:
- PyPI account with API token
- `twine` installed (`pip install twine`)
- Proper `pyproject.toml` configuration

## Usage Examples

### Development Workflow
```bash
# Test changes quickly (no model download)
python scripts/run_real_tests.py --fast

# Full validation before commit
python scripts/run_real_tests.py --verbose

# Demo topic detection capabilities
python scripts/topic_drift_demo.py

# Performance analysis
python scripts/run_real_tests.py --benchmark
```

### Release Workflow
```bash
# Full test suite with coverage
python scripts/run_real_tests.py --coverage

# Build and upload new version
python scripts/build_and_upload.py
```

## Script Dependencies

### Common Requirements
```bash
pip install pytest sentence-transformers numpy torch
```

### Build Script Requirements
```bash
pip install build twine
```

### Environment Variables

For `build_and_upload.py`:
```bash
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=your_pypi_api_token
```

## Development Notes

- **Model Caching**: SentenceTransformer models are cached in `~/.cache/huggingface/`
- **Performance**: First model download takes ~2-3 minutes, subsequent runs are fast
- **Testing**: Use `--fast` flag during development to skip slow model tests
- **Coverage**: HTML coverage reports generated in `htmlcov/` directory

## Troubleshooting

### NumPy Compatibility
```bash
# Fix version conflicts
pip install "numpy<2.0"
```

### Model Download Issues
```bash
# Clear model cache
rm -rf ~/.cache/huggingface/transformers/
```

### PyPI Upload Issues
```bash
# Check PyPI credentials
twine check dist/*
``` 