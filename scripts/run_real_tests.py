#!/usr/bin/env python3
"""
Test runner for topic detection tests with real SentenceTransformer model.

This script makes it easy to run the complete topic detection test suite
that uses the actual SentenceTransformer model.
"""

import subprocess
import sys
import argparse
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Run topic detection tests with real SentenceTransformer model")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--coverage", "-c", action="store_true", help="Run with coverage")
    parser.add_argument("--benchmark", "-b", action="store_true", help="Run performance benchmark tests only")
    parser.add_argument("--fast", "-f", action="store_true", help="Run only fast tests (skip slow model tests)")
    args = parser.parse_args()
    
    # Base command
    cmd = ["python", "-m", "pytest", "tests/test_topic_detection.py"]
    
    # Add markers based on options
    if args.fast:
        cmd.extend(["-m", "not slow"])
    elif args.benchmark:
        cmd.extend(["-m", "slow", "-k", "performance"])
    else:
        # Run all tests by default
        pass
    
    # Add options
    if args.verbose:
        cmd.append("-v")
    
    if args.coverage:
        cmd.extend(["--cov=remgpt.detection", "--cov-report=term-missing"])
    
    # Add other useful options
    cmd.extend([
        "--tb=short",
        "-s",  # Don't capture output (so we can see print statements)
        "--durations=10"  # Show slowest tests
    ])
    
    print("🚀 Running topic detection tests...")
    if not args.fast:
        print("📋 This will download the SentenceTransformer model (~90MB) if not already cached")
    print(f"🔧 Command: {' '.join(cmd)}")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except KeyboardInterrupt:
        print("\n❌ Tests interrupted by user")
        return 1
    finally:
        elapsed = time.time() - start_time
        print(f"\n⏱️  Total execution time: {elapsed:.2f} seconds")


if __name__ == "__main__":
    exit_code = main()
    
    if exit_code == 0:
        print("✅ All topic detection tests passed!")
    else:
        print("❌ Some tests failed or were interrupted")
    
    sys.exit(exit_code) 