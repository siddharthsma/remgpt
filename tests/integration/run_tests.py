#!/usr/bin/env python3
"""
Simple script to run integration tests.
Ensures proper environment setup and provides clear instructions.
"""

import os
import sys
import subprocess
from pathlib import Path

def check_environment():
    """Check if environment is properly set up."""
    integration_dir = Path(__file__).parent
    env_file = integration_dir / ".env"
    
    if not env_file.exists():
        print("❌ .env file not found!")
        print(f"Please create {env_file} with your OpenAI API key:")
        print("OPENAI_API_KEY=your_openai_api_key_here")
        return False
    
    # Check if OPENAI_API_KEY is in environment
    from dotenv import load_dotenv
    load_dotenv(env_file)
    
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not found in .env file!")
        print(f"Please add your OpenAI API key to {env_file}")
        return False
    
    print("✅ Environment configured correctly")
    return True

def run_integration_tests():
    """Run the integration tests."""
    if not check_environment():
        return 1
    
    print("\n🧪 Running integration tests with real OpenAI API calls...")
    print("⚠️  This will make real API calls and may incur small costs (~$0.01-0.02)")
    
    # Change to the integration tests directory
    integration_dir = Path(__file__).parent
    os.chdir(integration_dir)
    
    # Run pytest with integration marker
    cmd = [
        sys.executable, "-m", "pytest", 
        "test_end_to_end_system.py",
        "-v", "-s", "-m", "integration"
    ]
    
    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except KeyboardInterrupt:
        print("\n⚠️  Tests interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(run_integration_tests()) 