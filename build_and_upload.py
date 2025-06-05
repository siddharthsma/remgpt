#!/usr/bin/env python3
"""
Simple script to build and upload RemGPT package to PyPI.
"""

import subprocess
import sys
import os


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed")
        print(f"Error: {e.stderr}")
        return False


def main():
    """Main build and upload process."""
    print("RemGPT Package Builder")
    print("=" * 30)
    
    # Clean previous builds
    if run_command("rm -rf dist/ build/ *.egg-info/", "Cleaning previous builds"):
        pass
    
    # Build the package
    if not run_command("python -m build", "Building package"):
        sys.exit(1)
    
    # Check if we should upload
    upload = input("\nUpload to PyPI? (y/N): ").lower().strip()
    
    if upload == 'y':
        # Upload to PyPI
        if not run_command("python -m twine upload dist/*", "Uploading to PyPI"):
            sys.exit(1)
        print("\n🎉 Package successfully uploaded to PyPI!")
    else:
        print("\n📦 Package built successfully. Files are in the 'dist/' directory.")
        print("To upload later, run: python -m twine upload dist/*")


if __name__ == "__main__":
    main() 