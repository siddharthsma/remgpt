"""
RemGPT - A Python library for RemGPT functionality.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Import main classes/functions
from .core import RemGPT
from .utils import validate_config, format_output

__all__ = [
    "RemGPT",
    "validate_config", 
    "format_output",
] 