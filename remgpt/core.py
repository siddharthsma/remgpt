"""
Core functionality for RemGPT.
"""


class RemGPT:
    """Main RemGPT class."""
    
    def __init__(self, config=None):
        """Initialize RemGPT."""
        self.config = config or {}
    
    def process(self, input_data):
        """Process input data."""
        return f"Processed: {input_data}" 