"""
Utility functions for RemGPT.
"""


def validate_config(config):
    """Validate configuration dictionary."""
    return isinstance(config, dict)


def format_output(data, format_type="plain"):
    """Format output data."""
    if format_type == "upper":
        return data.upper()
    elif format_type == "lower":
        return data.lower()
    else:
        return data 