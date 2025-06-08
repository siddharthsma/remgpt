"""
Unit tests for remote tools module __init__.py.
"""

import pytest
from unittest.mock import patch


class TestRemoteToolsModuleImports:
    """Test imports from remote tools module."""
    
    def test_base_import(self):
        """Test importing base protocol."""
        from remgpt.tools.remote import RemoteToolProtocol
        assert RemoteToolProtocol is not None
    
    def test_tool_import(self):
        """Test importing RemoteTool."""
        from remgpt.tools.remote import RemoteTool
        assert RemoteTool is not None
    
    def test_manager_import(self):
        """Test importing RemoteToolManager."""
        from remgpt.tools.remote import RemoteToolManager
        assert RemoteToolManager is not None


class TestConditionalImports:
    """Test conditional imports based on dependency availability."""
    
    @patch('remgpt.tools.remote.mcp.mcp', None)
    def test_mcp_unavailable_import(self):
        """Test that MCPProtocol is None when mcp library is not available."""
        # Need to reload the module to test the import condition
        import importlib
        import remgpt.tools.remote.mcp
        importlib.reload(remgpt.tools.remote.mcp)
        
        from remgpt.tools.remote import MCPProtocol
        assert MCPProtocol is None
    
    @patch('remgpt.tools.remote.a2a.httpx', None)
    def test_a2a_unavailable_import(self):
        """Test that A2AProtocol is None when httpx library is not available."""
        # Need to reload the module to test the import condition
        import importlib
        import remgpt.tools.remote.a2a
        importlib.reload(remgpt.tools.remote.a2a)
        
        from remgpt.tools.remote import A2AProtocol
        assert A2AProtocol is None
    
    def test_all_imports_with_dependencies(self):
        """Test that all imports work when dependencies are available."""
        # Mock the dependencies to be available
        with patch('remgpt.tools.remote.mcp.mcp', object()), \
             patch('remgpt.tools.remote.a2a.httpx', object()):
            
            # Reload modules to test with mocked dependencies
            import importlib
            import remgpt.tools.remote.mcp
            import remgpt.tools.remote.a2a
            importlib.reload(remgpt.tools.remote.mcp)
            importlib.reload(remgpt.tools.remote.a2a)
            
            from remgpt.tools.remote import MCPProtocol, A2AProtocol
            assert MCPProtocol is not None
            assert A2AProtocol is not None


class TestModuleExports:
    """Test that the module exports the expected classes."""
    
    def test_all_exports_present(self):
        """Test that all expected exports are present."""
        import remgpt.tools.remote as remote_module
        
        # Base classes should always be available
        assert hasattr(remote_module, 'RemoteToolProtocol')
        assert hasattr(remote_module, 'RemoteTool')
        assert hasattr(remote_module, 'RemoteToolManager')
        
        # Protocol implementations may be None if dependencies are missing
        assert hasattr(remote_module, 'MCPProtocol')  # May be None
        assert hasattr(remote_module, 'A2AProtocol')  # May be None
    
    def test_exported_classes_are_correct_types(self):
        """Test that exported classes are the correct types when available."""
        from remgpt.tools.remote import (
            RemoteToolProtocol, RemoteTool, RemoteToolManager,
            MCPProtocol, A2AProtocol
        )
        
        # Base classes should always be classes
        assert isinstance(RemoteToolProtocol, type)
        assert isinstance(RemoteTool, type)
        assert isinstance(RemoteToolManager, type)
        
        # Protocol implementations should be classes if not None
        if MCPProtocol is not None:
            assert isinstance(MCPProtocol, type)
        
        if A2AProtocol is not None:
            assert isinstance(A2AProtocol, type)


class TestAvailabilityChecks:
    """Test availability check functions."""
    
    def test_availability_function_exists(self):
        """Test that availability check functions exist if defined."""
        import remgpt.tools.remote as remote_module
        
        # Check if availability functions are defined
        # These might be defined in the module for checking dependencies
        if hasattr(remote_module, 'is_mcp_available'):
            assert callable(remote_module.is_mcp_available)
        
        if hasattr(remote_module, 'is_a2a_available'):
            assert callable(remote_module.is_a2a_available)


class TestModuleDocumentation:
    """Test that the module has proper documentation."""
    
    def test_module_has_docstring(self):
        """Test that the module has a docstring."""
        import remgpt.tools.remote as remote_module
        assert remote_module.__doc__ is not None
        assert len(remote_module.__doc__.strip()) > 0
    
    def test_classes_have_docstrings(self):
        """Test that exported classes have docstrings."""
        from remgpt.tools.remote import (
            RemoteToolProtocol, RemoteTool, RemoteToolManager,
            MCPProtocol, A2AProtocol
        )
        
        assert RemoteToolProtocol.__doc__ is not None
        assert RemoteTool.__doc__ is not None
        assert RemoteToolManager.__doc__ is not None
        
        if MCPProtocol is not None:
            assert MCPProtocol.__doc__ is not None
        
        if A2AProtocol is not None:
            assert A2AProtocol.__doc__ is not None


class TestImportErrors:
    """Test proper handling of import errors."""
    
    def test_graceful_import_degradation(self):
        """Test that the module imports gracefully even with missing dependencies."""
        # This test verifies that importing the module doesn't crash
        # even when optional dependencies are missing
        
        with patch('remgpt.tools.remote.mcp.mcp', None), \
             patch('remgpt.tools.remote.a2a.httpx', None):
            
            # Should not raise any exceptions
            import remgpt.tools.remote as remote_module
            
            # Basic classes should still be available
            assert remote_module.RemoteToolProtocol is not None
            assert remote_module.RemoteTool is not None
            assert remote_module.RemoteToolManager is not None
            
            # Optional classes should be None
            assert remote_module.MCPProtocol is None
            assert remote_module.A2AProtocol is None
    
    def test_import_from_statements(self):
        """Test that 'from' import statements work correctly."""
        # Test individual imports
        from remgpt.tools.remote import RemoteToolProtocol
        from remgpt.tools.remote import RemoteTool
        from remgpt.tools.remote import RemoteToolManager
        
        # These may be None if dependencies are missing, but should not raise ImportError
        from remgpt.tools.remote import MCPProtocol
        from remgpt.tools.remote import A2AProtocol
        
        # Verify types
        assert RemoteToolProtocol is not None
        assert RemoteTool is not None
        assert RemoteToolManager is not None 