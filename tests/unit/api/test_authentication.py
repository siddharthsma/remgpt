"""
Focused tests for authentication functionality.
"""

import pytest
from fastapi import HTTPException
from unittest.mock import patch, MagicMock

from remgpt.api import get_current_user


class TestAuthenticationDetail:
    """Detailed authentication tests."""
    
    @pytest.mark.auth
    @pytest.mark.asyncio
    async def test_bearer_token_extraction(self):
        """Test correct extraction of bearer token."""
        token = "my_secret_token_123456"
        authorization = f"Bearer {token}"
        
        result = await get_current_user(authorization)
        
        # Should extract first 8 characters for mock user ID
        expected_user = f"user_{token[:8]}"
        assert result == expected_user
    
    @pytest.mark.auth
    @pytest.mark.asyncio
    async def test_case_sensitive_bearer(self):
        """Test that Bearer is case sensitive."""
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user("bearer token123456789")
        
        assert exc_info.value.status_code == 401
        assert "Invalid authorization format" in exc_info.value.detail
    
    @pytest.mark.auth
    @pytest.mark.asyncio
    async def test_whitespace_handling(self):
        """Test handling of whitespace in authorization header."""
        # Extra spaces after Bearer should not break parsing
        result = await get_current_user("Bearer  token_with_spaces")
        assert result.startswith("user_")
        
        # Leading/trailing whitespace in the header itself would be handled by HTTP parser
        # But our function doesn't handle leading whitespace before "Bearer"
        with pytest.raises(HTTPException):
            await get_current_user("  Bearer token123456789  ")
    
    @pytest.mark.auth
    @pytest.mark.asyncio
    async def test_minimum_token_length(self):
        """Test minimum token length requirement."""
        # Test boundary condition - exactly 10 characters should pass
        result = await get_current_user("Bearer 1234567890")
        assert result == "user_12345678"
        
        # 9 characters should fail
        with pytest.raises(HTTPException):
            await get_current_user("Bearer 123456789")
    
    @pytest.mark.auth
    @pytest.mark.asyncio
    async def test_very_long_token(self):
        """Test handling of very long tokens."""
        long_token = "a" * 1000
        authorization = f"Bearer {long_token}"
        
        result = await get_current_user(authorization)
        # Should still extract first 8 characters
        assert result == "user_aaaaaaaa"
    
    @pytest.mark.auth
    @pytest.mark.asyncio
    async def test_special_characters_in_token(self):
        """Test tokens with special characters."""
        special_token = "token-with_special.chars123"
        authorization = f"Bearer {special_token}"
        
        result = await get_current_user(authorization)
        assert result == "user_token-wi"
    
    @pytest.mark.auth
    @pytest.mark.asyncio 
    async def test_unicode_token(self):
        """Test tokens with unicode characters."""
        unicode_token = "tökën123456789"
        authorization = f"Bearer {unicode_token}"
        
        result = await get_current_user(authorization)
        assert result.startswith("user_")
    
    @pytest.mark.auth
    @pytest.mark.asyncio
    async def test_www_authenticate_header(self):
        """Test that WWW-Authenticate header is included in 401 responses."""
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(None)
        
        assert "WWW-Authenticate" in exc_info.value.headers
        assert exc_info.value.headers["WWW-Authenticate"] == "Bearer"
        
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user("Invalid format")
        
        assert "WWW-Authenticate" in exc_info.value.headers
        assert exc_info.value.headers["WWW-Authenticate"] == "Bearer"


class TestJWTIntegrationPrep:
    """Tests to prepare for future JWT integration."""
    
    @pytest.mark.auth
    @pytest.mark.asyncio
    async def test_jwt_like_token_structure(self):
        """Test handling of JWT-like token structures."""
        # Simulate a JWT token (header.payload.signature)
        jwt_like = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
        
        result = await get_current_user(f"Bearer {jwt_like}")
        
        # Should extract first 8 characters
        assert result == "user_eyJhbGci"
    
    @pytest.mark.auth
    def test_oauth_error_responses_compliance(self):
        """Test that error responses comply with OAuth 2.0 standards."""
        # OAuth 2.0 requires specific error response format
        # This test ensures our current implementation is ready for OAuth
        
        import asyncio
        
        async def check_error_format():
            try:
                await get_current_user(None)
            except HTTPException as e:
                # Should have 401 status
                assert e.status_code == 401
                # Should have WWW-Authenticate header
                assert "WWW-Authenticate" in e.headers
                # Should indicate Bearer scheme
                assert e.headers["WWW-Authenticate"] == "Bearer"
                # Detail should be descriptive
                assert "Authorization" in e.detail
                return True
            return False
        
        result = asyncio.run(check_error_format())
        assert result
    
    @pytest.mark.auth
    @pytest.mark.asyncio
    async def test_concurrent_authentication(self):
        """Test concurrent authentication requests."""
        import asyncio
        
        async def auth_request(token_suffix):
            # Use different prefixes to ensure unique first 8 characters
            return await get_current_user(f"Bearer token{token_suffix:03d}_12345")
        
        # Run 20 concurrent authentication requests
        tasks = [auth_request(i) for i in range(20)]
        results = await asyncio.gather(*tasks)
        
        # All should succeed
        assert len(results) == 20
        # All should be unique based on token (first 8 chars should differ)
        assert len(set(results)) == 20  # Each token has different first 8 chars
        # All should start with "user_"
        assert all(result.startswith("user_") for result in results)


@pytest.mark.auth
class TestAuthenticationErrorMessages:
    """Test authentication error message clarity."""
    
    @pytest.mark.asyncio
    async def test_missing_header_message(self):
        """Test clear error message for missing Authorization header."""
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(None)
        
        error_detail = exc_info.value.detail
        assert "Missing Authorization header" in error_detail
        assert "Bearer <token>" in error_detail
    
    @pytest.mark.asyncio
    async def test_invalid_format_message(self):
        """Test clear error message for invalid authorization format."""
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user("Basic username:password")
        
        error_detail = exc_info.value.detail
        assert "Invalid authorization format" in error_detail
        assert "Bearer <token>" in error_detail
    
    @pytest.mark.asyncio
    async def test_malformed_token_message(self):
        """Test clear error message for malformed tokens."""
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user("Bearer bad")
        
        error_detail = exc_info.value.detail
        assert "Invalid or malformed token" in error_detail 