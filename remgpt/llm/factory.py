from typing import Dict, Any, Type
from .base import BaseLLMClient
from .providers import OpenAIClient, ClaudeClient, GeminiClient, MockLLMClient


class LLMClientFactory:
    """Factory class for creating LLM clients based on provider configuration."""
    
    _PROVIDERS: Dict[str, Type[BaseLLMClient]] = {
        "openai": OpenAIClient,
        "claude": ClaudeClient,
        "anthropic": ClaudeClient,  # Alias for Claude
        "gemini": GeminiClient,
        "google": GeminiClient,  # Alias for Gemini
        "mock": MockLLMClient,  # For testing and development
    }
    
    @classmethod
    def create_client(cls, provider: str, **kwargs) -> BaseLLMClient:
        """
        Create an LLM client for the specified provider.
        
        Args:
            provider: The LLM provider name (openai, claude, gemini, etc.)
            **kwargs: Provider-specific configuration parameters
            
        Returns:
            BaseLLMClient: An instance of the appropriate LLM client
            
        Raises:
            ValueError: If provider is not supported
            TypeError: If required parameters are missing
        """
        provider_lower = provider.lower()
        
        if provider_lower not in cls._PROVIDERS:
            supported = ", ".join(cls._PROVIDERS.keys())
            raise ValueError(f"Unsupported provider '{provider}'. Supported providers: {supported}")
        
        client_class = cls._PROVIDERS[provider_lower]
        
        # Validate required parameters
        required_params = cls._get_required_params(provider_lower)
        missing_params = [param for param in required_params if param not in kwargs]
        
        if missing_params:
            raise TypeError(f"Missing required parameters for {provider}: {missing_params}")
        
        return client_class(**kwargs)
    
    @classmethod
    def _get_required_params(cls, provider: str) -> list:
        """Get required parameters for a provider."""
        required_params_map = {
            "openai": ["model_name", "api_key"],
            "claude": ["model_name", "api_key"],
            "anthropic": ["model_name", "api_key"],
            "gemini": ["model_name", "api_key"],
            "google": ["model_name", "api_key"],
            "mock": ["model_name"],  # Mock client only needs model_name
        }
        
        return required_params_map.get(provider, ["model_name"])
    
    @classmethod
    def get_supported_providers(cls) -> list:
        """Get list of supported provider names."""
        return list(cls._PROVIDERS.keys())
    
    @classmethod
    def register_provider(cls, name: str, client_class: Type[BaseLLMClient]) -> None:
        """
        Register a new LLM provider.
        
        Args:
            name: Provider name
            client_class: LLM client class that inherits from BaseLLMClient
        """
        if not issubclass(client_class, BaseLLMClient):
            raise TypeError("Client class must inherit from BaseLLMClient")
        
        cls._PROVIDERS[name.lower()] = client_class
    
    @classmethod
    def get_provider_models(cls, provider: str) -> list:
        """
        Get supported models for a specific provider.
        
        Args:
            provider: Provider name
            
        Returns:
            list: List of supported model names
        """
        provider_lower = provider.lower()
        
        if provider_lower not in cls._PROVIDERS:
            return []
        
        client_class = cls._PROVIDERS[provider_lower]
        
        # Create a temporary instance to get supported models
        try:
            temp_client = client_class(model_name="temp", api_key="temp")
            return temp_client.get_supported_models()
        except:
            # If we can't create temp client, return empty list
            return [] 