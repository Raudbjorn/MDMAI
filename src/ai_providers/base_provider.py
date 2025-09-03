"""
Base provider authentication interface for MDMAI TTRPG Assistant.

This module defines the abstract base classes and interfaces for LLM provider authentication,
following the specifications in the design document.
"""

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Dict, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ProviderType(Enum):
    """Enumeration of supported AI providers."""
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GOOGLE = "google"
    AZURE = "azure"
    COHERE = "cohere"
    LOCAL = "local"  # For Ollama/local models


@dataclass
class CompletionResponse:
    """Structured response from get_completion method."""
    content: str  # The actual completion text
    provider: ProviderType  # The provider that generated the response
    is_fallback: bool = False  # Whether this was from a fallback provider
    stream: Optional[AsyncIterator[str]] = None  # Stream iterator if streaming is enabled


# Custom exceptions for better error handling
class ProviderError(Exception):
    """Base exception for provider-related errors."""
    
    def __init__(self, message: str, provider: Optional[str] = None, retry_after: Optional[int] = None):
        super().__init__(message)
        self.provider = provider
        self.retry_after = retry_after


class ProviderAuthenticationError(ProviderError):
    """Raised when provider authentication fails."""
    
    def __init__(self, message: str, provider: Optional[str] = None):
        super().__init__(message, provider)
        self.error_code = "AUTH_FAILED"


class ProviderRateLimitError(ProviderError):
    """Raised when a provider hits rate limits."""
    
    def __init__(self, message: str, provider: Optional[str] = None, retry_after: Optional[int] = None):
        super().__init__(message, provider, retry_after)
        self.error_code = "RATE_LIMITED"


class ProviderTimeoutError(ProviderError):
    """Raised when a provider request times out."""
    
    def __init__(self, message: str, provider: Optional[str] = None, timeout_seconds: Optional[float] = None):
        super().__init__(message, provider)
        self.timeout_seconds = timeout_seconds
        self.error_code = "TIMEOUT"


class ProviderQuotaExceededError(ProviderError):
    """Raised when provider quota is exceeded."""
    
    def __init__(self, message: str, provider: Optional[str] = None, reset_time: Optional[datetime] = None):
        super().__init__(message, provider)
        self.reset_time = reset_time
        self.error_code = "QUOTA_EXCEEDED"


class ProviderInvalidRequestError(ProviderError):
    """Raised when request to provider is invalid."""
    
    def __init__(self, message: str, provider: Optional[str] = None, validation_errors: Optional[List[str]] = None):
        super().__init__(message, provider)
        self.validation_errors = validation_errors or []
        self.error_code = "INVALID_REQUEST"


class NoAvailableProvidersError(ProviderError):
    """Raised when all providers have failed."""
    
    def __init__(self, message: str = "All configured providers have failed", failed_providers: Optional[List[str]] = None):
        super().__init__(message)
        self.failed_providers = failed_providers or []
        self.error_code = "NO_PROVIDERS"


@dataclass
class ProviderConfig:
    """Configuration for AI provider."""
    type: ProviderType
    api_key: Optional[str] = None
    endpoint: Optional[str] = None
    model: str = ""  # Empty string means use provider's default model
    max_tokens: int = 4096
    temperature: float = 0.7
    timeout: float = 30.0
    max_retries: int = 3
    user_id: str = "default_user"  # User ID for rate limiting
    
    def validate(self) -> bool:
        """Validate the provider configuration."""
        if not self.api_key:
            raise ValueError("API key is required")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0")
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        return True


class BaseAIProvider(ABC):
    """
    Abstract base class for all AI providers.
    
    This class defines the interface that all AI providers must implement
    for authentication, completion generation, and cost estimation.
    """
    
    def __init__(self, config: ProviderConfig):
        """Initialize the provider with configuration."""
        config.validate()
        self.config = config
        self._authenticated = False
        self._last_health_check = None
        
    @abstractmethod
    async def complete(
        self, 
        messages: List[Dict[str, str]], 
        tools: Optional[List[Dict]] = None,
        stream: bool = False
    ) -> AsyncIterator[str]:
        """
        Generate completion from the AI provider.
        
        Args:
            messages: List of chat messages in standard format
            tools: Optional list of tools/functions available to the model
            stream: Whether to stream the response
            
        Yields:
            str: Content chunks from the provider
            
        Raises:
            ProviderError: If the request fails for any reason
        """
        pass
    
    @abstractmethod
    async def validate_credentials(self) -> bool:
        """
        Validate API credentials are working.
        
        Returns:
            bool: True if credentials are valid, False otherwise.
            
        Raises:
            ProviderAuthenticationError: If credentials are invalid.
            ProviderTimeoutError: If validation request times out.
        """
        pass
    
    @abstractmethod
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """
        Estimate cost for the request.
        
        Args:
            input_tokens: Number of input tokens
            output_tokens: Expected number of output tokens
            
        Returns:
            float: Estimated cost in USD
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if the provider is healthy and responsive.
        
        Returns:
            bool: True if provider is healthy, False otherwise
        """
        pass
    
    @property
    def is_authenticated(self) -> bool:
        """Check if the provider is authenticated."""
        return self._authenticated
    
    def get_ttrpg_system_prompt(self) -> str:
        """
        Get the TTRPG-specific system prompt.
        
        Returns:
            str: System prompt optimized for TTRPG gameplay
        """
        return """You are an expert Tabletop RPG Game Master and storyteller. 
        Your role is to create immersive narratives, manage game mechanics, 
        and ensure all players have an engaging experience. You have deep 
        knowledge of various TTRPG systems and can adapt to different playstyles.
        
        Guidelines:
        - Maintain narrative consistency and immersion
        - Balance challenge and player agency
        - Use vivid descriptions for scenes and encounters
        - Handle dice rolls and mechanics fairly
        - Encourage creative problem-solving
        - Adapt to the party's interests and playstyle"""
    
    def convert_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Convert generic message format to provider-specific format.
        
        Args:
            messages: List of messages in generic format
            
        Returns:
            List[Dict[str, str]]: Messages in provider-specific format
        """
        # Default implementation - override in subclasses if needed
        return messages
    
    def convert_tools(self, tools: Optional[List[Dict]]) -> Optional[List[Dict]]:
        """
        Convert generic tool format to provider-specific format.
        
        Args:
            tools: List of tools in generic format
            
        Returns:
            Optional[List[Dict]]: Tools in provider-specific format
        """
        # Default implementation - override in subclasses if needed
        return tools