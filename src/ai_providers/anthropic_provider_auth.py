"""
Anthropic provider implementation for MDMAI TTRPG Assistant.

This module provides Anthropic Claude integration with authentication,
validation, and TTRPG-optimized features.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, AsyncIterator, List, Optional, Tuple

try:
    from anthropic import Anthropic, AsyncAnthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    Anthropic = None
    AsyncAnthropic = None

from .base_provider import (
    BaseAIProvider, ProviderConfig, ProviderType,
    ProviderError, ProviderAuthenticationError, ProviderRateLimitError,
    ProviderTimeoutError, ProviderQuotaExceededError, ProviderInvalidRequestError
)
from .pricing_config import get_pricing_manager
from .rate_limiter import RateLimiter

logger = logging.getLogger(__name__)


class AnthropicProvider(BaseAIProvider):
    """
    Anthropic Claude provider implementation.
    
    Features:
    - Authentication with API key validation
    - TTRPG-optimized model selection
    - Prompt caching for efficiency
    - Tool calling support
    - Streaming responses
    - Cost estimation
    """
    
    def __init__(self, config: ProviderConfig):
        """
        Initialize Anthropic provider.
        
        Args:
            config: Provider configuration
            
        Raises:
            ImportError: If anthropic package is not installed
            ValueError: If configuration is invalid
        """
        if not ANTHROPIC_AVAILABLE:
            raise ImportError(
                "anthropic package is not installed. Install with: pip install anthropic"
            )
        
        super().__init__(config)
        
        if config.type != ProviderType.ANTHROPIC:
            raise ValueError(f"Invalid provider type: {config.type}. Expected ANTHROPIC")
        
        # Initialize both sync and async clients
        self.client = Anthropic(api_key=config.api_key)
        self.async_client = AsyncAnthropic(api_key=config.api_key)
        
        # Model selection with TTRPG optimization
        self.model_mapping = {
            'fast': 'claude-3-haiku',      # Quick responses, dice rolls
            'balanced': 'claude-3-5-sonnet',  # Main gameplay
            'powerful': 'claude-3-opus'    # Complex narratives
        }
        
        # Use centralized services
        self.pricing_manager = get_pricing_manager()
        self.rate_limiter = RateLimiter()
        
        logger.info(f"AnthropicProvider initialized with model mapping: {self.model_mapping}")
    
    async def complete(
        self, 
        messages: List[Dict[str, str]], 
        tools: Optional[List[Dict]] = None,
        stream: bool = False
    ) -> AsyncIterator[str]:
        """
        Generate completion from Anthropic Claude.
        
        Args:
            messages: List of chat messages
            tools: Optional tools for function calling
            stream: Whether to stream the response
            
        Yields:
            str: Content chunks from the model
            
        Raises:
            ProviderError: If request fails
        """
        if not self._authenticated:
            await self.validate_credentials()
        
        try:
            # Apply rate limiting using centralized service
            await self.rate_limiter.acquire(ProviderType.ANTHROPIC, self.config.user_id, priority=5)
            
            # Convert to Anthropic format
            anthropic_messages = self._convert_messages(messages)
            
            # Get TTRPG-specific system prompt
            system_prompt = self.get_ttrpg_system_prompt()
            
            # Select model
            model = self.config.model or self.model_mapping.get('balanced', 'claude-3-5-sonnet')
            
            # Convert tools if provided
            anthropic_tools = self._convert_tools_to_anthropic(tools) if tools else None
            
            # Prepare request parameters
            request_params = {
                "model": model,
                "messages": anthropic_messages,
                "system": system_prompt,
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "stream": stream,
                "extra_headers": {
                    "anthropic-beta": "prompt-caching-2024-07-31"
                }
            }
            
            if anthropic_tools:
                request_params["tools"] = anthropic_tools
            
            # Make request with timeout
            start_time = time.time()
            
            try:
                response = await asyncio.wait_for(
                    self.async_client.messages.create(**request_params),
                    timeout=self.config.timeout
                )
            except asyncio.TimeoutError:
                elapsed = time.time() - start_time
                raise ProviderTimeoutError(
                    f"Anthropic request timed out after {elapsed:.1f}s",
                    provider="anthropic",
                    timeout_seconds=elapsed
                )
            
            if stream:
                # Properly yield chunks as they arrive for true streaming
                try:
                    async for chunk in response:
                        if hasattr(chunk, 'delta') and hasattr(chunk.delta, 'text'):
                            if chunk.delta.text:
                                yield chunk.delta.text
                except Exception as e:
                    logger.error(f"Error during Anthropic streaming: {e}")
                    raise ProviderError(f"Streaming error: {e}", provider="anthropic")
            else:
                # For non-streaming, yield the complete response
                if hasattr(response, 'content') and response.content:
                    yield response.content[0].text
                    
        except ProviderError:
            raise
        except Exception as e:
            # Record failure for rate limiting
            is_rate_limit = "rate_limit" in str(e).lower() or "429" in str(e)
            self.rate_limiter.record_failure(ProviderType.ANTHROPIC, self.config.user_id, is_rate_limit)
            await self._handle_anthropic_error(e)
        else:
            # Record success for adaptive rate limiting
            self.rate_limiter.record_success(ProviderType.ANTHROPIC, self.config.user_id)
    
    async def validate_credentials(self) -> bool:
        """
        Validate Anthropic API credentials.
        
        Returns:
            bool: True if credentials are valid
            
        Raises:
            ProviderAuthenticationError: If credentials are invalid
        """
        try:
            # Test with a minimal request
            test_messages = [{"role": "user", "content": "Hi"}]
            
            start_time = time.time()
            response = await asyncio.wait_for(
                self.async_client.messages.create(
                    model=self.model_mapping['fast'],  # Use fastest model for validation
                    messages=test_messages,
                    max_tokens=10,
                    temperature=0.0
                ),
                timeout=10.0  # Short timeout for validation
            )
            
            if hasattr(response, 'content') and response.content:
                self._authenticated = True
                elapsed = time.time() - start_time
                logger.info(f"Anthropic credentials validated successfully in {elapsed:.2f}s")
                return True
            else:
                raise ProviderAuthenticationError(
                    "Invalid response from Anthropic API during validation",
                    provider="anthropic"
                )
                
        except asyncio.TimeoutError:
            raise ProviderAuthenticationError(
                "Anthropic API validation timed out",
                provider="anthropic"
            )
        except Exception as e:
            logger.error(f"Anthropic credential validation failed: {e}")
            raise ProviderAuthenticationError(
                f"Anthropic API key validation failed: {str(e)}",
                provider="anthropic"
            )
    
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """
        Estimate cost for Anthropic request using centralized pricing.
        
        Args:
            input_tokens: Number of input tokens
            output_tokens: Expected number of output tokens
            
        Returns:
            float: Estimated cost in USD
        """
        model = self.config.model or self.model_mapping.get('balanced', 'claude-3-5-sonnet')
        return self.pricing_manager.calculate_cost(
            ProviderType.ANTHROPIC, model, input_tokens, output_tokens
        )
    
    async def health_check(self) -> bool:
        """
        Check if Anthropic service is healthy.
        
        Returns:
            bool: True if service is healthy
        """
        try:
            # Simple health check with minimal request
            start_time = time.time()
            
            response = await asyncio.wait_for(
                self.async_client.messages.create(
                    model=self.model_mapping['fast'],
                    messages=[{"role": "user", "content": "ping"}],
                    max_tokens=1,
                    temperature=0.0
                ),
                timeout=5.0
            )
            
            elapsed = time.time() - start_time
            self._last_health_check = datetime.now()
            
            logger.debug(f"Anthropic health check passed in {elapsed:.2f}s")
            return True
            
        except Exception as e:
            logger.warning(f"Anthropic health check failed: {e}")
            return False
    
    def _convert_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Convert generic message format to Anthropic's specific format.
        
        Args:
            messages: Messages in generic format
            
        Returns:
            List[Dict[str, str]]: Messages in Anthropic format
        """
        anthropic_messages = []
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            # Anthropic uses 'user' and 'assistant' roles
            if role in ["user", "assistant"]:
                anthropic_messages.append({
                    "role": role,
                    "content": content
                })
            elif role == "system":
                # System messages are handled separately in Anthropic
                logger.debug("System message found - will be used in system parameter")
            else:
                # Convert unknown roles to user
                anthropic_messages.append({
                    "role": "user",
                    "content": content
                })
        
        return anthropic_messages
    
    def _convert_tools_to_anthropic(self, tools: List[Dict]) -> List[Dict]:
        """
        Convert generic tool format to Anthropic's tool format.
        
        Args:
            tools: Tools in generic format
            
        Returns:
            List[Dict]: Tools in Anthropic format
        """
        anthropic_tools = []
        
        for tool in tools:
            anthropic_tool = {
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
                "input_schema": tool.get("parameters", {})
            }
            
            # Ensure input_schema has required structure
            if "type" not in anthropic_tool["input_schema"]:
                anthropic_tool["input_schema"]["type"] = "object"
            
            anthropic_tools.append(anthropic_tool)
        
        return anthropic_tools
    
    
    async def _handle_anthropic_error(self, error: Exception) -> None:
        """
        Handle and convert Anthropic-specific errors.
        
        Args:
            error: The exception that occurred
            
        Raises:
            ProviderError: Converted error with appropriate type
        """
        error_str = str(error).lower()
        
        if "rate_limit" in error_str or "429" in error_str:
            # Extract retry-after if available
            retry_after = None
            if hasattr(error, 'response') and error.response:
                retry_after = error.response.headers.get('retry-after')
                if retry_after:
                    try:
                        retry_after = int(retry_after)
                    except (ValueError, TypeError):
                        retry_after = None
            
            raise ProviderRateLimitError(
                f"Anthropic rate limit exceeded: {error}",
                provider="anthropic",
                retry_after=retry_after
            )
        
        elif "authentication" in error_str or "unauthorized" in error_str or "401" in error_str:
            self._authenticated = False
            raise ProviderAuthenticationError(
                f"Anthropic authentication failed: {error}",
                provider="anthropic"
            )
        
        elif "quota" in error_str or "billing" in error_str:
            raise ProviderQuotaExceededError(
                f"Anthropic quota exceeded: {error}",
                provider="anthropic"
            )
        
        elif "invalid" in error_str or "400" in error_str:
            raise ProviderInvalidRequestError(
                f"Invalid request to Anthropic: {error}",
                provider="anthropic"
            )
        
        else:
            raise ProviderError(
                f"Anthropic API error: {error}",
                provider="anthropic"
            )