"""
OpenAI provider implementation for MDMAI TTRPG Assistant.

This module provides OpenAI GPT integration with authentication,
validation, and TTRPG-optimized features.
"""

from typing import Dict, Any, AsyncIterator, List, Optional
import asyncio
import logging
import time
from datetime import datetime, timedelta

try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    AsyncOpenAI = None

from .base_provider import (
    BaseAIProvider, ProviderConfig, ProviderType,
    ProviderError, ProviderAuthenticationError, ProviderRateLimitError,
    ProviderTimeoutError, ProviderQuotaExceededError, ProviderInvalidRequestError
)

logger = logging.getLogger(__name__)


class OpenAIProvider(BaseAIProvider):
    """
    OpenAI GPT provider implementation.
    
    Features:
    - Authentication with API key validation
    - TTRPG-optimized model selection
    - Function calling support
    - Streaming responses
    - Batch API support
    - Multi-modal features
    - Cost estimation
    """
    
    def __init__(self, config: ProviderConfig):
        """
        Initialize OpenAI provider.
        
        Args:
            config: Provider configuration
            
        Raises:
            ImportError: If openai package is not installed
            ValueError: If configuration is invalid
        """
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "openai package is not installed. Install with: pip install openai"
            )
        
        super().__init__(config)
        
        if config.type != ProviderType.OPENAI:
            raise ValueError(f"Invalid provider type: {config.type}. Expected OPENAI")
        
        # Initialize async client
        self.client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.endpoint  # Allow custom endpoints for Azure/other
        )
        
        # Model optimization for TTRPG
        self.model_mapping = {
            'fast': 'gpt-4o-mini',      # Economy option
            'balanced': 'gpt-4o',        # Recommended
            'powerful': 'gpt-4-turbo',   # Premium features
            'vision': 'gpt-4-vision-preview'  # Multi-modal support
        }
        
        # Cost tracking (per 1M tokens in USD)
        self.pricing = {
            'gpt-4o-mini': {'input': 0.15, 'output': 0.60},
            'gpt-4o': {'input': 2.50, 'output': 10.00},
            'gpt-4-turbo': {'input': 10.00, 'output': 30.00},
            'gpt-4-turbo-2024-04-09': {'input': 10.00, 'output': 30.00},
            'gpt-4-vision-preview': {'input': 10.00, 'output': 30.00}
        }
        
        # Rate limiting tracking
        self._last_request_time = 0.0
        self._request_count = 0
        self._reset_time = time.time() + 60  # Reset every minute
        
        logger.info(f"OpenAIProvider initialized with model mapping: {self.model_mapping}")
    
    async def complete(
        self, 
        messages: List[Dict[str, str]], 
        tools: Optional[List[Dict]] = None,
        stream: bool = False
    ) -> AsyncIterator[str]:
        """
        Generate completion from OpenAI GPT.
        
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
            # Apply rate limiting
            await self._apply_rate_limit()
            
            # Convert tools to OpenAI tool format
            openai_tools = self._convert_to_openai_tools(tools) if tools else None
            
            # Select model
            model = self.config.model or self.model_mapping.get('balanced', 'gpt-4o')
            
            # Prepare request parameters
            request_params = {
                "model": model,
                "messages": messages,  # OpenAI uses standard format
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "stream": stream
            }
            
            if openai_tools:
                request_params["tools"] = openai_tools
                request_params["tool_choice"] = "auto"
            
            # Make request with timeout
            start_time = time.time()
            
            try:
                response = await asyncio.wait_for(
                    self.client.chat.completions.create(**request_params),
                    timeout=self.config.timeout
                )
            except asyncio.TimeoutError:
                elapsed = time.time() - start_time
                raise ProviderTimeoutError(
                    f"OpenAI request timed out after {elapsed:.1f}s",
                    provider="openai",
                    timeout_seconds=elapsed
                )
            
            if stream:
                # Properly yield chunks as they arrive for true streaming
                try:
                    async for chunk in response:
                        if (hasattr(chunk, 'choices') and chunk.choices and 
                            hasattr(chunk.choices[0], 'delta') and 
                            hasattr(chunk.choices[0].delta, 'content') and 
                            chunk.choices[0].delta.content):
                            yield chunk.choices[0].delta.content
                except Exception as e:
                    logger.error(f"Error during OpenAI streaming: {e}")
                    raise ProviderError(f"Streaming error: {e}", provider="openai")
            else:
                # For non-streaming, yield the complete response
                if (hasattr(response, 'choices') and response.choices and 
                    hasattr(response.choices[0], 'message') and 
                    hasattr(response.choices[0].message, 'content') and
                    response.choices[0].message.content):
                    yield response.choices[0].message.content
                    
        except ProviderError:
            raise
        except Exception as e:
            await self._handle_openai_error(e)
    
    async def validate_credentials(self) -> bool:
        """
        Validate OpenAI API credentials.
        
        Returns:
            bool: True if credentials are valid
            
        Raises:
            ProviderAuthenticationError: If credentials are invalid
        """
        try:
            # Test with a minimal request
            start_time = time.time()
            response = await asyncio.wait_for(
                self.client.chat.completions.create(
                    model=self.model_mapping['fast'],  # Use fastest model for validation
                    messages=[{"role": "user", "content": "Hi"}],
                    max_tokens=5,
                    temperature=0.0
                ),
                timeout=10.0  # Short timeout for validation
            )
            
            if (hasattr(response, 'choices') and response.choices and 
                hasattr(response.choices[0], 'message')):
                self._authenticated = True
                elapsed = time.time() - start_time
                logger.info(f"OpenAI credentials validated successfully in {elapsed:.2f}s")
                return True
            else:
                raise ProviderAuthenticationError(
                    "Invalid response from OpenAI API during validation",
                    provider="openai"
                )
                
        except asyncio.TimeoutError:
            raise ProviderAuthenticationError(
                "OpenAI API validation timed out",
                provider="openai"
            )
        except Exception as e:
            logger.error(f"OpenAI credential validation failed: {e}")
            raise ProviderAuthenticationError(
                f"OpenAI API key validation failed: {str(e)}",
                provider="openai"
            )
    
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """
        Estimate cost for OpenAI request.
        
        Args:
            input_tokens: Number of input tokens
            output_tokens: Expected number of output tokens
            
        Returns:
            float: Estimated cost in USD
        """
        model = self.config.model or self.model_mapping.get('balanced')
        
        # Find matching pricing model
        pricing_model = None
        for price_key in self.pricing:
            if price_key in model:
                pricing_model = price_key
                break
        
        if not pricing_model:
            logger.warning(f"No pricing data for model {model}, using default")
            pricing_model = 'gpt-4o'
        
        pricing_data = self.pricing[pricing_model]
        
        input_cost = (input_tokens / 1_000_000) * pricing_data['input']
        output_cost = (output_tokens / 1_000_000) * pricing_data['output']
        
        return input_cost + output_cost
    
    async def health_check(self) -> bool:
        """
        Check if OpenAI service is healthy.
        
        Returns:
            bool: True if service is healthy
        """
        try:
            # Simple health check with minimal request
            start_time = time.time()
            
            response = await asyncio.wait_for(
                self.client.chat.completions.create(
                    model=self.model_mapping['fast'],
                    messages=[{"role": "user", "content": "ping"}],
                    max_tokens=1,
                    temperature=0.0
                ),
                timeout=5.0
            )
            
            elapsed = time.time() - start_time
            self._last_health_check = datetime.now()
            
            logger.debug(f"OpenAI health check passed in {elapsed:.2f}s")
            return True
            
        except Exception as e:
            logger.warning(f"OpenAI health check failed: {e}")
            return False
    
    def _convert_to_openai_tools(self, tools: List[Dict]) -> List[Dict]:
        """
        Convert generic tool format to OpenAI's tools format.
        
        Args:
            tools: Tools in generic format
            
        Returns:
            List[Dict]: Tools in OpenAI format
        """
        openai_tools = []
        
        for tool in tools:
            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters", {}),
                },
            }
            
            # Ensure parameters have required structure
            if "type" not in openai_tool["function"]["parameters"]:
                openai_tool["function"]["parameters"]["type"] = "object"
            
            openai_tools.append(openai_tool)
        
        return openai_tools
    
    async def create_batch_request(self, requests: List[Dict]) -> str:
        """
        Create a batch request for processing multiple completions.
        
        Args:
            requests: List of completion requests
            
        Returns:
            str: Batch ID for tracking
            
        Raises:
            ProviderError: If batch creation fails
        """
        try:
            # Convert requests to OpenAI batch format
            batch_requests = []
            for i, req in enumerate(requests):
                batch_requests.append({
                    "custom_id": f"request-{i}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": req
                })
            
            # Create JSONL file for batch
            import tempfile
            import json
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
                for req in batch_requests:
                    json.dump(req, f)
                    f.write('\n')
                jsonl_file = f.name
            
            # Upload file and create batch
            with open(jsonl_file, 'rb') as f:
                file_response = await self.client.files.create(
                    file=f,
                    purpose='batch'
                )
            
            batch_response = await self.client.batches.create(
                input_file_id=file_response.id,
                endpoint="/v1/chat/completions",
                completion_window="24h"
            )
            
            logger.info(f"Created batch request with ID: {batch_response.id}")
            return batch_response.id
            
        except Exception as e:
            logger.error(f"Failed to create batch request: {e}")
            raise ProviderError(f"Batch creation failed: {e}", provider="openai")
    
    async def get_batch_status(self, batch_id: str) -> Dict[str, Any]:
        """
        Get status of a batch request.
        
        Args:
            batch_id: Batch ID to check
            
        Returns:
            Dict[str, Any]: Batch status information
        """
        try:
            batch = await self.client.batches.retrieve(batch_id)
            return {
                "id": batch.id,
                "status": batch.status,
                "request_counts": batch.request_counts,
                "created_at": batch.created_at,
                "completed_at": getattr(batch, 'completed_at', None),
                "failed_at": getattr(batch, 'failed_at', None),
                "output_file_id": getattr(batch, 'output_file_id', None)
            }
        except Exception as e:
            logger.error(f"Failed to get batch status: {e}")
            raise ProviderError(f"Batch status check failed: {e}", provider="openai")
    
    async def _apply_rate_limit(self) -> None:
        """Apply rate limiting to prevent API abuse."""
        current_time = time.time()
        
        # Reset counter if minute has passed
        if current_time > self._reset_time:
            self._request_count = 0
            self._reset_time = current_time + 60
        
        # Check if we need to wait (OpenAI has higher limits)
        if self._request_count >= 500:  # More generous limit for OpenAI
            wait_time = self._reset_time - current_time
            if wait_time > 0:
                logger.info(f"Rate limiting: waiting {wait_time:.1f}s")
                await asyncio.sleep(wait_time)
                self._request_count = 0
                self._reset_time = time.time() + 60
        
        # Minimum time between requests
        time_since_last = current_time - self._last_request_time
        if time_since_last < 0.02:  # 20ms minimum (faster than Anthropic)
            await asyncio.sleep(0.02 - time_since_last)
        
        self._request_count += 1
        self._last_request_time = time.time()
    
    async def _handle_openai_error(self, error: Exception) -> None:
        """
        Handle and convert OpenAI-specific errors.
        
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
                f"OpenAI rate limit exceeded: {error}",
                provider="openai",
                retry_after=retry_after
            )
        
        elif "authentication" in error_str or "unauthorized" in error_str or "401" in error_str:
            self._authenticated = False
            raise ProviderAuthenticationError(
                f"OpenAI authentication failed: {error}",
                provider="openai"
            )
        
        elif "quota" in error_str or "billing" in error_str or "insufficient" in error_str:
            raise ProviderQuotaExceededError(
                f"OpenAI quota exceeded: {error}",
                provider="openai"
            )
        
        elif "invalid" in error_str or "400" in error_str:
            raise ProviderInvalidRequestError(
                f"Invalid request to OpenAI: {error}",
                provider="openai"
            )
        
        else:
            raise ProviderError(
                f"OpenAI API error: {error}",
                provider="openai"
            )