"""
Google Gemini provider implementation for MDMAI TTRPG Assistant.

This module provides Google Gemini integration with authentication,
validation, and TTRPG-optimized features.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, AsyncIterator, List, Optional, Tuple

try:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False
    genai = None

from .base_provider import (
    BaseAIProvider, ProviderConfig, ProviderType,
    ProviderError, ProviderAuthenticationError, ProviderRateLimitError,
    ProviderTimeoutError, ProviderQuotaExceededError, ProviderInvalidRequestError
)
from .pricing_config import get_pricing_manager
from .rate_limiter import RateLimiter

logger = logging.getLogger(__name__)


class GoogleProvider(BaseAIProvider):
    """
    Google Gemini provider implementation.
    
    Features:
    - Authentication with API key validation
    - TTRPG-optimized model selection
    - Multi-modal support (text, images)
    - Large context windows
    - Safety settings configuration
    - Cost estimation
    """
    
    def __init__(self, config: ProviderConfig):
        """
        Initialize Google Gemini provider.
        
        Args:
            config: Provider configuration
            
        Raises:
            ImportError: If google-generativeai package is not installed
            ValueError: If configuration is invalid
        """
        if not GOOGLE_AVAILABLE:
            raise ImportError(
                "google-generativeai package is not installed. Install with: pip install google-generativeai"
            )
        
        super().__init__(config)
        
        if config.type != ProviderType.GOOGLE:
            raise ValueError(f"Invalid provider type: {config.type}. Expected GOOGLE")
        
        # Configure the API key
        genai.configure(api_key=config.api_key)
        
        # Model optimization for TTRPG
        self.model_mapping = {
            'fast': 'gemini-1.5-flash',      # Fast, efficient responses
            'balanced': 'gemini-1.5-pro',   # Balanced performance
            'powerful': 'gemini-1.5-pro',   # Same as balanced for now
            'vision': 'gemini-1.5-pro'    # Multi-modal support (1.5-pro has vision)
        }
        
        # Use centralized services
        self.pricing_manager = get_pricing_manager()
        self.rate_limiter = RateLimiter()
        
        # Safety settings optimized for TTRPG content
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        }
        
        
        # Initialize model (will be set during validation)
        self.model = None
        
        logger.info(f"GoogleProvider initialized with model mapping: {self.model_mapping}")
    
    async def complete(
        self, 
        messages: List[Dict[str, str]], 
        tools: Optional[List[Dict]] = None,
        stream: bool = False
    ) -> AsyncIterator[str]:
        """
        Generate completion from Google Gemini.
        
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
            await self.rate_limiter.acquire(ProviderType.GOOGLE, "default_user", priority=5)
            
            # Convert messages to Gemini format
            gemini_messages = self._convert_messages_to_gemini(messages)
            
            # Get model
            model_name = self.config.model or self.model_mapping.get('balanced', 'gemini-1.5-pro')
            
            if not self.model or self.model.model_name != model_name:
                self.model = genai.GenerativeModel(
                    model_name=model_name,
                    safety_settings=self.safety_settings,
                    system_instruction=self.get_ttrpg_system_prompt()
                )
            
            # Prepare generation config
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )
            
            # Convert tools if provided
            if tools:
                # Google's function calling format is different
                logger.warning("Tool/function calling not yet fully implemented for Google Gemini")
            
            # Make request
            start_time = time.time()
            
            try:
                if stream:
                    # Streaming response
                    response = await asyncio.wait_for(
                        self._generate_stream_async(gemini_messages, generation_config),
                        timeout=self.config.timeout
                    )
                    
                    async for chunk in response:
                        if chunk.text:
                            yield chunk.text
                else:
                    # Non-streaming response
                    response = await asyncio.wait_for(
                        self._generate_content_async(gemini_messages, generation_config),
                        timeout=self.config.timeout
                    )
                    
                    if response.text:
                        yield response.text
                        
            except asyncio.TimeoutError:
                elapsed = time.time() - start_time
                raise ProviderTimeoutError(
                    f"Google Gemini request timed out after {elapsed:.1f}s",
                    provider="google",
                    timeout_seconds=elapsed
                )
                    
        except ProviderError:
            raise
        except Exception as e:
            # Record failure for rate limiting
            is_rate_limit = "rate" in str(e).lower() or "quota" in str(e).lower() or "429" in str(e)
            self.rate_limiter.record_failure(ProviderType.GOOGLE, "default_user", is_rate_limit)
            await self._handle_google_error(e)
        else:
            # Record success for adaptive rate limiting
            self.rate_limiter.record_success(ProviderType.GOOGLE, "default_user")
    
    async def validate_credentials(self) -> bool:
        """
        Validate Google Gemini API credentials.
        
        Returns:
            bool: True if credentials are valid
            
        Raises:
            ProviderAuthenticationError: If credentials are invalid
        """
        try:
            # Test with a minimal request
            model_name = self.model_mapping['fast']  # Use fastest model for validation
            test_model = genai.GenerativeModel(model_name)
            
            start_time = time.time()
            response = await asyncio.wait_for(
                self._generate_content_async(
                    [{"role": "user", "parts": [{"text": "Hi"}]}],
                    genai.types.GenerationConfig(max_output_tokens=5, temperature=0.0),
                    test_model
                ),
                timeout=10.0  # Short timeout for validation
            )
            
            if hasattr(response, 'text'):
                self._authenticated = True
                elapsed = time.time() - start_time
                logger.info(f"Google Gemini credentials validated successfully in {elapsed:.2f}s")
                return True
            else:
                raise ProviderAuthenticationError(
                    "Invalid response from Google Gemini API during validation",
                    provider="google"
                )
                
        except asyncio.TimeoutError:
            raise ProviderAuthenticationError(
                "Google Gemini API validation timed out",
                provider="google"
            )
        except Exception as e:
            logger.error(f"Google Gemini credential validation failed: {e}")
            raise ProviderAuthenticationError(
                f"Google Gemini API key validation failed: {str(e)}",
                provider="google"
            )
    
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """
        Estimate cost for Google Gemini request using centralized pricing.
        
        Args:
            input_tokens: Number of input tokens
            output_tokens: Expected number of output tokens
            
        Returns:
            float: Estimated cost in USD
        """
        model = self.config.model or self.model_mapping.get('balanced', 'gemini-1.5-pro')
        return self.pricing_manager.calculate_cost(
            ProviderType.GOOGLE, model, input_tokens, output_tokens
        )
    
    async def health_check(self) -> bool:
        """
        Check if Google Gemini service is healthy.
        
        Returns:
            bool: True if service is healthy
        """
        try:
            # Simple health check with minimal request
            model_name = self.model_mapping['fast']
            test_model = genai.GenerativeModel(model_name)
            
            start_time = time.time()
            
            response = await asyncio.wait_for(
                self._generate_content_async(
                    [{"role": "user", "parts": [{"text": "ping"}]}],
                    genai.types.GenerationConfig(max_output_tokens=1, temperature=0.0),
                    test_model
                ),
                timeout=5.0
            )
            
            elapsed = time.time() - start_time
            self._last_health_check = datetime.now()
            
            logger.debug(f"Google Gemini health check passed in {elapsed:.2f}s")
            return True
            
        except Exception as e:
            logger.warning(f"Google Gemini health check failed: {e}")
            return False
    
    def _convert_messages_to_gemini(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Convert generic message format to Gemini's format.
        
        Args:
            messages: Messages in generic format
            
        Returns:
            List[Dict[str, str]]: Messages in Gemini format
        """
        gemini_messages = []
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            # Convert role to Gemini format
            if role == "assistant":
                gemini_role = "model"
            elif role == "system":
                # System messages are handled via system_instruction in model
                continue
            else:
                gemini_role = "user"
            
            gemini_messages.append({
                "role": gemini_role,
                "parts": [{"text": content}]
            })
        
        return gemini_messages
    
    async def _generate_content_async(self, messages: List[Dict], generation_config, model=None):
        """Generate content using the Gemini model with native async methods."""
        if model is None:
            model = self.model
        
        # Use native async methods from google-generativeai library
        # For now, we'll use asyncio.to_thread as a bridge until native async is available
        # TODO: Replace with native async methods when google-generativeai adds them
        
        # Convert messages to chat format
        if len(messages) == 1:
            # Single message
            return await asyncio.to_thread(
                model.generate_content,
                messages[0]["parts"][0]["text"],
                generation_config=generation_config
            )
        else:
            # Multi-message chat
            chat = model.start_chat()
            for msg in messages[:-1]:
                await asyncio.to_thread(
                    chat.send_message,
                    msg["parts"][0]["text"]
                )
            
            return await asyncio.to_thread(
                chat.send_message,
                messages[-1]["parts"][0]["text"],
                generation_config=generation_config
            )
    
    async def _generate_stream_async(self, messages: List[Dict], generation_config):
        """Generate streaming content using the Gemini model with native async methods."""
        # Use native async methods from google-generativeai library
        # For now, we'll use asyncio.to_thread as a bridge until native async is available
        # TODO: Replace with native async methods when google-generativeai adds them
        
        # Convert messages to chat format
        if len(messages) == 1:
            # Single message streaming
            response = await asyncio.to_thread(
                self.model.generate_content,
                messages[0]["parts"][0]["text"],
                generation_config=generation_config,
                stream=True
            )
        else:
            # Multi-message chat streaming
            chat = self.model.start_chat()
            for msg in messages[:-1]:
                await asyncio.to_thread(
                    chat.send_message,
                    msg["parts"][0]["text"]
                )
            
            response = await asyncio.to_thread(
                chat.send_message,
                messages[-1]["parts"][0]["text"],
                generation_config=generation_config,
                stream=True
            )
        
        # Convert to async iterator
        for chunk in response:
            yield chunk
    
    
    async def _handle_google_error(self, error: Exception) -> None:
        """
        Handle and convert Google Gemini-specific errors.
        
        Args:
            error: The exception that occurred
            
        Raises:
            ProviderError: Converted error with appropriate type
        """
        error_str = str(error).lower()
        
        if "quota" in error_str or "rate" in error_str or "429" in error_str:
            raise ProviderRateLimitError(
                f"Google Gemini rate limit exceeded: {error}",
                provider="google"
            )
        
        elif "authentication" in error_str or "unauthorized" in error_str or "401" in error_str or "api_key" in error_str:
            self._authenticated = False
            raise ProviderAuthenticationError(
                f"Google Gemini authentication failed: {error}",
                provider="google"
            )
        
        elif "billing" in error_str or "payment" in error_str:
            raise ProviderQuotaExceededError(
                f"Google Gemini quota exceeded: {error}",
                provider="google"
            )
        
        elif "invalid" in error_str or "400" in error_str or "malformed" in error_str:
            raise ProviderInvalidRequestError(
                f"Invalid request to Google Gemini: {error}",
                provider="google"
            )
        
        elif "safety" in error_str or "blocked" in error_str:
            raise ProviderInvalidRequestError(
                f"Content blocked by Google Gemini safety filters: {error}",
                provider="google",
                validation_errors=["Content blocked by safety filters"]
            )
        
        else:
            raise ProviderError(
                f"Google Gemini API error: {error}",
                provider="google"
            )