"""Abstract base class for AI providers."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List

from structlog import get_logger

from .token_estimator import TokenEstimator
from .models import (
    AIRequest,
    AIResponse,
    ModelSpec,
    ProviderCapability,
    ProviderConfig,
    ProviderHealth,
    ProviderStatus,
    StreamingChunk,
)

logger = get_logger(__name__)


class AbstractProvider(ABC):
    """Abstract base class for all AI providers.
    
    This class defines the interface that all AI providers must implement
    to work with the MCP Bridge AI Provider Integration system.
    """
    
    def __init__(self, config: ProviderConfig):
        """Initialize the provider with configuration.
        
        Args:
            config: Provider configuration including API keys and settings
        """
        self.config = config
        self.provider_type = config.provider_type
        self._health = ProviderHealth(
            provider_type=self.provider_type,
            status=ProviderStatus.AVAILABLE,
        )
        self._models: Dict[str, ModelSpec] = {}
        self._client = None
        self._initialized = False
        self._token_estimator = TokenEstimator()
        
    @property
    def health(self) -> ProviderHealth:
        """Get the current health status of the provider."""
        return self._health
        
    @property
    def models(self) -> Dict[str, ModelSpec]:
        """Get available models for this provider."""
        return self._models
        
    @property
    def is_available(self) -> bool:
        """Check if the provider is currently available."""
        return (
            self.config.enabled 
            and self._health.status == ProviderStatus.AVAILABLE
            and self._initialized
        )
    
    async def initialize(self) -> None:
        """Initialize the provider and load available models.
        
        This method should be called before using the provider.
        It sets up the client connection and loads model information.
        """
        if self._initialized:
            return
            
        try:
            logger.info("Initializing AI provider", provider=self.provider_type.value)
            
            # Initialize the client
            await self._initialize_client()
            
            # Load available models
            await self._load_models()
            
            # Update health status
            self._health.status = ProviderStatus.AVAILABLE
            self._health.updated_at = datetime.now()
            
            self._initialized = True
            
            logger.info(
                "AI provider initialized successfully",
                provider=self.provider_type.value,
                models=len(self._models),
            )
            
        except Exception as e:
            logger.error(
                "Failed to initialize AI provider",
                provider=self.provider_type.value,
                error=str(e),
            )
            self._health.status = ProviderStatus.ERROR
            self._health.last_error = datetime.now()
            self._health.error_count += 1
            raise
    
    async def shutdown(self) -> None:
        """Shutdown the provider and cleanup resources."""
        if not self._initialized:
            return
            
        try:
            logger.info("Shutting down AI provider", provider=self.provider_type.value)
            
            await self._cleanup_client()
            
            self._initialized = False
            self._health.status = ProviderStatus.UNAVAILABLE
            self._health.updated_at = datetime.now()
            
        except Exception as e:
            logger.error(
                "Error during provider shutdown",
                provider=self.provider_type.value,
                error=str(e),
            )
    
    async def generate_response(
        self,
        request: AIRequest,
    ) -> AIResponse:
        """Generate a response to the given request.
        
        Args:
            request: The AI request to process
            
        Returns:
            AIResponse: The generated response
            
        Raises:
            Exception: If the request fails
        """
        if not self.is_available:
            raise RuntimeError(f"Provider {self.provider_type.value} is not available")
            
        start_time = datetime.now()
        
        try:
            # Validate the request
            await self._validate_request(request)
            
            # Check rate limits
            await self._check_rate_limits()
            
            # Generate response
            response = await self._generate_response_impl(request)
            
            # Update health metrics
            latency_ms = (datetime.now() - start_time).total_seconds() * 1000
            self._update_health_success(latency_ms)
            
            return response
            
        except Exception as e:
            # Update health metrics
            self._update_health_error(str(e))
            raise
    
    async def stream_response(
        self,
        request: AIRequest,
    ) -> AsyncGenerator[StreamingChunk, None]:
        """Stream a response to the given request.
        
        Args:
            request: The AI request to process
            
        Yields:
            StreamingChunk: Chunks of the streaming response
            
        Raises:
            Exception: If streaming is not supported or the request fails
        """
        if not self.is_available:
            raise RuntimeError(f"Provider {self.provider_type.value} is not available")
            
        if not self.supports_streaming(request.model):
            raise ValueError(f"Model {request.model} does not support streaming")
            
        start_time = datetime.now()
        
        try:
            # Validate the request
            await self._validate_request(request)
            
            # Check rate limits
            await self._check_rate_limits()
            
            # Stream response
            async for chunk in self._stream_response_impl(request):
                yield chunk
            
            # Update health metrics
            latency_ms = (datetime.now() - start_time).total_seconds() * 1000
            self._update_health_success(latency_ms)
            
        except Exception as e:
            # Update health metrics  
            self._update_health_error(str(e))
            raise
    
    def supports_capability(self, capability: ProviderCapability) -> bool:
        """Check if the provider supports a specific capability.
        
        Args:
            capability: The capability to check
            
        Returns:
            bool: True if the capability is supported
        """
        return capability in self._get_supported_capabilities()
    
    def supports_streaming(self, model: str) -> bool:
        """Check if a model supports streaming.
        
        Args:
            model: The model to check
            
        Returns:
            bool: True if streaming is supported
        """
        if model not in self._models:
            return False
        return self._models[model].supports_streaming
    
    def supports_tools(self, model: str) -> bool:
        """Check if a model supports tool calling.
        
        Args:
            model: The model to check
            
        Returns:
            bool: True if tool calling is supported
        """
        if model not in self._models:
            return False
        return self._models[model].supports_tools
    
    def get_model_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate the cost for a request.
        
        Args:
            model: The model used
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            
        Returns:
            float: Cost in USD
        """
        if model not in self._models:
            return 0.0
            
        model_spec = self._models[model]
        input_cost = (input_tokens / 1000) * model_spec.cost_per_input_token
        output_cost = (output_tokens / 1000) * model_spec.cost_per_output_token
        
        return input_cost + output_cost
    
    async def health_check(self) -> ProviderHealth:
        """Perform a health check on the provider.
        
        Returns:
            ProviderHealth: Current health status
        """
        try:
            await self._perform_health_check()
            return self._health
        except Exception as e:
            logger.error(
                "Health check failed",
                provider=self.provider_type.value,
                error=str(e),
            )
            self._health.status = ProviderStatus.ERROR
            self._health.last_error = datetime.now()
            self._health.error_count += 1
            return self._health
    
    # Abstract methods that must be implemented by subclasses
    
    @abstractmethod
    async def _initialize_client(self) -> None:
        """Initialize the provider-specific client."""
        pass
    
    @abstractmethod
    async def _cleanup_client(self) -> None:
        """Clean up the provider-specific client."""
        pass
    
    @abstractmethod
    async def _load_models(self) -> None:
        """Load available models for this provider."""
        pass
    
    @abstractmethod
    async def _generate_response_impl(self, request: AIRequest) -> AIResponse:
        """Provider-specific response generation implementation."""
        pass
    
    @abstractmethod
    async def _stream_response_impl(
        self, request: AIRequest
    ) -> AsyncGenerator[StreamingChunk, None]:
        """Provider-specific streaming response implementation."""
        pass
    
    @abstractmethod
    def _get_supported_capabilities(self) -> List[ProviderCapability]:
        """Get list of supported capabilities for this provider."""
        pass
    
    @abstractmethod
    async def _perform_health_check(self) -> None:
        """Perform provider-specific health check."""
        pass
    
    # Helper methods
    
    async def _validate_request(self, request: AIRequest) -> None:
        """Validate an AI request.
        
        Args:
            request: The request to validate
            
        Raises:
            ValueError: If the request is invalid
        """
        if not request.model:
            raise ValueError("Model is required")
            
        if request.model not in self._models:
            raise ValueError(f"Model {request.model} is not available")
            
        if not request.messages:
            raise ValueError("Messages are required")
        
        # Check token limits
        model_spec = self._models[request.model]
        estimated_tokens = self._estimate_tokens(request.messages)
        
        if estimated_tokens > model_spec.context_length:
            raise ValueError(
                f"Request exceeds context length: {estimated_tokens} > {model_spec.context_length}"
            )
    
    async def _check_rate_limits(self) -> None:
        """Check if rate limits allow the request.
        
        Raises:
            Exception: If rate limits are exceeded
        """
        if self._health.status == ProviderStatus.RATE_LIMITED:
            if self._health.rate_limit_reset and datetime.now() < self._health.rate_limit_reset:
                raise RuntimeError("Rate limit exceeded, please wait")
            else:
                # Reset rate limit status
                self._health.status = ProviderStatus.AVAILABLE
    
    def _estimate_tokens(self, messages: List[Dict[str, Any]]) -> int:
        """Estimate token count for messages.
        
        Args:
            messages: List of messages
            
        Returns:
            int: Estimated token count
        """
        # Use advanced token estimator for accurate counts
        # Get the model from the first available model if not specified
        model = None
        if self._models:
            model = next(iter(self._models.keys()))
        
        return self._token_estimator.estimate_tokens(
            content=messages,
            provider_type=self.provider_type,
            model=model
        )
    
    def _update_health_success(self, latency_ms: float) -> None:
        """Update health metrics after a successful request."""
        self._health.last_success = datetime.now()
        self._health.success_count += 1
        
        # Update average latency
        if self._health.avg_latency_ms == 0:
            self._health.avg_latency_ms = latency_ms
        else:
            # Simple moving average
            self._health.avg_latency_ms = (
                self._health.avg_latency_ms * 0.9 + latency_ms * 0.1
            )
        
        # Update uptime percentage
        total_requests = self._health.success_count + self._health.error_count
        self._health.uptime_percentage = (self._health.success_count / total_requests) * 100
        
        self._health.updated_at = datetime.now()
    
    def _update_health_error(self, error_message: str) -> None:
        """Update health metrics after a failed request."""
        self._health.last_error = datetime.now()
        self._health.error_count += 1
        
        # Update uptime percentage
        total_requests = self._health.success_count + self._health.error_count
        if total_requests > 0:
            self._health.uptime_percentage = (self._health.success_count / total_requests) * 100
        
        # Update status based on error pattern
        if self._health.error_count >= 5:
            recent_errors = self._health.error_count
            recent_successes = self._health.success_count
            
            if recent_errors > recent_successes:
                self._health.status = ProviderStatus.ERROR
        
        self._health.updated_at = datetime.now()
    
    def __str__(self) -> str:
        """String representation of the provider."""
        return f"{self.provider_type.value.title()}Provider({self._health.status.value})"
    
    def __repr__(self) -> str:
        """Detailed representation of the provider."""
        return (
            f"{self.__class__.__name__}("
            f"provider_type={self.provider_type.value}, "
            f"status={self._health.status.value}, "
            f"models={len(self._models)}, "
            f"initialized={self._initialized}"
            f")"
        )