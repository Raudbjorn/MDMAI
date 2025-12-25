"""Abstract base class for voice synthesis providers."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional

from structlog import get_logger

from .models import (
    AudioFormat,
    StreamingAudioChunk,
    VoiceProfile,
    VoiceProviderConfig,
    VoiceProviderHealth,
    VoiceProviderStatus,
    VoiceProviderType,
    VoiceRequest,
    VoiceResponse,
    VoiceSpec,
)

logger = get_logger(__name__)


class AbstractVoiceProvider(ABC):
    """Abstract base class for all voice synthesis providers.

    This class defines the interface that all voice providers must implement
    to work with the MDMAI voice synthesis system.

    Attributes:
        config: Provider configuration including API keys and settings
        provider_type: The type of this provider
        is_local: Whether this provider runs locally (no API costs)
    """

    def __init__(self, config: VoiceProviderConfig):
        """Initialize the provider with configuration.

        Args:
            config: Provider configuration
        """
        self.config = config
        self.provider_type = config.provider_type
        self._health = VoiceProviderHealth(
            provider_type=self.provider_type,
            status=VoiceProviderStatus.INITIALIZING,
        )
        self._available_voices: Dict[str, VoiceSpec] = {}
        self._client: Any = None
        self._initialized = False

    @property
    def health(self) -> VoiceProviderHealth:
        """Get the current health status of the provider."""
        return self._health

    @property
    def available_voices(self) -> Dict[str, VoiceSpec]:
        """Get available voices for this provider."""
        return self._available_voices

    @property
    def is_available(self) -> bool:
        """Check if the provider is currently available."""
        return (
            self.config.enabled
            and self._health.status == VoiceProviderStatus.AVAILABLE
            and self._initialized
        )

    @property
    def is_local(self) -> bool:
        """Whether this provider runs locally (no API costs)."""
        return self.provider_type in [
            VoiceProviderType.OLLAMA_TTS,
            VoiceProviderType.COQUI,
            VoiceProviderType.PIPER,
            VoiceProviderType.SYSTEM,
        ]

    @property
    @abstractmethod
    def supports_streaming(self) -> bool:
        """Whether provider supports streaming synthesis."""
        pass

    @property
    @abstractmethod
    def supports_emotion(self) -> bool:
        """Whether provider supports emotion control."""
        pass

    async def initialize(self) -> None:
        """Initialize the provider and load available voices.

        This method should be called before using the provider.
        It sets up the client connection and loads voice information.
        """
        if self._initialized:
            return

        try:
            logger.info("Initializing voice provider", provider=self.provider_type.value)

            # Initialize the client
            await self._initialize_client()

            # Load available voices
            await self._load_available_voices()

            # Update health status
            self._health.status = VoiceProviderStatus.AVAILABLE
            self._health.updated_at = datetime.utcnow()

            self._initialized = True

            logger.info(
                "Voice provider initialized successfully",
                provider=self.provider_type.value,
                voices=len(self._available_voices),
            )

        except Exception as e:
            logger.error(
                "Failed to initialize voice provider",
                provider=self.provider_type.value,
                error=str(e),
            )
            self._health.status = VoiceProviderStatus.ERROR
            self._health.last_error = datetime.utcnow()
            self._health.error_count += 1
            raise

    async def shutdown(self) -> None:
        """Shutdown the provider and cleanup resources."""
        if not self._initialized:
            return

        try:
            logger.info("Shutting down voice provider", provider=self.provider_type.value)

            await self._cleanup_client()

            self._initialized = False
            self._health.status = VoiceProviderStatus.UNAVAILABLE
            self._health.updated_at = datetime.utcnow()

        except Exception as e:
            logger.error(
                "Error during voice provider shutdown",
                provider=self.provider_type.value,
                error=str(e),
            )

    async def synthesize(
        self,
        request: VoiceRequest,
        profile: VoiceProfile,
    ) -> VoiceResponse:
        """Synthesize speech from text.

        Args:
            request: The voice synthesis request
            profile: The voice profile to use

        Returns:
            VoiceResponse with audio data

        Raises:
            RuntimeError: If provider is not available
            ValueError: If request is invalid
        """
        if not self.is_available:
            raise RuntimeError(f"Provider {self.provider_type.value} is not available")

        start_time = datetime.utcnow()

        try:
            # Validate the request
            self._validate_request(request)

            # Check rate limits
            await self._check_rate_limits()

            # Perform synthesis
            response = await self._synthesize_impl(request, profile)

            # Update metrics
            latency_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            response.latency_ms = latency_ms
            self._update_health_success(latency_ms)

            return response

        except Exception as e:
            self._update_health_error(str(e))
            raise

    async def stream_synthesize(
        self,
        request: VoiceRequest,
        profile: VoiceProfile,
    ) -> AsyncGenerator[StreamingAudioChunk, None]:
        """Stream audio chunks for real-time playback.

        Args:
            request: The voice synthesis request
            profile: The voice profile to use

        Yields:
            StreamingAudioChunk instances with audio data

        Raises:
            RuntimeError: If provider is not available
            NotImplementedError: If streaming is not supported
        """
        if not self.is_available:
            raise RuntimeError(f"Provider {self.provider_type.value} is not available")

        if not self.supports_streaming:
            raise NotImplementedError(
                f"Provider {self.provider_type.value} does not support streaming"
            )

        start_time = datetime.utcnow()

        try:
            # Validate the request
            self._validate_request(request)

            # Check rate limits
            await self._check_rate_limits()

            # Stream synthesis
            async for chunk in self._stream_synthesize_impl(request, profile):
                yield chunk

            # Update metrics
            latency_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            self._update_health_success(latency_ms)

        except Exception as e:
            self._update_health_error(str(e))
            raise

    async def health_check(self) -> VoiceProviderHealth:
        """Perform a health check on the provider.

        Returns:
            Current health status
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
            self._health.status = VoiceProviderStatus.ERROR
            self._health.last_error = datetime.utcnow()
            self._health.error_count += 1
            return self._health

    def get_voice_for_profile(self, profile: VoiceProfile) -> Optional[str]:
        """Get the provider-specific voice ID for a profile.

        First checks explicit mapping, then auto-selects based on characteristics.

        Args:
            profile: The voice profile

        Returns:
            Provider-specific voice ID, or None if no match
        """
        # Check for explicit mapping first
        if self.provider_type in profile.provider_voice_ids:
            voice_id = profile.provider_voice_ids[self.provider_type]
            if voice_id in self._available_voices:
                return voice_id

        # Auto-select based on profile characteristics
        return self._select_best_match_voice(profile)

    def estimate_cost(self, text: str) -> float:
        """Estimate cost for synthesizing text.

        Args:
            text: Text to synthesize

        Returns:
            Estimated cost in USD
        """
        if self.is_local:
            return 0.0

        char_count = len(text)
        return char_count * self.config.cost_per_character

    def get_supported_formats(self) -> List[AudioFormat]:
        """Get supported audio output formats."""
        return self.config.supported_formats

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
    async def _load_available_voices(self) -> None:
        """Load available voices for this provider."""
        pass

    @abstractmethod
    async def _synthesize_impl(
        self,
        request: VoiceRequest,
        profile: VoiceProfile,
    ) -> VoiceResponse:
        """Provider-specific synthesis implementation.

        Args:
            request: The voice request
            profile: The voice profile

        Returns:
            VoiceResponse with audio data
        """
        pass

    @abstractmethod
    async def _stream_synthesize_impl(
        self,
        request: VoiceRequest,
        profile: VoiceProfile,
    ) -> AsyncGenerator[StreamingAudioChunk, None]:
        """Provider-specific streaming synthesis implementation.

        Args:
            request: The voice request
            profile: The voice profile

        Yields:
            StreamingAudioChunk instances
        """
        pass

    @abstractmethod
    async def _perform_health_check(self) -> None:
        """Perform provider-specific health check."""
        pass

    @abstractmethod
    def _select_best_match_voice(self, profile: VoiceProfile) -> Optional[str]:
        """Select the best matching voice for a profile.

        Args:
            profile: The voice profile to match

        Returns:
            Voice ID or None if no match
        """
        pass

    # Helper methods

    def _validate_request(self, request: VoiceRequest) -> None:
        """Validate a voice synthesis request.

        Args:
            request: The request to validate

        Raises:
            ValueError: If the request is invalid
        """
        if not request.text:
            raise ValueError("Text is required for synthesis")

        if len(request.text) > self.config.max_text_length:
            raise ValueError(
                f"Text exceeds maximum length: {len(request.text)} > {self.config.max_text_length}"
            )

        if request.output_format not in self.config.supported_formats:
            raise ValueError(
                f"Output format {request.output_format.value} not supported. "
                f"Supported: {[f.value for f in self.config.supported_formats]}"
            )

    async def _check_rate_limits(self) -> None:
        """Check if rate limits allow the request.

        Raises:
            RuntimeError: If rate limits are exceeded
        """
        if self._health.status == VoiceProviderStatus.RATE_LIMITED:
            if (
                self._health.rate_limit_reset
                and datetime.utcnow() < self._health.rate_limit_reset
            ):
                raise RuntimeError("Rate limit exceeded, please wait")
            else:
                # Reset rate limit status
                self._health.status = VoiceProviderStatus.AVAILABLE

    def _update_health_success(self, latency_ms: float) -> None:
        """Update health metrics after a successful request."""
        self._health.last_success = datetime.utcnow()
        self._health.success_count += 1

        # Update average latency with exponential moving average
        if self._health.avg_latency_ms == 0:
            self._health.avg_latency_ms = latency_ms
        else:
            self._health.avg_latency_ms = (
                self._health.avg_latency_ms * 0.9 + latency_ms * 0.1
            )

        # Update uptime percentage
        total_requests = self._health.success_count + self._health.error_count
        if total_requests > 0:
            self._health.uptime_percentage = (
                self._health.success_count / total_requests
            ) * 100

        self._health.updated_at = datetime.utcnow()

    def _update_health_error(self, error_message: str) -> None:
        """Update health metrics after a failed request."""
        self._health.last_error = datetime.utcnow()
        self._health.error_count += 1

        # Update uptime percentage
        total_requests = self._health.success_count + self._health.error_count
        if total_requests > 0:
            self._health.uptime_percentage = (
                self._health.success_count / total_requests
            ) * 100

        # Update status based on error pattern
        if self._health.error_count >= 5:
            if self._health.error_count > self._health.success_count:
                self._health.status = VoiceProviderStatus.ERROR

        self._health.updated_at = datetime.utcnow()

        logger.warning(
            "Voice synthesis error",
            provider=self.provider_type.value,
            error=error_message,
            error_count=self._health.error_count,
        )

    def __str__(self) -> str:
        """String representation of the provider."""
        return f"{self.provider_type.value.title()}VoiceProvider({self._health.status.value})"

    def __repr__(self) -> str:
        """Detailed representation of the provider."""
        return (
            f"{self.__class__.__name__}("
            f"provider_type={self.provider_type.value}, "
            f"status={self._health.status.value}, "
            f"voices={len(self._available_voices)}, "
            f"initialized={self._initialized}"
            f")"
        )
