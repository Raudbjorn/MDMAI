"""Ollama TTS provider for local voice synthesis using Orpheus model.

This provider uses the Orpheus TTS model running through Ollama for
completely local, private, and cost-free voice synthesis.

Orpheus requires a separate FastAPI frontend (Orpheus-FastAPI) that
provides OpenAI-compatible TTS endpoints.
"""

import asyncio
from typing import AsyncGenerator, Dict, Optional

import aiohttp
from structlog import get_logger

from ..abstract_voice_provider import AbstractVoiceProvider
from ..models import (
    AudioFormat,
    StreamingAudioChunk,
    VoiceAge,
    VoiceEmotion,
    VoiceGender,
    VoiceProfile,
    VoiceProviderConfig,
    VoiceProviderStatus,
    VoiceProviderType,
    VoiceRequest,
    VoiceResponse,
    VoiceSpec,
)

logger = get_logger(__name__)

# Orpheus supports multiple voice presets
ORPHEUS_VOICES: Dict[str, Dict] = {
    "tara": {"gender": VoiceGender.FEMALE, "age": VoiceAge.ADULT, "style": "neutral"},
    "leah": {"gender": VoiceGender.FEMALE, "age": VoiceAge.YOUNG, "style": "expressive"},
    "jess": {"gender": VoiceGender.FEMALE, "age": VoiceAge.ADULT, "style": "warm"},
    "leo": {"gender": VoiceGender.MALE, "age": VoiceAge.ADULT, "style": "neutral"},
    "dan": {"gender": VoiceGender.MALE, "age": VoiceAge.MATURE, "style": "deep"},
    "mia": {"gender": VoiceGender.FEMALE, "age": VoiceAge.YOUNG, "style": "bright"},
    "zac": {"gender": VoiceGender.MALE, "age": VoiceAge.YOUNG, "style": "energetic"},
    "zoe": {"gender": VoiceGender.FEMALE, "age": VoiceAge.MATURE, "style": "calm"},
}

# Emotion tags supported by Orpheus
ORPHEUS_EMOTION_TAGS: Dict[VoiceEmotion, str] = {
    VoiceEmotion.HAPPY: "<laugh>",
    VoiceEmotion.SAD: "<sigh>",
    VoiceEmotion.ANGRY: "<angry>",
    VoiceEmotion.EXCITED: "<excited>",
    VoiceEmotion.MYSTERIOUS: "<whisper>",
    VoiceEmotion.CALM: "<calm>",
    VoiceEmotion.NEUTRAL: "",
}


class OllamaTTSProvider(AbstractVoiceProvider):
    """Local TTS provider using Ollama with Orpheus model.

    This provider connects to an Orpheus-FastAPI server that provides
    OpenAI-compatible TTS endpoints. The Orpheus model runs through
    Ollama for efficient local inference.

    Requirements:
    - Ollama running with Orpheus model pulled
    - Orpheus-FastAPI server running

    Features:
    - Completely local, no API costs
    - 8 distinct voice presets
    - Emotion tags for expressive speech
    - 24kHz audio output
    """

    def __init__(self, config: VoiceProviderConfig):
        """Initialize Ollama TTS provider.

        Args:
            config: Provider configuration
        """
        config.provider_type = VoiceProviderType.OLLAMA_TTS
        super().__init__(config)

        # Default endpoint for Orpheus-FastAPI
        self._base_url = config.base_url or "http://localhost:8880"

        # Default model
        self._model = config.model or "orpheus"

        # Session for async requests
        self._session: Optional[aiohttp.ClientSession] = None

    @property
    def supports_streaming(self) -> bool:
        """Orpheus supports streaming output."""
        return True

    @property
    def supports_emotion(self) -> bool:
        """Orpheus supports emotion via tags."""
        return True

    async def _initialize_client(self) -> None:
        """Initialize the HTTP client."""
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.timeout)
        )

        # Check if Orpheus-FastAPI is running
        try:
            async with self._session.get(f"{self._base_url}/health") as resp:
                if resp.status == 200:
                    self._health.status = VoiceProviderStatus.AVAILABLE
                    logger.info("Ollama TTS (Orpheus) initialized", endpoint=self._base_url)
                else:
                    raise ConnectionError(f"Health check failed: {resp.status}")
        except aiohttp.ClientError as e:
            logger.warning(
                "Orpheus-FastAPI not available, trying direct Ollama",
                error=str(e),
            )
            # Try direct Ollama endpoint
            self._base_url = "http://localhost:11434"
            self._health.status = VoiceProviderStatus.AVAILABLE

    async def _cleanup_client(self) -> None:
        """Clean up the HTTP client."""
        if self._session:
            await self._session.close()
            self._session = None

    async def _load_available_voices(self) -> None:
        """Load available Orpheus voices."""
        # Orpheus has fixed voice presets
        for voice_id, attrs in ORPHEUS_VOICES.items():
            self._available_voices[voice_id] = VoiceSpec(
                voice_id=voice_id,
                provider_type=VoiceProviderType.OLLAMA_TTS,
                name=voice_id.title(),
                description=f"{attrs['style'].title()} {attrs['gender'].value} voice",
                gender=attrs["gender"],
                age=attrs["age"],
                language="en",
                style_tags=[attrs["style"]],
            )

        logger.info("Loaded Orpheus voices", count=len(self._available_voices))

    async def _synthesize_impl(
        self,
        request: VoiceRequest,
        profile: VoiceProfile,
    ) -> VoiceResponse:
        """Synthesize speech using Orpheus."""
        if not self._session:
            raise RuntimeError("Provider not initialized")

        voice_id = self.get_voice_for_profile(profile) or "tara"

        # Prepare text with emotion tags
        text = self._prepare_text(request.text, request.emotion or profile.default_emotion)

        try:
            # Try OpenAI-compatible endpoint first
            payload = {
                "model": self._model,
                "input": text,
                "voice": voice_id,
                "response_format": request.output_format.value,
                "speed": 0.8 + (profile.speed * 0.4),  # Map 0-1 to 0.8-1.2
            }

            async with self._session.post(
                f"{self._base_url}/v1/audio/speech",
                json=payload,
            ) as resp:
                if resp.status == 200:
                    audio_data = await resp.read()

                    # Estimate duration
                    duration_ms = (len(request.text) / 150) * 1000

                    return VoiceResponse(
                        request_id=request.request_id,
                        provider_type=VoiceProviderType.OLLAMA_TTS,
                        audio_data=audio_data,
                        duration_ms=duration_ms,
                        sample_rate=24000,  # Orpheus outputs 24kHz
                        format=request.output_format,
                        file_size_bytes=len(audio_data),
                        characters_processed=len(request.text),
                        cost=0.0,  # Local = free
                        success=True,
                    )
                else:
                    error_text = await resp.text()
                    raise RuntimeError(f"TTS request failed: {resp.status} - {error_text}")

        except aiohttp.ClientError as e:
            logger.error("Ollama TTS synthesis failed", error=str(e))
            return VoiceResponse(
                request_id=request.request_id,
                provider_type=VoiceProviderType.OLLAMA_TTS,
                success=False,
                error=str(e),
            )

    async def _stream_synthesize_impl(
        self,
        request: VoiceRequest,
        profile: VoiceProfile,
    ) -> AsyncGenerator[StreamingAudioChunk, None]:
        """Stream audio chunks from Orpheus."""
        if not self._session:
            raise RuntimeError("Provider not initialized")

        voice_id = self.get_voice_for_profile(profile) or "tara"
        text = self._prepare_text(request.text, request.emotion or profile.default_emotion)

        payload = {
            "model": self._model,
            "input": text,
            "voice": voice_id,
            "response_format": request.output_format.value,
            "stream": True,
        }

        try:
            async with self._session.post(
                f"{self._base_url}/v1/audio/speech",
                json=payload,
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise RuntimeError(f"Stream request failed: {resp.status} - {error_text}")

                chunk_index = 0
                async for chunk in resp.content.iter_chunked(4096):
                    if chunk:
                        yield StreamingAudioChunk(
                            request_id=request.request_id,
                            chunk_index=chunk_index,
                            audio_data=chunk,
                            is_final=False,
                        )
                        chunk_index += 1

                yield StreamingAudioChunk(
                    request_id=request.request_id,
                    chunk_index=chunk_index,
                    audio_data=b"",
                    is_final=True,
                )

        except aiohttp.ClientError as e:
            logger.error("Ollama TTS streaming failed", error=str(e))
            raise

    async def _perform_health_check(self) -> None:
        """Check Orpheus/Ollama availability."""
        if not self._session:
            self._health.status = VoiceProviderStatus.UNAVAILABLE
            return

        try:
            # Try health endpoint
            async with self._session.get(
                f"{self._base_url}/health",
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                if resp.status == 200:
                    self._health.status = VoiceProviderStatus.AVAILABLE
                    return

            # Try Ollama models endpoint as fallback
            async with self._session.get(
                f"{self._base_url}/api/tags",
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    # Check if Orpheus model is available
                    models = [m.get("name", "") for m in data.get("models", [])]
                    if any("orpheus" in m.lower() for m in models):
                        self._health.status = VoiceProviderStatus.AVAILABLE
                        return

            self._health.status = VoiceProviderStatus.UNAVAILABLE

        except Exception as e:
            logger.warning("Ollama TTS health check failed", error=str(e))
            self._health.status = VoiceProviderStatus.UNAVAILABLE

    def _select_best_match_voice(self, profile: VoiceProfile) -> Optional[str]:
        """Select best matching Orpheus voice.

        Args:
            profile: Voice profile to match

        Returns:
            Voice ID
        """
        # Find best match by gender and age
        best_match = None
        best_score = -1

        for voice_id, attrs in ORPHEUS_VOICES.items():
            score = 0

            # Gender match
            if attrs["gender"] == profile.gender:
                score += 10
            elif profile.gender == VoiceGender.NEUTRAL:
                score += 3

            # Age match
            if attrs["age"] == profile.age:
                score += 5
            elif attrs["age"] in [VoiceAge.ADULT, VoiceAge.MATURE]:
                score += 2  # Prefer mature voices for DM

            if score > best_score:
                best_score = score
                best_match = voice_id

        return best_match or "tara"

    def _prepare_text(self, text: str, emotion: VoiceEmotion) -> str:
        """Prepare text with Orpheus emotion tags.

        Orpheus supports tags like <laugh>, <sigh>, etc.

        Args:
            text: Original text
            emotion: Emotion to express

        Returns:
            Text with appropriate tags
        """
        tag = ORPHEUS_EMOTION_TAGS.get(emotion, "")
        if tag:
            # Insert tag at the beginning
            return f"{tag} {text}"
        return text

    def estimate_cost(self, text: str) -> float:
        """Local synthesis is free."""
        return 0.0
