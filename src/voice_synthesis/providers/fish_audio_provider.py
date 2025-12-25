"""Fish Audio voice synthesis provider."""

from typing import AsyncGenerator, Dict, Optional

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

# Lazy import
FishAudio = None


def _ensure_fish_audio_import():
    """Lazy import Fish Audio SDK."""
    global FishAudio
    if FishAudio is None:
        try:
            from fish_audio_sdk import Session
            FishAudio = Session
        except ImportError:
            raise ImportError(
                "Fish Audio SDK not installed. "
                "Install with: pip install fish-audio-sdk"
            )


# Emotion markers for Fish Audio (embedded in text)
EMOTION_MARKERS: Dict[VoiceEmotion, str] = {
    VoiceEmotion.HAPPY: "(happy)",
    VoiceEmotion.SAD: "(sad)",
    VoiceEmotion.ANGRY: "(angry)",
    VoiceEmotion.FEARFUL: "(fearful)",
    VoiceEmotion.EXCITED: "(excited)",
    VoiceEmotion.MYSTERIOUS: "(mysterious)",
    VoiceEmotion.AUTHORITATIVE: "(authoritative)",
    VoiceEmotion.WHIMSICAL: "(playful)",
    VoiceEmotion.OMINOUS: "(ominous)",
    VoiceEmotion.CALM: "(calm)",
    VoiceEmotion.URGENT: "(urgent)",
    VoiceEmotion.NEUTRAL: "",  # No marker
}


class FishAudioProvider(AbstractVoiceProvider):
    """Fish Audio cloud voice synthesis provider.

    Fish Audio provides high-quality TTS with emotion control via
    text markers. Good for expressive TTRPG narration.

    Features:
    - Emotion markers embedded in text
    - WebSocket streaming
    - Voice cloning support
    """

    def __init__(self, config: VoiceProviderConfig):
        """Initialize Fish Audio provider.

        Args:
            config: Provider configuration with API key
        """
        config.provider_type = VoiceProviderType.FISH_AUDIO
        super().__init__(config)

        # Default cost per second of audio
        if config.cost_per_second == 0.0:
            self.config.cost_per_second = 0.01

        # Default voice reference
        self._default_voice_id: Optional[str] = config.metadata.get("default_voice_id")

    @property
    def supports_streaming(self) -> bool:
        """Fish Audio supports WebSocket streaming."""
        return True

    @property
    def supports_emotion(self) -> bool:
        """Fish Audio supports emotion via text markers."""
        return True

    async def _initialize_client(self) -> None:
        """Initialize the Fish Audio client."""
        _ensure_fish_audio_import()

        api_key = self.config.api_key
        if not api_key:
            raise ValueError("Fish Audio API key is required")

        self._client = FishAudio(api_key=api_key)
        self._health.status = VoiceProviderStatus.AVAILABLE

        logger.info("Fish Audio client initialized")

    async def _cleanup_client(self) -> None:
        """Clean up the Fish Audio client."""
        self._client = None

    async def _load_available_voices(self) -> None:
        """Load available voices from Fish Audio."""
        if not self._client:
            return

        try:
            # List available voice models
            voices = self._client.voice.list(page_size=100)

            for voice in voices.items:
                self._available_voices[voice.id] = VoiceSpec(
                    voice_id=voice.id,
                    provider_type=VoiceProviderType.FISH_AUDIO,
                    name=voice.name or voice.id,
                    description=voice.description or "",
                    gender=VoiceGender.NEUTRAL,  # Fish Audio doesn't provide gender
                    age=VoiceAge.ADULT,
                    language=voice.language or "en",
                    style_tags=voice.tags or [],
                )

            logger.info(
                "Loaded Fish Audio voices",
                count=len(self._available_voices),
            )

        except Exception as e:
            logger.warning("Failed to load Fish Audio voices", error=str(e))
            self._available_voices = {}

    async def _synthesize_impl(
        self,
        request: VoiceRequest,
        profile: VoiceProfile,
    ) -> VoiceResponse:
        """Synthesize speech using Fish Audio."""
        voice_id = self.get_voice_for_profile(profile)

        # Get emotion marker
        emotion = request.emotion or profile.default_emotion
        emotion_marker = EMOTION_MARKERS.get(emotion, "")

        # Prepare text with emotion marker
        text = request.text
        if emotion_marker:
            text = f"{emotion_marker} {text}"

        try:
            # Generate audio
            if voice_id:
                audio = self._client.tts.convert(
                    text=text,
                    reference_id=voice_id,
                )
            else:
                audio = self._client.tts.convert(text=text)

            # Estimate duration (~150 chars/sec)
            duration_ms = (len(request.text) / 150) * 1000

            return VoiceResponse(
                request_id=request.request_id,
                provider_type=VoiceProviderType.FISH_AUDIO,
                audio_data=audio,
                duration_ms=duration_ms,
                sample_rate=self.config.default_sample_rate,
                format=request.output_format,
                file_size_bytes=len(audio),
                characters_processed=len(request.text),
                cost=(duration_ms / 1000) * self.config.cost_per_second,
                success=True,
            )

        except Exception as e:
            logger.error("Fish Audio synthesis failed", error=str(e))
            return VoiceResponse(
                request_id=request.request_id,
                provider_type=VoiceProviderType.FISH_AUDIO,
                success=False,
                error=str(e),
            )

    async def _stream_synthesize_impl(
        self,
        request: VoiceRequest,
        profile: VoiceProfile,
    ) -> AsyncGenerator[StreamingAudioChunk, None]:
        """Stream audio chunks from Fish Audio using WebSocket."""
        voice_id = self.get_voice_for_profile(profile)

        # Get emotion marker
        emotion = request.emotion or profile.default_emotion
        emotion_marker = EMOTION_MARKERS.get(emotion, "")

        text = request.text
        if emotion_marker:
            text = f"{emotion_marker} {text}"

        try:
            # Use streaming API
            def text_generator():
                yield text

            stream = self._client.tts.stream(
                text=text_generator(),
                reference_id=voice_id,
            )

            chunk_index = 0
            for chunk in stream:
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

        except Exception as e:
            logger.error("Fish Audio streaming failed", error=str(e))
            raise

    async def _perform_health_check(self) -> None:
        """Check Fish Audio API health."""
        if not self._client:
            self._health.status = VoiceProviderStatus.UNAVAILABLE
            return

        try:
            # Try listing voices as health check
            self._client.voice.list(page_size=1)
            self._health.status = VoiceProviderStatus.AVAILABLE
        except Exception as e:
            logger.error("Fish Audio health check failed", error=str(e))
            self._health.status = VoiceProviderStatus.ERROR

    def _select_best_match_voice(self, profile: VoiceProfile) -> Optional[str]:
        """Select best matching voice for profile.

        Fish Audio has limited voice metadata, so we rely on
        explicit mappings or the default voice.

        Args:
            profile: Voice profile to match

        Returns:
            Voice ID or None (uses default)
        """
        # Check explicit mapping
        if VoiceProviderType.FISH_AUDIO in profile.provider_voice_ids:
            return profile.provider_voice_ids[VoiceProviderType.FISH_AUDIO]

        # Use configured default if available
        if self._default_voice_id:
            return self._default_voice_id

        # Return first available voice or None
        if self._available_voices:
            return next(iter(self._available_voices.keys()))

        return None

    def embed_emotion(self, text: str, emotion: VoiceEmotion) -> str:
        """Embed emotion marker in text.

        Fish Audio supports inline emotion markers that affect
        the synthesis of the following text.

        Args:
            text: Original text
            emotion: Emotion to embed

        Returns:
            Text with emotion marker
        """
        marker = EMOTION_MARKERS.get(emotion, "")
        if marker:
            return f"{marker} {text}"
        return text
