"""ElevenLabs voice synthesis provider."""

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

# Lazy import to avoid dependency if not using ElevenLabs
ElevenLabs = None
Voice = None
VoiceSettings = None


def _ensure_elevenlabs_import():
    """Lazy import ElevenLabs SDK."""
    global ElevenLabs, Voice, VoiceSettings
    if ElevenLabs is None:
        try:
            from elevenlabs import ElevenLabs as EL
            from elevenlabs import Voice as V
            from elevenlabs import VoiceSettings as VS
            ElevenLabs = EL
            Voice = V
            VoiceSettings = VS
        except ImportError:
            raise ImportError(
                "ElevenLabs SDK not installed. "
                "Install with: pip install elevenlabs"
            )


# ElevenLabs voice settings by emotion
EMOTION_SETTINGS: Dict[VoiceEmotion, Dict[str, float]] = {
    VoiceEmotion.NEUTRAL: {"stability": 0.5, "similarity_boost": 0.75, "style": 0.0},
    VoiceEmotion.HAPPY: {"stability": 0.3, "similarity_boost": 0.8, "style": 0.3},
    VoiceEmotion.SAD: {"stability": 0.7, "similarity_boost": 0.6, "style": 0.2},
    VoiceEmotion.ANGRY: {"stability": 0.2, "similarity_boost": 0.9, "style": 0.5},
    VoiceEmotion.FEARFUL: {"stability": 0.3, "similarity_boost": 0.7, "style": 0.4},
    VoiceEmotion.EXCITED: {"stability": 0.2, "similarity_boost": 0.85, "style": 0.4},
    VoiceEmotion.MYSTERIOUS: {"stability": 0.6, "similarity_boost": 0.5, "style": 0.3},
    VoiceEmotion.AUTHORITATIVE: {"stability": 0.8, "similarity_boost": 0.8, "style": 0.2},
    VoiceEmotion.WHIMSICAL: {"stability": 0.3, "similarity_boost": 0.7, "style": 0.5},
    VoiceEmotion.OMINOUS: {"stability": 0.7, "similarity_boost": 0.6, "style": 0.4},
    VoiceEmotion.CALM: {"stability": 0.8, "similarity_boost": 0.7, "style": 0.1},
    VoiceEmotion.URGENT: {"stability": 0.3, "similarity_boost": 0.85, "style": 0.4},
}

# Default voice selections by gender/age
DEFAULT_VOICES: Dict[tuple, str] = {
    # (gender, age): voice_id
    (VoiceGender.MALE, VoiceAge.YOUNG): "pNInz6obpgDQGcFmaJgB",     # Adam
    (VoiceGender.MALE, VoiceAge.ADULT): "VR6AewLTigWG4xSOukaG",     # Arnold
    (VoiceGender.MALE, VoiceAge.MATURE): "yoZ06aMxZJJ28mfd3POQ",    # Sam
    (VoiceGender.MALE, VoiceAge.ELDERLY): "GBv7mTt0atIp3Br8iCZE",   # Thomas
    (VoiceGender.FEMALE, VoiceAge.YOUNG): "jBpfuIE2acCO8z3wKNLl",   # Gigi
    (VoiceGender.FEMALE, VoiceAge.ADULT): "EXAVITQu4vr4xnSDxMaL",   # Bella
    (VoiceGender.FEMALE, VoiceAge.MATURE): "21m00Tcm4TlvDq8ikWAM",  # Rachel
    (VoiceGender.FEMALE, VoiceAge.ELDERLY): "ThT5KcBeYPX3keUQqHPh", # Dorothy
    (VoiceGender.NEUTRAL, VoiceAge.ADULT): "21m00Tcm4TlvDq8ikWAM",  # Rachel (default)
}

# Format mapping for ElevenLabs
FORMAT_MAPPING: Dict[AudioFormat, str] = {
    AudioFormat.MP3: "mp3_44100_128",
    AudioFormat.WAV: "pcm_44100",
    AudioFormat.OGG: "ogg_opus",
}


class ElevenLabsProvider(AbstractVoiceProvider):
    """ElevenLabs cloud voice synthesis provider.

    Provides high-quality text-to-speech using the ElevenLabs API.
    Supports streaming, multiple voices, and emotion control via
    stability/similarity settings.

    Pricing: ~$0.00003 per character ($30 per 1M characters)
    """

    def __init__(self, config: VoiceProviderConfig):
        """Initialize ElevenLabs provider.

        Args:
            config: Provider configuration with API key
        """
        # Ensure config has correct provider type
        config.provider_type = VoiceProviderType.ELEVENLABS
        super().__init__(config)

        # Default costs if not configured
        if config.cost_per_character == 0.0:
            self.config.cost_per_character = 0.00003

        # Default model
        self._model_id = config.metadata.get("model_id", "eleven_multilingual_v2")

    @property
    def supports_streaming(self) -> bool:
        """ElevenLabs supports streaming synthesis."""
        return True

    @property
    def supports_emotion(self) -> bool:
        """ElevenLabs supports emotion via stability/similarity settings."""
        return True

    async def _initialize_client(self) -> None:
        """Initialize the ElevenLabs client."""
        _ensure_elevenlabs_import()

        api_key = self.config.api_key
        if not api_key:
            raise ValueError("ElevenLabs API key is required")

        self._client = ElevenLabs(api_key=api_key)
        self._health.status = VoiceProviderStatus.AVAILABLE

        logger.info("ElevenLabs client initialized")

    async def _cleanup_client(self) -> None:
        """Clean up the ElevenLabs client."""
        self._client = None

    async def _load_available_voices(self) -> None:
        """Load available voices from ElevenLabs."""
        if not self._client:
            return

        try:
            voices_response = self._client.voices.get_all()

            for voice in voices_response.voices:
                # Determine gender and age from labels
                gender = VoiceGender.NEUTRAL
                age = VoiceAge.ADULT

                if voice.labels:
                    if voice.labels.get("gender") == "male":
                        gender = VoiceGender.MALE
                    elif voice.labels.get("gender") == "female":
                        gender = VoiceGender.FEMALE

                    age_label = voice.labels.get("age", "").lower()
                    if "young" in age_label:
                        age = VoiceAge.YOUNG
                    elif "old" in age_label or "elderly" in age_label:
                        age = VoiceAge.ELDERLY
                    elif "middle" in age_label or "mature" in age_label:
                        age = VoiceAge.MATURE

                self._available_voices[voice.voice_id] = VoiceSpec(
                    voice_id=voice.voice_id,
                    provider_type=VoiceProviderType.ELEVENLABS,
                    name=voice.name,
                    description=voice.description or "",
                    gender=gender,
                    age=age,
                    language="en",  # Most voices are English
                    style_tags=list(voice.labels.values()) if voice.labels else [],
                    preview_url=voice.preview_url,
                    is_cloned=voice.category == "cloned",
                )

            logger.info(
                "Loaded ElevenLabs voices",
                count=len(self._available_voices),
            )

        except Exception as e:
            logger.error("Failed to load ElevenLabs voices", error=str(e))
            # Don't fail initialization, just use defaults
            self._available_voices = {}

    async def _synthesize_impl(
        self,
        request: VoiceRequest,
        profile: VoiceProfile,
    ) -> VoiceResponse:
        """Synthesize speech using ElevenLabs."""
        voice_id = self.get_voice_for_profile(profile)
        if not voice_id:
            voice_id = DEFAULT_VOICES.get(
                (VoiceGender.NEUTRAL, VoiceAge.ADULT),
                "21m00Tcm4TlvDq8ikWAM",
            )

        # Get emotion settings
        emotion = request.emotion or profile.default_emotion
        base_settings = EMOTION_SETTINGS.get(emotion, EMOTION_SETTINGS[VoiceEmotion.NEUTRAL])

        # Apply profile adjustments
        stability = base_settings["stability"]
        stability = stability * (1.0 - profile.pitch_variance * 0.5)  # More variance = less stability

        similarity = base_settings["similarity_boost"]
        style = base_settings["style"] + (profile.energy - 0.5) * 0.4  # Energy affects style

        # Clamp values
        stability = max(0.0, min(1.0, stability))
        similarity = max(0.0, min(1.0, similarity))
        style = max(0.0, min(1.0, style))

        voice_settings = VoiceSettings(
            stability=stability,
            similarity_boost=similarity,
            style=style,
            use_speaker_boost=True,
        )

        # Determine output format
        output_format = FORMAT_MAPPING.get(request.output_format, "mp3_44100_128")

        try:
            # Generate audio
            audio_generator = self._client.generate(
                text=request.text,
                voice=Voice(
                    voice_id=voice_id,
                    settings=voice_settings,
                ),
                model=self._model_id,
                output_format=output_format,
            )

            # Collect audio bytes
            audio_bytes = b"".join(audio_generator)

            # Estimate duration (rough: ~150 chars/sec at normal speed)
            chars_per_sec = 150 * profile.speed * 2  # speed 0.5 = normal
            duration_ms = (len(request.text) / chars_per_sec) * 1000

            return VoiceResponse(
                request_id=request.request_id,
                provider_type=VoiceProviderType.ELEVENLABS,
                audio_data=audio_bytes,
                duration_ms=duration_ms,
                sample_rate=44100,
                format=request.output_format,
                file_size_bytes=len(audio_bytes),
                characters_processed=len(request.text),
                cost=self.estimate_cost(request.text),
                success=True,
            )

        except Exception as e:
            logger.error("ElevenLabs synthesis failed", error=str(e))
            return VoiceResponse(
                request_id=request.request_id,
                provider_type=VoiceProviderType.ELEVENLABS,
                success=False,
                error=str(e),
            )

    async def _stream_synthesize_impl(
        self,
        request: VoiceRequest,
        profile: VoiceProfile,
    ) -> AsyncGenerator[StreamingAudioChunk, None]:
        """Stream audio chunks from ElevenLabs."""
        voice_id = self.get_voice_for_profile(profile)
        if not voice_id:
            voice_id = DEFAULT_VOICES.get(
                (VoiceGender.NEUTRAL, VoiceAge.ADULT),
                "21m00Tcm4TlvDq8ikWAM",
            )

        # Get emotion settings
        emotion = request.emotion or profile.default_emotion
        base_settings = EMOTION_SETTINGS.get(emotion, EMOTION_SETTINGS[VoiceEmotion.NEUTRAL])

        voice_settings = VoiceSettings(
            stability=base_settings["stability"],
            similarity_boost=base_settings["similarity_boost"],
            style=base_settings["style"],
            use_speaker_boost=True,
        )

        output_format = FORMAT_MAPPING.get(request.output_format, "mp3_44100_128")

        try:
            # Stream audio
            audio_stream = self._client.generate(
                text=request.text,
                voice=Voice(
                    voice_id=voice_id,
                    settings=voice_settings,
                ),
                model=self._model_id,
                output_format=output_format,
                stream=True,
            )

            chunk_index = 0
            for chunk in audio_stream:
                if chunk:
                    yield StreamingAudioChunk(
                        request_id=request.request_id,
                        chunk_index=chunk_index,
                        audio_data=chunk,
                        is_final=False,
                    )
                    chunk_index += 1

            # Send final chunk marker
            yield StreamingAudioChunk(
                request_id=request.request_id,
                chunk_index=chunk_index,
                audio_data=b"",
                is_final=True,
            )

        except Exception as e:
            logger.error("ElevenLabs streaming failed", error=str(e))
            raise

    async def _perform_health_check(self) -> None:
        """Check ElevenLabs API health."""
        if not self._client:
            self._health.status = VoiceProviderStatus.UNAVAILABLE
            return

        try:
            # Check user/subscription info
            user = self._client.user.get()
            subscription = user.subscription

            # Update quota info
            if subscription:
                self._health.quota_remaining = (
                    subscription.character_limit - subscription.character_count
                )

            self._health.status = VoiceProviderStatus.AVAILABLE

        except Exception as e:
            logger.error("ElevenLabs health check failed", error=str(e))
            self._health.status = VoiceProviderStatus.ERROR

    def _select_best_match_voice(self, profile: VoiceProfile) -> Optional[str]:
        """Select best matching voice for profile.

        Args:
            profile: Voice profile to match

        Returns:
            Voice ID or None
        """
        # First try exact gender/age match
        key = (profile.gender, profile.age)
        if key in DEFAULT_VOICES:
            voice_id = DEFAULT_VOICES[key]
            if voice_id in self._available_voices or not self._available_voices:
                return voice_id

        # Search available voices
        best_match = None
        best_score = -1

        for voice_id, spec in self._available_voices.items():
            score = 0

            # Gender match
            if spec.gender == profile.gender:
                score += 10
            elif spec.gender == VoiceGender.NEUTRAL:
                score += 3

            # Age match
            if spec.age == profile.age:
                score += 5
            elif abs(list(VoiceAge).index(spec.age) - list(VoiceAge).index(profile.age)) == 1:
                score += 2  # Adjacent age

            # Prefer non-cloned voices
            if not spec.is_cloned:
                score += 1

            if score > best_score:
                best_score = score
                best_match = voice_id

        return best_match or DEFAULT_VOICES.get((VoiceGender.NEUTRAL, VoiceAge.ADULT))

    async def text_to_dialogue(
        self,
        inputs: list[Dict],
    ) -> bytes:
        """Generate dialogue with multiple speakers.

        ElevenLabs supports multi-voice dialogue generation.

        Args:
            inputs: List of {"text": str, "voice_id": str} dicts

        Returns:
            Combined audio bytes
        """
        _ensure_elevenlabs_import()

        try:
            from elevenlabs import DialogueInput

            dialogue_inputs = [
                DialogueInput(text=item["text"], voice_id=item["voice_id"])
                for item in inputs
            ]

            audio_stream = self._client.text_to_dialogue.stream(inputs=dialogue_inputs)
            return b"".join(audio_stream)

        except ImportError:
            # Fall back to sequential generation
            logger.warning("DialogueInput not available, using sequential generation")
            audio_parts = []

            for item in inputs:
                profile = VoiceProfile(
                    provider_voice_ids={VoiceProviderType.ELEVENLABS: item["voice_id"]}
                )
                request = VoiceRequest(text=item["text"])
                response = await self._synthesize_impl(request, profile)
                if response.audio_data:
                    audio_parts.append(response.audio_data)

            return b"".join(audio_parts)
