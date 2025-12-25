"""Voice Synthesis module for TTRPG AI Dungeon Master.

This module provides text-to-speech capabilities with support for:
- Multiple providers (ElevenLabs, Fish Audio, Ollama TTS, Coqui, Piper)
- Voice profiles mapped from personality and NPC traits
- Pre-generation and caching for session preparation
- Real-time streaming for improvisation

Usage:
    from voice_synthesis import VoiceManager, VoiceProfile

    # Initialize voice manager
    manager = VoiceManager(config)
    await manager.initialize()

    # Synthesize speech
    response = await manager.synthesize(
        text="Welcome, adventurers!",
        voice_profile_id="dm_narrator",
    )
"""

from .models import (
    AudioFormat,
    CachedAudio,
    PreGenerationJob,
    StreamingAudioChunk,
    SynthesisPriority,
    VoiceAge,
    VoiceEmotion,
    VoiceGender,
    VoiceProfile,
    VoiceProviderConfig,
    VoiceProviderHealth,
    VoiceProviderStats,
    VoiceProviderStatus,
    VoiceProviderType,
    VoiceRequest,
    VoiceResponse,
    VoiceSpec,
    VoiceUsageRecord,
)
from .abstract_voice_provider import AbstractVoiceProvider
from .audio_cache_manager import AudioCacheManager
from .voice_profile_mapper import VoiceProfileMapper
from .voice_manager import VoiceManager
from .mcp_tools import initialize_voice_tools, register_voice_tools

__all__ = [
    # Enums
    "AudioFormat",
    "SynthesisPriority",
    "VoiceAge",
    "VoiceEmotion",
    "VoiceGender",
    "VoiceProviderStatus",
    "VoiceProviderType",
    # Data classes
    "CachedAudio",
    "PreGenerationJob",
    "StreamingAudioChunk",
    "VoiceProfile",
    "VoiceProviderConfig",
    "VoiceProviderHealth",
    "VoiceProviderStats",
    "VoiceRequest",
    "VoiceResponse",
    "VoiceSpec",
    "VoiceUsageRecord",
    # Core classes
    "AbstractVoiceProvider",
    "AudioCacheManager",
    "VoiceManager",
    "VoiceProfileMapper",
    # Functions
    "initialize_voice_tools",
    "register_voice_tools",
]
