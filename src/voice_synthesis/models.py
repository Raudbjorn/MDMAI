"""Data models for Voice Synthesis."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class VoiceProviderType(Enum):
    """Supported voice synthesis providers."""

    OLLAMA_TTS = "ollama_tts"       # Local Orpheus model
    COQUI = "coqui"                  # Local Coqui XTTS
    PIPER = "piper"                  # Local Piper TTS
    ELEVENLABS = "elevenlabs"        # Cloud ElevenLabs
    FISH_AUDIO = "fish_audio"        # Cloud Fish Audio
    SYSTEM = "system"                # OS-native fallback


class VoiceProviderStatus(Enum):
    """Voice provider status."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    RATE_LIMITED = "rate_limited"
    QUOTA_EXCEEDED = "quota_exceeded"
    ERROR = "error"
    INITIALIZING = "initializing"


class VoiceGender(Enum):
    """Voice gender options."""

    MALE = "male"
    FEMALE = "female"
    NEUTRAL = "neutral"


class VoiceAge(Enum):
    """Voice age ranges."""

    YOUNG = "young"       # ~18-25
    ADULT = "adult"       # ~25-45
    MATURE = "mature"     # ~45-60
    ELDERLY = "elderly"   # ~60+


class VoiceEmotion(Enum):
    """Voice emotion/tone options."""

    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    FEARFUL = "fearful"
    EXCITED = "excited"
    MYSTERIOUS = "mysterious"
    AUTHORITATIVE = "authoritative"
    WHIMSICAL = "whimsical"
    OMINOUS = "ominous"
    CALM = "calm"
    URGENT = "urgent"


class AudioFormat(Enum):
    """Supported audio output formats."""

    MP3 = "mp3"
    WAV = "wav"
    OGG = "ogg"
    OPUS = "opus"
    FLAC = "flac"


class SynthesisPriority(Enum):
    """Voice synthesis priority levels."""

    REALTIME = "realtime"   # Immediate, for live improvisation
    HIGH = "high"           # Soon, for active session
    NORMAL = "normal"       # Background pre-generation
    LOW = "low"             # Batch processing


@dataclass
class VoiceProviderConfig:
    """Configuration for a voice provider."""

    provider_type: VoiceProviderType
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model: Optional[str] = None
    timeout: float = 30.0
    max_retries: int = 3
    retry_delay: float = 1.0
    rate_limit_rpm: int = 60
    enabled: bool = True
    priority: int = 1  # Lower = preferred
    cost_per_character: float = 0.0
    cost_per_second: float = 0.0
    supports_streaming: bool = False
    supports_emotion: bool = False
    supported_formats: List[AudioFormat] = field(default_factory=lambda: [AudioFormat.MP3])
    default_sample_rate: int = 22050
    max_text_length: int = 5000
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VoiceSpec:
    """Specification for a provider voice."""

    voice_id: str
    provider_type: VoiceProviderType
    name: str
    description: str = ""
    gender: VoiceGender = VoiceGender.NEUTRAL
    age: VoiceAge = VoiceAge.ADULT
    language: str = "en"
    accent: Optional[str] = None
    style_tags: List[str] = field(default_factory=list)
    preview_url: Optional[str] = None
    is_cloned: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VoiceProfile:
    """Voice profile for DM or NPC.

    Defines how text should be synthesized, including prosody parameters,
    emotion defaults, and provider-specific voice mappings.
    """

    profile_id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    description: str = ""

    # Voice characteristics
    gender: VoiceGender = VoiceGender.NEUTRAL
    age: VoiceAge = VoiceAge.ADULT

    # Prosody parameters (normalized 0.0 - 1.0)
    pitch: float = 0.5           # Base pitch (0=low, 1=high)
    pitch_variance: float = 0.3  # How much pitch varies (monotone vs expressive)
    speed: float = 0.5           # Speaking rate (0=slow, 1=fast)
    energy: float = 0.5          # Voice energy/intensity

    # Voice quality modifiers
    breathiness: float = 0.0     # Airy quality (0-1)
    roughness: float = 0.0       # Gravelly quality (0-1)
    warmth: float = 0.5          # Warm vs cold tone (0-1)

    # Provider-specific voice ID mappings
    provider_voice_ids: Dict[VoiceProviderType, str] = field(default_factory=dict)

    # Emotion defaults
    default_emotion: VoiceEmotion = VoiceEmotion.NEUTRAL

    # Linked entities
    personality_profile_id: Optional[str] = None
    npc_id: Optional[str] = None
    character_id: Optional[str] = None
    campaign_id: Optional[str] = None

    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    usage_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "profile_id": self.profile_id,
            "name": self.name,
            "description": self.description,
            "gender": self.gender.value,
            "age": self.age.value,
            "pitch": self.pitch,
            "pitch_variance": self.pitch_variance,
            "speed": self.speed,
            "energy": self.energy,
            "breathiness": self.breathiness,
            "roughness": self.roughness,
            "warmth": self.warmth,
            "provider_voice_ids": {k.value: v for k, v in self.provider_voice_ids.items()},
            "default_emotion": self.default_emotion.value,
            "personality_profile_id": self.personality_profile_id,
            "npc_id": self.npc_id,
            "character_id": self.character_id,
            "campaign_id": self.campaign_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "usage_count": self.usage_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VoiceProfile":
        """Create from dictionary."""
        provider_voice_ids = {}
        for k, v in data.get("provider_voice_ids", {}).items():
            try:
                provider_voice_ids[VoiceProviderType(k)] = v
            except ValueError:
                pass

        return cls(
            profile_id=data.get("profile_id", str(uuid4())),
            name=data.get("name", ""),
            description=data.get("description", ""),
            gender=VoiceGender(data.get("gender", "neutral")),
            age=VoiceAge(data.get("age", "adult")),
            pitch=data.get("pitch", 0.5),
            pitch_variance=data.get("pitch_variance", 0.3),
            speed=data.get("speed", 0.5),
            energy=data.get("energy", 0.5),
            breathiness=data.get("breathiness", 0.0),
            roughness=data.get("roughness", 0.0),
            warmth=data.get("warmth", 0.5),
            provider_voice_ids=provider_voice_ids,
            default_emotion=VoiceEmotion(data.get("default_emotion", "neutral")),
            personality_profile_id=data.get("personality_profile_id"),
            npc_id=data.get("npc_id"),
            character_id=data.get("character_id"),
            campaign_id=data.get("campaign_id"),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.utcnow(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if "updated_at" in data else datetime.utcnow(),
            usage_count=data.get("usage_count", 0),
        )


class VoiceRequest(BaseModel):
    """Request for voice synthesis."""

    request_id: str = Field(default_factory=lambda: str(uuid4()))
    text: str
    voice_profile_id: Optional[str] = None

    # Override emotion for this request
    emotion: Optional[VoiceEmotion] = None

    # Output preferences
    output_format: AudioFormat = AudioFormat.MP3
    sample_rate: int = 22050

    # Context for cache tagging
    session_id: Optional[str] = None
    campaign_id: Optional[str] = None
    npc_id: Optional[str] = None
    scene_id: Optional[str] = None

    # Cache control
    cache_tag: Optional[str] = None  # Custom tag for retrieval
    skip_cache: bool = False

    # Priority
    priority: SynthesisPriority = SynthesisPriority.NORMAL

    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class VoiceResponse(BaseModel):
    """Response from voice synthesis."""

    request_id: str
    provider_type: VoiceProviderType

    # Audio data (one of these will be set)
    audio_data: Optional[bytes] = None
    audio_url: Optional[str] = None  # For cached/streaming
    audio_path: Optional[str] = None  # Local file path

    # Audio metadata
    duration_ms: float = 0.0
    sample_rate: int = 22050
    format: AudioFormat = AudioFormat.MP3
    file_size_bytes: int = 0

    # Cost tracking
    characters_processed: int = 0
    cost: float = 0.0

    # Performance
    latency_ms: float = 0.0
    cached: bool = False

    # Cache info
    cache_key: Optional[str] = None
    cache_tag: Optional[str] = None

    # Status
    success: bool = True
    error: Optional[str] = None

    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        arbitrary_types_allowed = True


class StreamingAudioChunk(BaseModel):
    """Streaming audio chunk for real-time synthesis."""

    request_id: str
    chunk_id: str = Field(default_factory=lambda: str(uuid4()))
    chunk_index: int = 0
    audio_data: bytes
    duration_ms: float = 0.0
    is_final: bool = False

    class Config:
        arbitrary_types_allowed = True


@dataclass
class CachedAudio:
    """Cached audio entry."""

    cache_key: str
    audio_path: str
    voice_profile_id: str
    text_hash: str

    # Audio metadata
    duration_ms: float
    file_size_bytes: int
    format: AudioFormat
    sample_rate: int = 22050

    # Provider info
    provider_type: VoiceProviderType = VoiceProviderType.SYSTEM

    # Tracking
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    access_count: int = 0

    # Context tags for retrieval
    cache_tag: Optional[str] = None
    session_id: Optional[str] = None
    campaign_id: Optional[str] = None
    npc_id: Optional[str] = None
    scene_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "cache_key": self.cache_key,
            "audio_path": self.audio_path,
            "voice_profile_id": self.voice_profile_id,
            "text_hash": self.text_hash,
            "duration_ms": self.duration_ms,
            "file_size_bytes": self.file_size_bytes,
            "format": self.format.value,
            "sample_rate": self.sample_rate,
            "provider_type": self.provider_type.value,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "access_count": self.access_count,
            "cache_tag": self.cache_tag,
            "session_id": self.session_id,
            "campaign_id": self.campaign_id,
            "npc_id": self.npc_id,
            "scene_id": self.scene_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CachedAudio":
        """Create from dictionary."""
        provider_type = VoiceProviderType.SYSTEM
        if "provider_type" in data:
            try:
                provider_type = VoiceProviderType(data["provider_type"])
            except ValueError:
                pass

        return cls(
            cache_key=data["cache_key"],
            audio_path=data["audio_path"],
            voice_profile_id=data["voice_profile_id"],
            text_hash=data["text_hash"],
            duration_ms=data["duration_ms"],
            file_size_bytes=data["file_size_bytes"],
            format=AudioFormat(data["format"]),
            sample_rate=data.get("sample_rate", 22050),
            provider_type=provider_type,
            created_at=datetime.fromisoformat(data["created_at"]),
            last_accessed=datetime.fromisoformat(data["last_accessed"]),
            access_count=data.get("access_count", 0),
            cache_tag=data.get("cache_tag"),
            session_id=data.get("session_id"),
            campaign_id=data.get("campaign_id"),
            npc_id=data.get("npc_id"),
            scene_id=data.get("scene_id"),
        )


@dataclass
class VoiceProviderHealth:
    """Health status of a voice provider."""

    provider_type: VoiceProviderType
    status: VoiceProviderStatus
    last_success: Optional[datetime] = None
    last_error: Optional[datetime] = None
    error_count: int = 0
    success_count: int = 0
    avg_latency_ms: float = 0.0
    rate_limit_remaining: Optional[int] = None
    rate_limit_reset: Optional[datetime] = None
    quota_remaining: Optional[float] = None
    uptime_percentage: float = 100.0
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class PreGenerationJob:
    """Job for batch audio pre-generation."""

    job_id: str = field(default_factory=lambda: str(uuid4()))
    session_id: str = ""
    campaign_id: Optional[str] = None

    # Items to generate
    items: List[Dict[str, Any]] = field(default_factory=list)
    total_items: int = 0
    completed_items: int = 0
    failed_items: int = 0

    # Status
    status: str = "pending"  # pending, running, completed, failed, cancelled
    priority: SynthesisPriority = SynthesisPriority.NORMAL

    # Tracking
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Results
    generated_cache_keys: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def progress_percent(self) -> float:
        """Get completion percentage."""
        if self.total_items == 0:
            return 0.0
        return (self.completed_items / self.total_items) * 100


class VoiceUsageRecord(BaseModel):
    """Usage tracking record for voice synthesis."""

    record_id: str = Field(default_factory=lambda: str(uuid4()))
    provider_type: VoiceProviderType
    request_id: str
    voice_profile_id: Optional[str] = None
    session_id: Optional[str] = None
    campaign_id: Optional[str] = None

    # Usage metrics
    characters_synthesized: int = 0
    audio_duration_ms: float = 0.0
    cost: float = 0.0
    latency_ms: float = 0.0

    # Status
    success: bool = True
    cached: bool = False
    error_message: Optional[str] = None

    timestamp: datetime = Field(default_factory=datetime.utcnow)


class VoiceProviderStats(BaseModel):
    """Statistics for voice provider usage."""

    provider_type: VoiceProviderType
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    cached_requests: int = 0
    total_characters: int = 0
    total_audio_duration_ms: float = 0.0
    total_cost: float = 0.0
    avg_latency_ms: float = 0.0
    uptime_percentage: float = 100.0
    last_request: Optional[datetime] = None
