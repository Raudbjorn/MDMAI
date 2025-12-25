"""Voice synthesis manager - main orchestrator for TTS functionality."""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

from structlog import get_logger

from .abstract_voice_provider import AbstractVoiceProvider
from .audio_cache_manager import AudioCacheManager
from .models import (
    PreGenerationJob,
    StreamingAudioChunk,
    SynthesisPriority,
    VoiceProfile,
    VoiceProviderConfig,
    VoiceProviderStatus,
    VoiceProviderType,
    VoiceRequest,
    VoiceResponse,
)
from .voice_profile_mapper import VoiceProfileMapper

logger = get_logger(__name__)


class VoiceManager:
    """Main orchestrator for voice synthesis.

    Manages multiple voice providers with failover, voice profiles,
    audio caching, and pre-generation queues.

    Usage:
        manager = VoiceManager(config)
        await manager.initialize()

        response = await manager.synthesize(
            text="Welcome, adventurers!",
            voice_profile_id="dm_narrator",
        )
    """

    def __init__(
        self,
        provider_configs: Optional[List[VoiceProviderConfig]] = None,
        cache_dir: Optional[Path] = None,
        max_cache_size_mb: int = 5000,
        prefer_local: bool = True,
    ):
        """Initialize the voice manager.

        Args:
            provider_configs: List of provider configurations
            cache_dir: Directory for audio cache
            max_cache_size_mb: Maximum cache size
            prefer_local: Prefer local providers over cloud
        """
        self._providers: Dict[VoiceProviderType, AbstractVoiceProvider] = {}
        self._provider_priority: List[VoiceProviderType] = []
        self._prefer_local = prefer_local

        # Voice profile storage
        self._profiles: Dict[str, VoiceProfile] = {}
        self._profiles_dir: Optional[Path] = None

        # Cache manager
        self._cache = AudioCacheManager(
            cache_dir=cache_dir,
            max_size_mb=max_cache_size_mb,
        )

        # Pre-generation queue
        self._pregeneration_queue: asyncio.Queue[PreGenerationJob] = asyncio.Queue()
        self._pregeneration_jobs: Dict[str, PreGenerationJob] = {}
        self._pregeneration_task: Optional[asyncio.Task] = None

        # Initialize providers from config
        if provider_configs:
            for config in provider_configs:
                self._register_provider_from_config(config)

        # Track initialization state
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize all registered providers.

        Initializes providers in priority order and starts
        the pre-generation worker.
        """
        if self._initialized:
            return

        logger.info("Initializing voice manager")

        # Initialize providers
        for provider_type in self._provider_priority:
            provider = self._providers.get(provider_type)
            if provider:
                try:
                    await provider.initialize()
                    logger.info(
                        "Voice provider initialized",
                        provider=provider_type.value,
                    )
                except Exception as e:
                    logger.error(
                        "Failed to initialize voice provider",
                        provider=provider_type.value,
                        error=str(e),
                    )

        # Load saved profiles
        await self._load_profiles()

        # Create default profiles if none exist
        if not self._profiles:
            self._create_default_profiles()

        # Start pre-generation worker
        self._pregeneration_task = asyncio.create_task(self._pregeneration_worker())

        self._initialized = True
        logger.info(
            "Voice manager initialized",
            providers=len([p for p in self._providers.values() if p.is_available]),
            profiles=len(self._profiles),
        )

    async def shutdown(self) -> None:
        """Shutdown the voice manager and cleanup resources."""
        if not self._initialized:
            return

        logger.info("Shutting down voice manager")

        # Cancel pre-generation worker
        if self._pregeneration_task:
            self._pregeneration_task.cancel()
            try:
                await self._pregeneration_task
            except asyncio.CancelledError:
                pass

        # Shutdown providers
        for provider in self._providers.values():
            await provider.shutdown()

        # Save profiles
        await self._save_profiles()

        self._initialized = False

    async def synthesize(
        self,
        text: str,
        voice_profile_id: Optional[str] = None,
        request: Optional[VoiceRequest] = None,
        use_cache: bool = True,
    ) -> VoiceResponse:
        """Synthesize speech from text.

        Args:
            text: Text to synthesize
            voice_profile_id: ID of voice profile to use
            request: Optional full VoiceRequest (overrides text/profile_id)
            use_cache: Whether to use/store in cache

        Returns:
            VoiceResponse with audio data
        """
        # Build request if not provided
        if request is None:
            request = VoiceRequest(
                text=text,
                voice_profile_id=voice_profile_id,
                skip_cache=not use_cache,
            )

        # Get voice profile
        profile = self._get_or_create_profile(request.voice_profile_id)

        # Check cache first
        if use_cache and not request.skip_cache:
            cache_key = self._cache.generate_cache_key(
                text=request.text,
                voice_profile_id=profile.profile_id,
                emotion=request.emotion,
            )

            cached = self._cache.get_cached(cache_key)
            if cached:
                audio_path = self._cache.get_audio_path(cache_key)
                return VoiceResponse(
                    request_id=request.request_id,
                    provider_type=cached.provider_type,
                    audio_path=str(audio_path) if audio_path else None,
                    audio_data=self._cache.get_audio_bytes(cache_key),
                    duration_ms=cached.duration_ms,
                    format=cached.format,
                    cached=True,
                    cache_key=cache_key,
                    cache_tag=cached.cache_tag,
                    success=True,
                )

        # Get available provider
        provider = self._get_available_provider()
        if not provider:
            return VoiceResponse(
                request_id=request.request_id,
                provider_type=VoiceProviderType.SYSTEM,
                success=False,
                error="No voice providers available",
            )

        # Synthesize
        response = await provider.synthesize(request, profile)

        # Store in cache if successful
        if response.success and response.audio_data and use_cache:
            cache_key = self._cache.generate_cache_key(
                text=request.text,
                voice_profile_id=profile.profile_id,
                emotion=request.emotion,
            )

            await self._cache.store_audio(
                cache_key=cache_key,
                audio_data=response.audio_data,
                voice_profile_id=profile.profile_id,
                text=request.text,
                format=response.format,
                duration_ms=response.duration_ms,
                provider_type=response.provider_type,
                cache_tag=request.cache_tag,
                session_id=request.session_id,
                campaign_id=request.campaign_id,
                npc_id=request.npc_id,
                scene_id=request.scene_id,
            )

            response.cache_key = cache_key
            response.cache_tag = request.cache_tag

        # Update profile usage
        profile.usage_count += 1
        profile.updated_at = datetime.utcnow()

        return response

    async def stream_synthesize(
        self,
        text: str,
        voice_profile_id: Optional[str] = None,
        request: Optional[VoiceRequest] = None,
    ) -> AsyncGenerator[StreamingAudioChunk, None]:
        """Stream audio synthesis for real-time playback.

        Args:
            text: Text to synthesize
            voice_profile_id: ID of voice profile
            request: Optional full VoiceRequest

        Yields:
            StreamingAudioChunk instances
        """
        if request is None:
            request = VoiceRequest(
                text=text,
                voice_profile_id=voice_profile_id,
            )

        profile = self._get_or_create_profile(request.voice_profile_id)

        # Get streaming-capable provider
        provider = self._get_available_provider(require_streaming=True)
        if not provider:
            raise RuntimeError("No streaming-capable providers available")

        async for chunk in provider.stream_synthesize(request, profile):
            yield chunk

    async def queue_pre_generation(
        self,
        session_id: str,
        texts: List[Dict[str, Any]],
        priority: SynthesisPriority = SynthesisPriority.NORMAL,
        campaign_id: Optional[str] = None,
    ) -> str:
        """Queue texts for pre-generation.

        Args:
            session_id: Session to pre-generate for
            texts: List of {"text": str, "voice_profile_id": str, "emotion": str?, "tag": str?}
            priority: Generation priority
            campaign_id: Optional campaign context

        Returns:
            Job ID for tracking
        """
        job = PreGenerationJob(
            session_id=session_id,
            campaign_id=campaign_id,
            items=texts,
            total_items=len(texts),
            priority=priority,
        )

        self._pregeneration_jobs[job.job_id] = job
        await self._pregeneration_queue.put(job)

        logger.info(
            "Queued pre-generation job",
            job_id=job.job_id,
            items=len(texts),
            priority=priority.value,
        )

        return job.job_id

    def get_job_status(self, job_id: str) -> Optional[PreGenerationJob]:
        """Get status of a pre-generation job.

        Args:
            job_id: Job ID

        Returns:
            PreGenerationJob or None if not found
        """
        return self._pregeneration_jobs.get(job_id)

    # Profile management

    async def create_profile(
        self,
        name: str,
        personality_profile_id: Optional[str] = None,
        npc_id: Optional[str] = None,
        **kwargs,
    ) -> VoiceProfile:
        """Create a new voice profile.

        Args:
            name: Profile name
            personality_profile_id: Link to personality profile
            npc_id: Link to NPC
            **kwargs: Additional profile parameters

        Returns:
            Created VoiceProfile
        """
        profile = VoiceProfile(
            name=name,
            personality_profile_id=personality_profile_id,
            npc_id=npc_id,
            **kwargs,
        )

        self._profiles[profile.profile_id] = profile
        await self._save_profiles()

        logger.info("Created voice profile", profile_id=profile.profile_id, name=name)
        return profile

    def get_profile(self, profile_id: str) -> Optional[VoiceProfile]:
        """Get a voice profile by ID.

        Args:
            profile_id: Profile ID

        Returns:
            VoiceProfile or None
        """
        return self._profiles.get(profile_id)

    def list_profiles(
        self,
        campaign_id: Optional[str] = None,
        npc_id: Optional[str] = None,
    ) -> List[VoiceProfile]:
        """List voice profiles with optional filtering.

        Args:
            campaign_id: Filter by campaign
            npc_id: Filter by NPC

        Returns:
            List of matching VoiceProfiles
        """
        profiles = list(self._profiles.values())

        if campaign_id:
            profiles = [p for p in profiles if p.campaign_id == campaign_id]
        if npc_id:
            profiles = [p for p in profiles if p.npc_id == npc_id]

        return profiles

    async def delete_profile(self, profile_id: str) -> bool:
        """Delete a voice profile.

        Args:
            profile_id: Profile ID to delete

        Returns:
            True if deleted, False if not found
        """
        if profile_id not in self._profiles:
            return False

        del self._profiles[profile_id]
        await self._save_profiles()
        return True

    # Provider management

    def register_provider(self, provider: AbstractVoiceProvider) -> None:
        """Register a voice provider.

        Args:
            provider: Provider instance to register
        """
        self._providers[provider.provider_type] = provider

        # Update priority list
        if provider.provider_type not in self._provider_priority:
            self._provider_priority.append(provider.provider_type)
            self._sort_provider_priority()

        logger.info("Registered voice provider", provider=provider.provider_type.value)

    def get_provider(self, provider_type: VoiceProviderType) -> Optional[AbstractVoiceProvider]:
        """Get a specific provider.

        Args:
            provider_type: Type of provider

        Returns:
            Provider instance or None
        """
        return self._providers.get(provider_type)

    async def get_all_available_voices(self) -> Dict[str, List[Dict]]:
        """Get all available voices from all providers.

        Returns:
            Dictionary of provider name to list of voice specs
        """
        result = {}

        for provider_type, provider in self._providers.items():
            if provider.is_available:
                voices = []
                for voice_id, spec in provider.available_voices.items():
                    voices.append({
                        "voice_id": voice_id,
                        "name": spec.name,
                        "gender": spec.gender.value,
                        "age": spec.age.value,
                        "description": spec.description,
                    })
                result[provider_type.value] = voices

        return result

    async def get_system_status(self) -> Dict[str, Any]:
        """Get system status including all providers.

        Returns:
            Status dictionary
        """
        providers = {}
        for provider_type, provider in self._providers.items():
            providers[provider_type.value] = {
                "available": provider.is_available,
                "status": provider.health.status.value,
                "local": provider.is_local,
                "streaming": provider.supports_streaming,
                "voices": len(provider.available_voices),
            }

        active = next(
            (pt.value for pt in self._provider_priority if self._providers.get(pt, None) and self._providers[pt].is_available),
            None,
        )

        return {
            "providers": providers,
            "active_provider": active,
            "queue_depth": self._pregeneration_queue.qsize(),
            "profiles_count": len(self._profiles),
            "cache_stats": self._cache.get_stats(),
        }

    # Private methods

    def _register_provider_from_config(self, config: VoiceProviderConfig) -> None:
        """Create and register a provider from config."""
        from .providers.elevenlabs_provider import ElevenLabsProvider
        from .providers.fish_audio_provider import FishAudioProvider
        from .providers.ollama_tts_provider import OllamaTTSProvider

        provider_classes = {
            VoiceProviderType.ELEVENLABS: ElevenLabsProvider,
            VoiceProviderType.FISH_AUDIO: FishAudioProvider,
            VoiceProviderType.OLLAMA_TTS: OllamaTTSProvider,
        }

        provider_class = provider_classes.get(config.provider_type)
        if provider_class:
            provider = provider_class(config)
            self.register_provider(provider)

    def _get_available_provider(
        self,
        require_streaming: bool = False,
    ) -> Optional[AbstractVoiceProvider]:
        """Get the best available provider.

        Args:
            require_streaming: Require streaming support

        Returns:
            Available provider or None
        """
        for provider_type in self._provider_priority:
            provider = self._providers.get(provider_type)
            if provider and provider.is_available:
                if require_streaming and not provider.supports_streaming:
                    continue
                return provider
        return None

    def _sort_provider_priority(self) -> None:
        """Sort providers by priority, preferring local if configured."""
        def priority_key(pt: VoiceProviderType) -> tuple:
            provider = self._providers.get(pt)
            if not provider:
                return (999, 999)

            # Lower is better
            config_priority = provider.config.priority

            # Prefer local providers if configured
            local_bonus = 0 if (self._prefer_local and provider.is_local) else 100

            return (local_bonus, config_priority)

        self._provider_priority.sort(key=priority_key)

    def _get_or_create_profile(self, profile_id: Optional[str]) -> VoiceProfile:
        """Get profile by ID or create default.

        Args:
            profile_id: Profile ID or None for default

        Returns:
            VoiceProfile
        """
        if profile_id and profile_id in self._profiles:
            return self._profiles[profile_id]

        # Check for default DM profile
        if "default_dm" in self._profiles:
            return self._profiles["default_dm"]

        # Create default
        return VoiceProfileMapper.create_default_dm_voice()

    def _create_default_profiles(self) -> None:
        """Create default voice profiles."""
        # Default DM voice
        dm_profile = VoiceProfileMapper.create_default_dm_voice()
        self._profiles[dm_profile.profile_id] = dm_profile
        self._profiles["default_dm"] = dm_profile

        # Narrator presets
        presets = VoiceProfileMapper.create_narrator_presets()
        for name, profile in presets.items():
            self._profiles[profile.profile_id] = profile
            self._profiles[name] = profile

        logger.info("Created default voice profiles", count=len(self._profiles))

    async def _load_profiles(self) -> None:
        """Load saved voice profiles."""
        if not self._profiles_dir:
            # Use cache dir parent for profiles
            self._profiles_dir = self._cache.cache_dir.parent / "voice_profiles"
            self._profiles_dir.mkdir(parents=True, exist_ok=True)

        profiles_file = self._profiles_dir / "profiles.json"
        if profiles_file.exists():
            try:
                data = json.loads(profiles_file.read_text())
                for profile_data in data.get("profiles", []):
                    profile = VoiceProfile.from_dict(profile_data)
                    self._profiles[profile.profile_id] = profile

                logger.info("Loaded voice profiles", count=len(self._profiles))
            except Exception as e:
                logger.error("Failed to load voice profiles", error=str(e))

    async def _save_profiles(self) -> None:
        """Save voice profiles to disk."""
        if not self._profiles_dir:
            return

        profiles_file = self._profiles_dir / "profiles.json"
        try:
            # Deduplicate (remove alias entries)
            unique_profiles = {
                p.profile_id: p
                for p in self._profiles.values()
            }

            data = {
                "profiles": [p.to_dict() for p in unique_profiles.values()],
                "saved_at": datetime.utcnow().isoformat(),
            }
            profiles_file.write_text(json.dumps(data, indent=2, default=str))
        except Exception as e:
            logger.error("Failed to save voice profiles", error=str(e))

    async def _pregeneration_worker(self) -> None:
        """Background worker for processing pre-generation queue."""
        logger.info("Pre-generation worker started")

        while True:
            try:
                job = await self._pregeneration_queue.get()

                job.status = "running"
                job.started_at = datetime.utcnow()

                logger.info(
                    "Processing pre-generation job",
                    job_id=job.job_id,
                    items=job.total_items,
                )

                for item in job.items:
                    try:
                        request = VoiceRequest(
                            text=item["text"],
                            voice_profile_id=item.get("voice_profile_id"),
                            cache_tag=item.get("tag"),
                            session_id=job.session_id,
                            campaign_id=job.campaign_id,
                        )

                        response = await self.synthesize(
                            text=item["text"],
                            voice_profile_id=item.get("voice_profile_id"),
                            request=request,
                            use_cache=True,
                        )

                        if response.success and response.cache_key:
                            job.generated_cache_keys.append(response.cache_key)
                            job.completed_items += 1
                        else:
                            job.failed_items += 1
                            if response.error:
                                job.errors.append(response.error)

                    except Exception as e:
                        job.failed_items += 1
                        job.errors.append(str(e))
                        logger.error(
                            "Pre-generation item failed",
                            job_id=job.job_id,
                            error=str(e),
                        )

                job.status = "completed"
                job.completed_at = datetime.utcnow()

                logger.info(
                    "Pre-generation job completed",
                    job_id=job.job_id,
                    completed=job.completed_items,
                    failed=job.failed_items,
                )

            except asyncio.CancelledError:
                logger.info("Pre-generation worker cancelled")
                break
            except Exception as e:
                logger.error("Pre-generation worker error", error=str(e))
                await asyncio.sleep(1)
