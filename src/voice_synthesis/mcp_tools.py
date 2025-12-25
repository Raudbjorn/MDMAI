"""MCP tool definitions for voice synthesis."""

from typing import Any, Dict, List, Optional

from structlog import get_logger

from .models import SynthesisPriority, VoiceEmotion, VoiceRequest

logger = get_logger(__name__)

# Global references initialized by main.py
_voice_manager = None
_audio_cache = None


def initialize_voice_tools(voice_manager, audio_cache=None) -> None:
    """Initialize voice synthesis tools with manager instances.

    Args:
        voice_manager: VoiceManager instance
        audio_cache: Optional AudioCacheManager (uses manager's cache if not provided)
    """
    global _voice_manager, _audio_cache
    _voice_manager = voice_manager
    _audio_cache = audio_cache or (voice_manager._cache if voice_manager else None)
    logger.info("Voice synthesis MCP tools initialized")


def register_voice_tools(mcp_server) -> None:
    """Register voice synthesis tools with MCP server.

    Args:
        mcp_server: FastMCP server instance
    """
    mcp_server.tool()(synthesize_voice)
    mcp_server.tool()(create_voice_profile)
    mcp_server.tool()(get_voice_profile)
    mcp_server.tool()(list_voice_profiles)
    mcp_server.tool()(pre_generate_session_audio)
    mcp_server.tool()(get_cached_audio)
    mcp_server.tool()(list_available_voices)
    mcp_server.tool()(get_voice_synthesis_status)
    mcp_server.tool()(get_pregeneration_job_status)

    logger.info("Voice synthesis MCP tools registered")


async def synthesize_voice(
    text: str,
    voice_profile_id: Optional[str] = None,
    emotion: Optional[str] = None,
    npc_id: Optional[str] = None,
    cache_tag: Optional[str] = None,
    use_cache: bool = True,
    stream: bool = False,
    session_id: Optional[str] = None,
    campaign_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Synthesize speech from text using configured voice profile.

    Generates audio from text using the voice synthesis system.
    Supports caching for session preparation and streaming for
    real-time playback.

    Args:
        text: Text to synthesize into speech
        voice_profile_id: ID of voice profile to use (optional, uses default DM voice if not specified)
        emotion: Override emotion for this synthesis (happy, sad, angry, mysterious, etc.)
        npc_id: NPC ID to auto-select voice for (optional)
        cache_tag: Custom tag for retrieving this audio later (e.g., "npc_bob_greeting")
        use_cache: Whether to use/store in cache (default True)
        stream: Whether to stream audio for real-time playback (default False)
        session_id: Session context for caching
        campaign_id: Campaign context for caching

    Returns:
        Dict with:
        - success: bool
        - audio_url: URL to retrieve audio (if not streaming)
        - stream_endpoint: WebSocket endpoint (if streaming)
        - duration_ms: Audio duration in milliseconds
        - cached: Whether audio was retrieved from cache
        - cost: Estimated cost in USD
        - error: Error message if failed
    """
    if not _voice_manager:
        return {"success": False, "error": "Voice manager not initialized"}

    # Parse emotion if provided
    voice_emotion = None
    if emotion:
        try:
            voice_emotion = VoiceEmotion(emotion.lower())
        except ValueError:
            valid_emotions = [e.value for e in VoiceEmotion]
            return {
                "success": False,
                "error": f"Invalid emotion '{emotion}'. Valid options: {valid_emotions}",
            }

    # Build request
    request = VoiceRequest(
        text=text,
        voice_profile_id=voice_profile_id,
        emotion=voice_emotion,
        npc_id=npc_id,
        cache_tag=cache_tag,
        skip_cache=not use_cache,
        session_id=session_id,
        campaign_id=campaign_id,
    )

    try:
        if stream:
            # For streaming, return endpoint info
            # Actual streaming handled by bridge server
            return {
                "success": True,
                "streaming": True,
                "stream_endpoint": f"/api/voice/stream/{request.request_id}",
                "request_id": request.request_id,
            }
        else:
            response = await _voice_manager.synthesize(
                text=text,
                voice_profile_id=voice_profile_id,
                request=request,
                use_cache=use_cache,
            )

            if response.success:
                return {
                    "success": True,
                    "audio_url": f"/api/voice/audio/{response.request_id}",
                    "cache_key": response.cache_key,
                    "cache_tag": response.cache_tag,
                    "duration_ms": response.duration_ms,
                    "cached": response.cached,
                    "cost": response.cost,
                    "provider": response.provider_type.value,
                }
            else:
                return {"success": False, "error": response.error}

    except Exception as e:
        logger.error("Voice synthesis failed", error=str(e))
        return {"success": False, "error": str(e)}


async def create_voice_profile(
    name: str,
    personality_profile_id: Optional[str] = None,
    npc_id: Optional[str] = None,
    gender: str = "neutral",
    age: str = "adult",
    pitch: float = 0.5,
    speed: float = 0.5,
    energy: float = 0.5,
    default_emotion: str = "neutral",
    campaign_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create a new voice profile for DM narration or NPC dialogue.

    Voice profiles define how text should be synthesized, including
    pitch, speed, and default emotion. Profiles can be linked to
    personality profiles or NPCs for automatic voice matching.

    Args:
        name: Human-readable profile name
        personality_profile_id: Link to existing personality profile for auto-mapping
        npc_id: Link to existing NPC for auto-mapping
        gender: Voice gender (male, female, neutral)
        age: Voice age (young, adult, mature, elderly)
        pitch: Base pitch 0.0-1.0 (0=low, 1=high)
        speed: Speaking rate 0.0-1.0 (0=slow, 1=fast)
        energy: Voice energy/intensity 0.0-1.0
        default_emotion: Default emotion (neutral, happy, sad, angry, mysterious, etc.)
        campaign_id: Campaign to associate profile with

    Returns:
        Dict with:
        - success: bool
        - profile_id: Created profile ID
        - name: Profile name
        - error: Error message if failed
    """
    if not _voice_manager:
        return {"success": False, "error": "Voice manager not initialized"}

    try:
        # Validate emotion
        try:
            emotion = VoiceEmotion(default_emotion.lower())
        except ValueError:
            emotion = VoiceEmotion.NEUTRAL

        profile = await _voice_manager.create_profile(
            name=name,
            personality_profile_id=personality_profile_id,
            npc_id=npc_id,
            gender=gender,
            age=age,
            pitch=pitch,
            speed=speed,
            energy=energy,
            default_emotion=emotion,
            campaign_id=campaign_id,
        )

        return {
            "success": True,
            "profile_id": profile.profile_id,
            "name": profile.name,
        }

    except Exception as e:
        logger.error("Failed to create voice profile", error=str(e))
        return {"success": False, "error": str(e)}


async def get_voice_profile(profile_id: str) -> Dict[str, Any]:
    """
    Get a voice profile by ID.

    Args:
        profile_id: ID of the voice profile to retrieve

    Returns:
        Dict with:
        - success: bool
        - profile: Voice profile details (if found)
        - error: Error message if not found
    """
    if not _voice_manager:
        return {"success": False, "error": "Voice manager not initialized"}

    profile = _voice_manager.get_profile(profile_id)
    if profile:
        return {
            "success": True,
            "profile": profile.to_dict(),
        }

    return {"success": False, "error": f"Profile '{profile_id}' not found"}


async def list_voice_profiles(
    campaign_id: Optional[str] = None,
    npc_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    List voice profiles with optional filtering.

    Args:
        campaign_id: Filter by campaign ID
        npc_id: Filter by NPC ID

    Returns:
        Dict with:
        - success: bool
        - profiles: List of voice profile summaries
        - count: Number of profiles returned
    """
    if not _voice_manager:
        return {"success": False, "error": "Voice manager not initialized"}

    profiles = _voice_manager.list_profiles(
        campaign_id=campaign_id,
        npc_id=npc_id,
    )

    return {
        "success": True,
        "profiles": [
            {
                "profile_id": p.profile_id,
                "name": p.name,
                "gender": p.gender.value,
                "age": p.age.value,
                "default_emotion": p.default_emotion.value,
                "npc_id": p.npc_id,
                "usage_count": p.usage_count,
            }
            for p in profiles
        ],
        "count": len(profiles),
    }


async def pre_generate_session_audio(
    session_id: str,
    texts: List[Dict[str, str]],
    priority: str = "normal",
    campaign_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Pre-generate audio for planned session content.

    Queue multiple text items for background audio generation.
    Use this during session preparation to pre-generate NPC dialogue,
    scene descriptions, and combat narration.

    Args:
        session_id: Session to pre-generate for
        texts: List of items to generate, each with:
            - text: The text to synthesize (required)
            - voice_profile_id: Voice profile to use (optional)
            - emotion: Emotion override (optional)
            - tag: Custom cache tag for retrieval (optional, e.g., "scene_1_intro")
        priority: Generation priority (realtime, high, normal, low)
        campaign_id: Campaign context

    Returns:
        Dict with:
        - success: bool
        - job_id: ID for tracking generation progress
        - queued_count: Number of items queued
        - status: Current job status
    """
    if not _voice_manager:
        return {"success": False, "error": "Voice manager not initialized"}

    try:
        # Parse priority
        try:
            synth_priority = SynthesisPriority(priority.lower())
        except ValueError:
            synth_priority = SynthesisPriority.NORMAL

        job_id = await _voice_manager.queue_pre_generation(
            session_id=session_id,
            texts=texts,
            priority=synth_priority,
            campaign_id=campaign_id,
        )

        return {
            "success": True,
            "job_id": job_id,
            "queued_count": len(texts),
            "status": "queued",
        }

    except Exception as e:
        logger.error("Failed to queue pre-generation", error=str(e))
        return {"success": False, "error": str(e)}


async def get_pregeneration_job_status(job_id: str) -> Dict[str, Any]:
    """
    Get status of a pre-generation job.

    Args:
        job_id: ID of the pre-generation job

    Returns:
        Dict with:
        - success: bool
        - job: Job details including progress
        - error: Error message if job not found
    """
    if not _voice_manager:
        return {"success": False, "error": "Voice manager not initialized"}

    job = _voice_manager.get_job_status(job_id)
    if job:
        return {
            "success": True,
            "job": {
                "job_id": job.job_id,
                "session_id": job.session_id,
                "status": job.status,
                "total_items": job.total_items,
                "completed_items": job.completed_items,
                "failed_items": job.failed_items,
                "progress_percent": job.progress_percent,
                "generated_cache_keys": job.generated_cache_keys,
                "errors": job.errors[:5],  # Limit errors returned
                "started_at": job.started_at.isoformat() if job.started_at else None,
                "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            },
        }

    return {"success": False, "error": f"Job '{job_id}' not found"}


async def get_cached_audio(
    text: str,
    voice_profile_id: str,
    emotion: Optional[str] = None,
    cache_tag: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Check if audio is cached and return URL if available.

    Use this to check if pre-generated audio exists before
    requesting new synthesis.

    Args:
        text: Original text that was synthesized
        voice_profile_id: Voice profile that was used
        emotion: Emotion that was used (optional)
        cache_tag: Custom cache tag to look up (alternative to text/profile)

    Returns:
        Dict with:
        - success: bool
        - cached: Whether audio was found in cache
        - audio_url: URL to retrieve audio (if cached)
        - duration_ms: Audio duration (if cached)
    """
    if not _audio_cache:
        return {"success": False, "error": "Audio cache not initialized"}

    # Try by tag first
    if cache_tag:
        cached = _audio_cache.get_by_tag(cache_tag)
        if cached:
            return {
                "success": True,
                "cached": True,
                "audio_url": f"/api/voice/cache/{cached.cache_key}",
                "duration_ms": cached.duration_ms,
                "cache_tag": cached.cache_tag,
            }

    # Try by content hash
    voice_emotion = None
    if emotion:
        try:
            voice_emotion = VoiceEmotion(emotion.lower())
        except ValueError:
            pass

    cache_key = _audio_cache.generate_cache_key(text, voice_profile_id, voice_emotion)
    cached = _audio_cache.get_cached(cache_key)

    if cached:
        return {
            "success": True,
            "cached": True,
            "audio_url": f"/api/voice/cache/{cache_key}",
            "duration_ms": cached.duration_ms,
            "cache_tag": cached.cache_tag,
        }

    return {
        "success": True,
        "cached": False,
    }


async def list_available_voices() -> Dict[str, Any]:
    """
    List all available voices across all providers.

    Returns voices from all configured and available voice
    synthesis providers.

    Returns:
        Dict with:
        - success: bool
        - voices: Dictionary of provider name to list of voice specs
    """
    if not _voice_manager:
        return {"success": False, "error": "Voice manager not initialized"}

    try:
        voices = await _voice_manager.get_all_available_voices()
        return {
            "success": True,
            "voices": voices,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


async def get_voice_synthesis_status() -> Dict[str, Any]:
    """
    Get voice synthesis system status.

    Returns status of all providers, cache statistics,
    and queue depth.

    Returns:
        Dict with:
        - success: bool
        - providers: Status of each provider
        - active_provider: Currently active provider
        - queue_depth: Number of items in pre-generation queue
        - cache: Cache statistics
    """
    if not _voice_manager:
        return {"success": False, "error": "Voice manager not initialized"}

    try:
        status = await _voice_manager.get_system_status()
        return {
            "success": True,
            **status,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
