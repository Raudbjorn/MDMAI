"""Audio cache manager for pre-generated voice synthesis."""

import hashlib
import json
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from structlog import get_logger

from .models import AudioFormat, CachedAudio, VoiceEmotion, VoiceProviderType

logger = get_logger(__name__)

# Default cache directory relative to project
DEFAULT_CACHE_DIR = Path("data/voice_cache")


class AudioCacheManager:
    """Manages pre-generated and cached audio files.

    This manager handles storage, retrieval, and cleanup of synthesized
    audio files. It supports tagging for session-based retrieval and
    automatic cleanup of old entries.

    The cache is designed for the session prep workflow where most audio
    is pre-generated before a session, then played back during the session.
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        max_size_mb: int = 5000,
        max_age_days: int = 30,
    ):
        """Initialize the cache manager.

        Args:
            cache_dir: Directory to store cached audio files
            max_size_mb: Maximum cache size in megabytes
            max_age_days: Maximum age of cache entries in days
        """
        self.cache_dir = cache_dir or DEFAULT_CACHE_DIR
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.max_age_days = max_age_days

        self.index_file = self.cache_dir / "cache_index.json"
        self._index: Dict[str, CachedAudio] = {}

        self._load_index()

        logger.info(
            "Audio cache manager initialized",
            cache_dir=str(self.cache_dir),
            entries=len(self._index),
            max_size_mb=max_size_mb,
        )

    def generate_cache_key(
        self,
        text: str,
        voice_profile_id: str,
        emotion: Optional[VoiceEmotion] = None,
    ) -> str:
        """Generate a deterministic cache key.

        The key is based on the text content, voice profile, and emotion
        to ensure the same inputs always produce the same key.

        Args:
            text: The text that was synthesized
            voice_profile_id: ID of the voice profile used
            emotion: Optional emotion override

        Returns:
            16-character hex string cache key
        """
        emotion_str = emotion.value if emotion else "default"
        key_parts = f"{voice_profile_id}:{emotion_str}:{text}"
        return hashlib.sha256(key_parts.encode()).hexdigest()[:16]

    def get_cached(self, cache_key: str) -> Optional[CachedAudio]:
        """Retrieve cached audio entry if exists.

        Updates access tracking on successful retrieval.

        Args:
            cache_key: The cache key to look up

        Returns:
            CachedAudio entry or None if not found
        """
        if cache_key not in self._index:
            return None

        entry = self._index[cache_key]

        # Verify file still exists
        audio_path = self.cache_dir / entry.audio_path
        if not audio_path.exists():
            logger.warning("Cache entry file missing, removing", cache_key=cache_key)
            del self._index[cache_key]
            self._save_index()
            return None

        # Update access tracking
        entry.last_accessed = datetime.utcnow()
        entry.access_count += 1
        self._save_index()

        return entry

    def get_by_tag(self, cache_tag: str) -> Optional[CachedAudio]:
        """Retrieve cached audio by custom tag.

        Tags are used for session-based retrieval, e.g., "npc_bob_greeting".

        Args:
            cache_tag: The tag to search for

        Returns:
            CachedAudio entry or None if not found
        """
        for entry in self._index.values():
            if entry.cache_tag == cache_tag:
                # Verify file exists
                audio_path = self.cache_dir / entry.audio_path
                if audio_path.exists():
                    entry.last_accessed = datetime.utcnow()
                    entry.access_count += 1
                    self._save_index()
                    return entry
        return None

    def get_audio_path(self, cache_key: str) -> Optional[Path]:
        """Get full path to cached audio file.

        Args:
            cache_key: The cache key

        Returns:
            Path to audio file or None if not cached
        """
        entry = self.get_cached(cache_key)
        if entry:
            return self.cache_dir / entry.audio_path
        return None

    def get_audio_bytes(self, cache_key: str) -> Optional[bytes]:
        """Get cached audio data as bytes.

        Args:
            cache_key: The cache key

        Returns:
            Audio data bytes or None if not cached
        """
        path = self.get_audio_path(cache_key)
        if path and path.exists():
            return path.read_bytes()
        return None

    async def store_audio(
        self,
        cache_key: str,
        audio_data: bytes,
        voice_profile_id: str,
        text: str,
        format: AudioFormat = AudioFormat.MP3,
        duration_ms: float = 0.0,
        sample_rate: int = 22050,
        provider_type: Optional[VoiceProviderType] = None,
        cache_tag: Optional[str] = None,
        session_id: Optional[str] = None,
        campaign_id: Optional[str] = None,
        npc_id: Optional[str] = None,
        scene_id: Optional[str] = None,
    ) -> CachedAudio:
        """Store audio in cache.

        Args:
            cache_key: Unique cache key
            audio_data: Raw audio bytes
            voice_profile_id: ID of voice profile used
            text: Original text (for hash verification)
            format: Audio format
            duration_ms: Audio duration in milliseconds
            sample_rate: Audio sample rate
            provider_type: The provider that generated the audio
            cache_tag: Optional custom tag for retrieval
            session_id: Optional session context
            campaign_id: Optional campaign context
            npc_id: Optional NPC context
            scene_id: Optional scene context

        Returns:
            CachedAudio entry
        """
        # Create subdirectory structure by campaign/session
        if campaign_id:
            subdir = self.cache_dir / campaign_id
            if session_id:
                subdir = subdir / session_id
        elif session_id:
            subdir = self.cache_dir / "sessions" / session_id
        else:
            subdir = self.cache_dir / "general"

        subdir.mkdir(parents=True, exist_ok=True)

        # Write audio file
        filename = f"{cache_key}.{format.value}"
        filepath = subdir / filename
        filepath.write_bytes(audio_data)

        # Create cache entry
        entry = CachedAudio(
            cache_key=cache_key,
            audio_path=str(filepath.relative_to(self.cache_dir)),
            voice_profile_id=voice_profile_id,
            text_hash=hashlib.sha256(text.encode()).hexdigest(),
            duration_ms=duration_ms,
            file_size_bytes=len(audio_data),
            format=format,
            sample_rate=sample_rate,
            provider_type=provider_type or VoiceProviderType.SYSTEM,
            cache_tag=cache_tag,
            session_id=session_id,
            campaign_id=campaign_id,
            npc_id=npc_id,
            scene_id=scene_id,
        )

        self._index[cache_key] = entry
        self._save_index()

        logger.debug(
            "Stored audio in cache",
            cache_key=cache_key,
            cache_tag=cache_tag,
            size_bytes=len(audio_data),
            duration_ms=duration_ms,
        )

        return entry

    def get_session_audio(self, session_id: str) -> List[CachedAudio]:
        """Get all cached audio for a session.

        Args:
            session_id: Session ID to filter by

        Returns:
            List of CachedAudio entries for the session
        """
        return [
            entry
            for entry in self._index.values()
            if entry.session_id == session_id
        ]

    def get_campaign_audio(self, campaign_id: str) -> List[CachedAudio]:
        """Get all cached audio for a campaign.

        Args:
            campaign_id: Campaign ID to filter by

        Returns:
            List of CachedAudio entries for the campaign
        """
        return [
            entry
            for entry in self._index.values()
            if entry.campaign_id == campaign_id
        ]

    def get_npc_audio(self, npc_id: str) -> List[CachedAudio]:
        """Get all cached audio for an NPC.

        Args:
            npc_id: NPC ID to filter by

        Returns:
            List of CachedAudio entries for the NPC
        """
        return [
            entry
            for entry in self._index.values()
            if entry.npc_id == npc_id
        ]

    def delete_entry(self, cache_key: str) -> bool:
        """Delete a specific cache entry.

        Args:
            cache_key: Cache key to delete

        Returns:
            True if entry was deleted, False if not found
        """
        if cache_key not in self._index:
            return False

        entry = self._index[cache_key]
        audio_path = self.cache_dir / entry.audio_path

        if audio_path.exists():
            audio_path.unlink()

        del self._index[cache_key]
        self._save_index()

        logger.debug("Deleted cache entry", cache_key=cache_key)
        return True

    def delete_session_audio(self, session_id: str) -> int:
        """Delete all cached audio for a session.

        Args:
            session_id: Session ID to delete audio for

        Returns:
            Number of entries deleted
        """
        entries = self.get_session_audio(session_id)
        for entry in entries:
            self.delete_entry(entry.cache_key)
        return len(entries)

    def cleanup_old_cache(self, max_age_days: Optional[int] = None) -> int:
        """Remove cache entries older than max_age_days.

        Args:
            max_age_days: Override default max age

        Returns:
            Number of entries removed
        """
        age_limit = max_age_days or self.max_age_days
        cutoff = datetime.utcnow() - timedelta(days=age_limit)
        removed = 0

        keys_to_remove = []
        for key, entry in self._index.items():
            if entry.last_accessed < cutoff:
                keys_to_remove.append(key)
                audio_path = self.cache_dir / entry.audio_path
                if audio_path.exists():
                    audio_path.unlink()
                removed += 1

        for key in keys_to_remove:
            del self._index[key]

        if removed > 0:
            self._save_index()
            logger.info("Cleaned up old cache entries", removed=removed)

        return removed

    def cleanup_by_size(self) -> int:
        """Remove oldest entries if cache exceeds max size.

        Uses LRU (least recently used) eviction strategy.

        Returns:
            Number of entries removed
        """
        current_size = self._get_total_size()
        if current_size <= self.max_size_bytes:
            return 0

        # Sort by last accessed (oldest first)
        sorted_entries = sorted(
            self._index.items(),
            key=lambda x: x[1].last_accessed,
        )

        removed = 0
        for key, entry in sorted_entries:
            if current_size <= self.max_size_bytes:
                break

            audio_path = self.cache_dir / entry.audio_path
            if audio_path.exists():
                current_size -= entry.file_size_bytes
                audio_path.unlink()

            del self._index[key]
            removed += 1

        if removed > 0:
            self._save_index()
            logger.info("Cleaned up cache by size", removed=removed)

        return removed

    def clear_all(self) -> int:
        """Clear all cached audio.

        Returns:
            Number of entries cleared
        """
        count = len(self._index)

        # Remove all files
        for entry in self._index.values():
            audio_path = self.cache_dir / entry.audio_path
            if audio_path.exists():
                audio_path.unlink()

        # Clear subdirectories (but keep cache_dir itself)
        for item in self.cache_dir.iterdir():
            if item.is_dir() and item.name != ".":
                shutil.rmtree(item, ignore_errors=True)
            elif item.is_file() and item.name != "cache_index.json":
                item.unlink()

        self._index.clear()
        self._save_index()

        logger.info("Cleared all cache entries", count=count)
        return count

    def get_stats(self) -> Dict:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        total_size = self._get_total_size()
        total_duration = sum(e.duration_ms for e in self._index.values())

        # Count by category
        by_campaign: Dict[str, int] = {}
        by_session: Dict[str, int] = {}
        by_npc: Dict[str, int] = {}

        for entry in self._index.values():
            if entry.campaign_id:
                by_campaign[entry.campaign_id] = by_campaign.get(entry.campaign_id, 0) + 1
            if entry.session_id:
                by_session[entry.session_id] = by_session.get(entry.session_id, 0) + 1
            if entry.npc_id:
                by_npc[entry.npc_id] = by_npc.get(entry.npc_id, 0) + 1

        oldest_entry = min(
            (e.created_at for e in self._index.values()),
            default=None,
        )
        newest_entry = max(
            (e.created_at for e in self._index.values()),
            default=None,
        )

        return {
            "total_entries": len(self._index),
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "total_duration_minutes": round(total_duration / 60000, 2),
            "max_size_mb": round(self.max_size_bytes / (1024 * 1024), 2),
            "usage_percent": round((total_size / self.max_size_bytes) * 100, 1) if self.max_size_bytes > 0 else 0,
            "campaigns": len(by_campaign),
            "sessions": len(by_session),
            "npcs": len(by_npc),
            "oldest_entry": oldest_entry.isoformat() if oldest_entry else None,
            "newest_entry": newest_entry.isoformat() if newest_entry else None,
        }

    def _get_total_size(self) -> int:
        """Calculate total size of cached files."""
        return sum(e.file_size_bytes for e in self._index.values())

    def _load_index(self) -> None:
        """Load cache index from disk."""
        if not self.index_file.exists():
            self._index = {}
            return

        try:
            data = json.loads(self.index_file.read_text())
            self._index = {}

            for key, entry_data in data.items():
                try:
                    self._index[key] = CachedAudio.from_dict(entry_data)
                except Exception as e:
                    logger.warning(
                        "Failed to load cache entry",
                        cache_key=key,
                        error=str(e),
                    )

        except Exception as e:
            logger.error("Failed to load cache index", error=str(e))
            self._index = {}

    def _save_index(self) -> None:
        """Save cache index to disk."""
        try:
            data = {k: v.to_dict() for k, v in self._index.items()}
            self.index_file.write_text(json.dumps(data, indent=2, default=str))
        except Exception as e:
            logger.error("Failed to save cache index", error=str(e))

    def __len__(self) -> int:
        """Return number of cached entries."""
        return len(self._index)

    def __contains__(self, cache_key: str) -> bool:
        """Check if cache key exists."""
        return cache_key in self._index
