"""Cache configuration management."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from config.logging_config import get_logger
from config.settings import settings

logger = get_logger(__name__)


class CacheType(Enum):
    """Types of caches in the system."""
    SEARCH = "search"
    EMBEDDING = "embedding"
    CAMPAIGN = "campaign"
    SESSION = "session"
    CHARACTER = "character"
    SOURCE = "source"
    PERSONALITY = "personality"
    CROSS_REFERENCE = "cross_reference"
    GENERAL = "general"


@dataclass
class CacheProfile:
    """Cache configuration profile."""
    name: str
    cache_type: CacheType
    max_size: int = 1000
    max_memory_mb: int = 50
    ttl_seconds: int = 3600
    policy: str = "lru"
    persistent: bool = False
    auto_cleanup_interval: int = 300
    priority_threshold: int = 0
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "cache_type": self.cache_type.value,
            "max_size": self.max_size,
            "max_memory_mb": self.max_memory_mb,
            "ttl_seconds": self.ttl_seconds,
            "policy": self.policy,
            "persistent": self.persistent,
            "auto_cleanup_interval": self.auto_cleanup_interval,
            "priority_threshold": self.priority_threshold,
            "tags": self.tags,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CacheProfile":
        """Create from dictionary."""
        data["cache_type"] = CacheType(data["cache_type"])
        return cls(**data)


class CacheConfiguration:
    """Manages cache configuration for the entire system."""
    
    def __init__(self, config_file: Optional[Path] = None):
        """
        Initialize cache configuration.
        
        Args:
            config_file: Optional configuration file path
        """
        self.config_file = config_file or settings.cache_dir / "cache_config.yaml"
        self.profiles: Dict[str, CacheProfile] = {}
        self.global_settings: Dict[str, Any] = {}
        
        # Load configuration
        self._load_config()
        
        # Set up default profiles if none exist
        if not self.profiles:
            self._setup_default_profiles()
            self._save_config()
        
        logger.info(f"Cache configuration initialized with {len(self.profiles)} profiles")
    
    def get_profile(self, name: str) -> Optional[CacheProfile]:
        """
        Get a cache profile by name.
        
        Args:
            name: Profile name
            
        Returns:
            Cache profile or None
        """
        return self.profiles.get(name)
    
    def get_profiles_by_type(self, cache_type: CacheType) -> List[CacheProfile]:
        """
        Get all profiles of a specific type.
        
        Args:
            cache_type: Cache type to filter by
            
        Returns:
            List of matching profiles
        """
        return [
            p for p in self.profiles.values()
            if p.cache_type == cache_type
        ]
    
    def add_profile(self, profile: CacheProfile) -> None:
        """
        Add or update a cache profile.
        
        Args:
            profile: Cache profile to add
        """
        self.profiles[profile.name] = profile
        self._save_config()
        logger.info(f"Added cache profile: {profile.name}")
    
    def remove_profile(self, name: str) -> bool:
        """
        Remove a cache profile.
        
        Args:
            name: Profile name to remove
            
        Returns:
            True if removed, False if not found
        """
        if name in self.profiles:
            del self.profiles[name]
            self._save_config()
            logger.info(f"Removed cache profile: {name}")
            return True
        return False
    
    def update_global_setting(self, key: str, value: Any) -> None:
        """
        Update a global cache setting.
        
        Args:
            key: Setting key
            value: Setting value
        """
        self.global_settings[key] = value
        self._save_config()
        logger.info(f"Updated global setting: {key}={value}")
    
    def get_global_setting(self, key: str, default: Any = None) -> Any:
        """
        Get a global cache setting.
        
        Args:
            key: Setting key
            default: Default value if not found
            
        Returns:
            Setting value or default
        """
        return self.global_settings.get(key, default)
    
    def optimize_memory_allocation(self, total_memory_mb: int) -> Dict[str, int]:
        """
        Optimize memory allocation across cache profiles.
        
        Args:
            total_memory_mb: Total memory budget in MB
            
        Returns:
            Dictionary mapping profile names to allocated memory
        """
        allocations = {}
        
        # Calculate weights based on cache importance
        weights = {
            CacheType.EMBEDDING: 0.25,  # Embeddings are expensive to compute
            CacheType.SEARCH: 0.20,  # Search results are frequently accessed
            CacheType.CAMPAIGN: 0.15,  # Campaign data is important
            CacheType.SESSION: 0.10,  # Session data is temporary
            CacheType.CHARACTER: 0.10,  # Character data is moderate
            CacheType.SOURCE: 0.10,  # Source data is large but less frequent
            CacheType.PERSONALITY: 0.05,  # Personality data is small
            CacheType.CROSS_REFERENCE: 0.05,  # Cross-references are derived
        }
        
        # Allocate memory based on weights
        for profile in self.profiles.values():
            weight = weights.get(profile.cache_type, 0.05)
            allocated = int(total_memory_mb * weight)
            allocations[profile.name] = allocated
            profile.max_memory_mb = allocated
        
        # Save updated profiles
        self._save_config()
        
        logger.info(f"Optimized memory allocation across {len(allocations)} profiles")
        return allocations
    
    def get_cache_recommendations(self) -> Dict[str, Any]:
        """
        Get recommendations for cache configuration.
        
        Returns:
            Dictionary with recommendations
        """
        recommendations = {
            "profiles": [],
            "global": [],
            "memory": {},
        }
        
        # Analyze each profile
        for profile in self.profiles.values():
            profile_recs = []
            
            # Check TTL
            if profile.cache_type == CacheType.EMBEDDING and profile.ttl_seconds < 86400:
                profile_recs.append(
                    "Consider increasing TTL for embeddings (expensive to compute)"
                )
            elif profile.cache_type == CacheType.SESSION and profile.ttl_seconds > 7200:
                profile_recs.append(
                    "Consider reducing TTL for sessions (temporary data)"
                )
            
            # Check persistence
            if profile.cache_type in [CacheType.EMBEDDING, CacheType.SOURCE] and not profile.persistent:
                profile_recs.append(
                    "Consider enabling persistence for expensive-to-compute data"
                )
            
            # Check policy
            if profile.cache_type == CacheType.SEARCH and profile.policy != "lru":
                profile_recs.append(
                    "LRU policy recommended for search cache"
                )
            
            if profile_recs:
                recommendations["profiles"].append({
                    "name": profile.name,
                    "recommendations": profile_recs,
                })
        
        # Global recommendations
        total_memory = sum(p.max_memory_mb for p in self.profiles.values())
        
        if total_memory > settings.cache_max_memory_mb * 1.5:
            recommendations["global"].append(
                f"Total allocated memory ({total_memory}MB) exceeds recommended limit"
            )
        
        if not self.global_settings.get("enable_compression", False):
            recommendations["global"].append(
                "Consider enabling compression for large cache entries"
            )
        
        # Memory recommendations
        recommendations["memory"] = {
            "total_allocated": total_memory,
            "system_limit": settings.cache_max_memory_mb,
            "utilization": f"{(total_memory / settings.cache_max_memory_mb * 100):.1f}%",
        }
        
        return recommendations
    
    def export_config(self, path: Path) -> None:
        """
        Export configuration to file.
        
        Args:
            path: Path to export to
        """
        config = {
            "global_settings": self.global_settings,
            "profiles": {
                name: profile.to_dict()
                for name, profile in self.profiles.items()
            },
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        logger.info(f"Exported cache configuration to {path}")
    
    def import_config(self, path: Path) -> None:
        """
        Import configuration from file.
        
        Args:
            path: Path to import from
        """
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.global_settings = config.get("global_settings", {})
        
        self.profiles = {}
        for name, profile_data in config.get("profiles", {}).items():
            self.profiles[name] = CacheProfile.from_dict(profile_data)
        
        self._save_config()
        logger.info(f"Imported cache configuration from {path}")
    
    def _load_config(self) -> None:
        """Load configuration from file."""
        if not self.config_file.exists():
            return
        
        try:
            with open(self.config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            self.global_settings = config.get("global_settings", {})
            
            for name, profile_data in config.get("profiles", {}).items():
                self.profiles[name] = CacheProfile.from_dict(profile_data)
            
            logger.info(f"Loaded cache configuration from {self.config_file}")
        except (yaml.YAMLError, FileNotFoundError, PermissionError) as e:
    
    def _save_config(self) -> None:
        """Save configuration to file."""
        try:
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            
            config = {
                "global_settings": self.global_settings,
                "profiles": {
                    name: profile.to_dict()
                    for name, profile in self.profiles.items()
                },
            }
            
            with open(self.config_file, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            
        except Exception as e:
            logger.error(f"Failed to save cache configuration: {e}")
    
    def _setup_default_profiles(self) -> None:
        """Set up default cache profiles."""
        # Search cache - frequently accessed, medium size
        self.profiles["search_main"] = CacheProfile(
            name="search_main",
            cache_type=CacheType.SEARCH,
            max_size=1000,
            max_memory_mb=30,
            ttl_seconds=3600,
            policy="lru",
            persistent=False,
        )
        
        # Embedding cache - expensive to compute, should persist
        self.profiles["embedding_main"] = CacheProfile(
            name="embedding_main",
            cache_type=CacheType.EMBEDDING,
            max_size=5000,
            max_memory_mb=50,
            ttl_seconds=86400,  # 24 hours
            policy="lfu",  # Keep frequently used embeddings
            persistent=True,
        )
        
        # Campaign cache - important data, moderate size
        self.profiles["campaign_main"] = CacheProfile(
            name="campaign_main",
            cache_type=CacheType.CAMPAIGN,
            max_size=500,
            max_memory_mb=20,
            ttl_seconds=7200,  # 2 hours
            policy="lru",
            persistent=True,
        )
        
        # Session cache - temporary data, small size
        self.profiles["session_main"] = CacheProfile(
            name="session_main",
            cache_type=CacheType.SESSION,
            max_size=100,
            max_memory_mb=10,
            ttl_seconds=1800,  # 30 minutes
            policy="fifo",
            persistent=False,
        )
        
        # Character cache - moderate importance
        self.profiles["character_main"] = CacheProfile(
            name="character_main",
            cache_type=CacheType.CHARACTER,
            max_size=200,
            max_memory_mb=15,
            ttl_seconds=3600,
            policy="lru",
            persistent=False,
        )
        
        # Source cache - large but infrequent
        self.profiles["source_main"] = CacheProfile(
            name="source_main",
            cache_type=CacheType.SOURCE,
            max_size=100,
            max_memory_mb=25,
            ttl_seconds=14400,  # 4 hours
            policy="lru",
            persistent=True,
        )
        
        # Personality cache - small and stable
        self.profiles["personality_main"] = CacheProfile(
            name="personality_main",
            cache_type=CacheType.PERSONALITY,
            max_size=50,
            max_memory_mb=5,
            ttl_seconds=28800,  # 8 hours
            policy="lru",
            persistent=True,
        )
        
        # Cross-reference cache - derived data
        self.profiles["cross_ref_main"] = CacheProfile(
            name="cross_ref_main",
            cache_type=CacheType.CROSS_REFERENCE,
            max_size=300,
            max_memory_mb=10,
            ttl_seconds=1800,  # 30 minutes
            policy="lru",
            persistent=False,
        )
        
        # Set global settings
        self.global_settings = {
            "enable_compression": False,
            "enable_statistics": True,
            "statistics_interval": 300,
            "max_total_memory_mb": settings.cache_max_memory_mb,
            "enable_auto_optimization": True,
            "optimization_interval": 3600,
        }