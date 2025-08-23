"""Performance optimization module for TTRPG Assistant."""

from .cache_config import CacheConfiguration
from .cache_invalidator import CacheInvalidator
from .cache_manager import GlobalCacheManager
from .cache_system import (
    CacheEntry,
    CachePolicy,
    CacheStatistics,
    CacheSystem,
    cache_key_generator,
)
from .mcp_tools import initialize_performance_tools, register_performance_tools
from .result_cache import ResultCache, batch_cached, cached

__all__ = [
    "CacheSystem",
    "CachePolicy",
    "CacheEntry",
    "CacheStatistics",
    "cache_key_generator",
    "CacheInvalidator",
    "CacheConfiguration",
    "ResultCache",
    "cached",
    "batch_cached",
    "GlobalCacheManager",
    "initialize_performance_tools",
    "register_performance_tools",
]
