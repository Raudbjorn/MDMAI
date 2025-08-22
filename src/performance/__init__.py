"""Performance optimization module for TTRPG Assistant."""

from .cache_system import (
    CacheSystem,
    CachePolicy,
    CacheEntry,
    CacheStatistics,
    cache_key_generator,
)
from .cache_invalidator import CacheInvalidator
from .cache_config import CacheConfiguration
from .result_cache import ResultCache, cached, batch_cached
from .cache_manager import GlobalCacheManager
from .mcp_tools import initialize_performance_tools, register_performance_tools

__all__ = [
    'CacheSystem',
    'CachePolicy',
    'CacheEntry',
    'CacheStatistics',
    'cache_key_generator',
    'CacheInvalidator',
    'CacheConfiguration',
    'ResultCache',
    'cached',
    'batch_cached',
    'GlobalCacheManager',
    'initialize_performance_tools',
    'register_performance_tools',
]