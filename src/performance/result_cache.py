"""Result caching decorator and utilities."""

import functools
import hashlib
import inspect
import json
from typing import Any, Callable, Dict, Optional, Set, Union

from config.logging_config import get_logger
from .cache_system import CacheSystem, cache_key_generator

logger = get_logger(__name__)


class ResultCache:
    """Manages result caching for functions and methods."""
    
    def __init__(self):
        """Initialize result cache manager."""
        self.caches: Dict[str, CacheSystem] = {}
        self.cache_stats: Dict[str, Dict[str, Any]] = {}
        
        logger.info("Result cache manager initialized")
    
    def cached(
        self,
        cache_name: Optional[str] = None,
        ttl: Optional[int] = None,
        tags: Optional[Set[str]] = None,
        key_prefix: Optional[str] = None,
        ignore_params: Optional[Set[str]] = None,
        condition: Optional[Callable[..., bool]] = None,
    ) -> Callable:
        """
        Decorator for caching function results.
        
        Args:
            cache_name: Name of cache to use (creates if doesn't exist)
            ttl: Time-to-live for cached results
            tags: Tags to apply to cached entries
            key_prefix: Prefix for cache keys
            ignore_params: Parameters to ignore when generating cache key
            condition: Function to determine if result should be cached
            
        Returns:
            Decorated function
        """
        def decorator(func: Callable) -> Callable:
            # Determine cache name
            nonlocal cache_name
            if not cache_name:
                cache_name = f"{func.__module__}.{func.__name__}"
            
            # Ensure cache exists
            if cache_name not in self.caches:
                self.caches[cache_name] = CacheSystem(
                    name=cache_name,
                    max_size=100,
                    max_memory_mb=10,
                    ttl_seconds=ttl or 3600,
                )
            
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                # Check if caching should be applied
                if condition and not condition(*args, **kwargs):
                    return await func(*args, **kwargs)
                
                # Generate cache key
                cache_key = self._generate_cache_key(
                    func, args, kwargs, key_prefix, ignore_params
                )
                
                # Try to get from cache
                cache = self.caches[cache_name]
                cached_result = cache.get(cache_key)
                
                if cached_result is not None:
                    logger.debug(f"Cache hit for {func.__name__}: {cache_key}")
                    return cached_result
                
                # Execute function
                result = await func(*args, **kwargs)
                
                # Cache result
                cache.put(cache_key, result, ttl=ttl, tags=tags)
                logger.debug(f"Cached result for {func.__name__}: {cache_key}")
                
                return result
            
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                # Check if caching should be applied
                if condition and not condition(*args, **kwargs):
                    return func(*args, **kwargs)
                
                # Generate cache key
                cache_key = self._generate_cache_key(
                    func, args, kwargs, key_prefix, ignore_params
                )
                
                # Try to get from cache
                cache = self.caches[cache_name]
                cached_result = cache.get(cache_key)
                
                if cached_result is not None:
                    logger.debug(f"Cache hit for {func.__name__}: {cache_key}")
                    return cached_result
                
                # Execute function
                result = func(*args, **kwargs)
                
                # Cache result
                cache.put(cache_key, result, ttl=ttl, tags=tags)
                logger.debug(f"Cached result for {func.__name__}: {cache_key}")
                
                return result
            
            # Return appropriate wrapper
            if inspect.iscoroutinefunction(func):
                wrapper = async_wrapper
            else:
                wrapper = sync_wrapper
            
            # Add cache control methods
            wrapper.invalidate_cache = lambda: self.invalidate_function_cache(cache_name)
            wrapper.get_cache_stats = lambda: self.get_function_cache_stats(cache_name)
            wrapper.cache_name = cache_name
            
            return wrapper
        
        return decorator
    
    def batch_cached(
        self,
        cache_name: Optional[str] = None,
        ttl: Optional[int] = None,
        batch_key_func: Optional[Callable] = None,
    ) -> Callable:
        """
        Decorator for caching batch operation results.
        
        Args:
            cache_name: Name of cache to use
            ttl: Time-to-live for cached results
            batch_key_func: Function to extract key from batch item
            
        Returns:
            Decorated function
        """
        def decorator(func: Callable) -> Callable:
            # Determine cache name
            nonlocal cache_name
            if not cache_name:
                cache_name = f"{func.__module__}.{func.__name__}_batch"
            
            # Ensure cache exists
            if cache_name not in self.caches:
                self.caches[cache_name] = CacheSystem(
                    name=cache_name,
                    max_size=500,
                    max_memory_mb=20,
                    ttl_seconds=ttl or 3600,
                )
            
            @functools.wraps(func)
            async def async_wrapper(items: list, *args, **kwargs):
                cache = self.caches[cache_name]
                cached_results = {}
                uncached_items = []
                
                # Check cache for each item
                for item in items:
                    if batch_key_func:
                        item_key = batch_key_func(item)
                    else:
                        item_key = str(item)
                    
                    cached = cache.get(item_key)
                    if cached is not None:
                        cached_results[item_key] = cached
                    else:
                        uncached_items.append(item)
                
                # Process uncached items
                if uncached_items:
                    new_results = await func(uncached_items, *args, **kwargs)
                    
                    # Cache new results
                    for item, result in zip(uncached_items, new_results):
                        if batch_key_func:
                            item_key = batch_key_func(item)
                        else:
                            item_key = str(item)
                        
                        cache.put(item_key, result, ttl=ttl)
                        cached_results[item_key] = result
                
                # Return results in original order
                results = []
                for item in items:
                    if batch_key_func:
                        item_key = batch_key_func(item)
                    else:
                        item_key = str(item)
                    results.append(cached_results.get(item_key))
                
                return results
            
            @functools.wraps(func)
            def sync_wrapper(items: list, *args, **kwargs):
                cache = self.caches[cache_name]
                cached_results = {}
                uncached_items = []
                
                # Check cache for each item
                for item in items:
                    if batch_key_func:
                        item_key = batch_key_func(item)
                    else:
                        item_key = str(item)
                    
                    cached = cache.get(item_key)
                    if cached is not None:
                        cached_results[item_key] = cached
                    else:
                        uncached_items.append(item)
                
                # Process uncached items
                if uncached_items:
                    new_results = func(uncached_items, *args, **kwargs)
                    
                    # Cache new results
                    for item, result in zip(uncached_items, new_results):
                        if batch_key_func:
                            item_key = batch_key_func(item)
                        else:
                            item_key = str(item)
                        
                        cache.put(item_key, result, ttl=ttl)
                        cached_results[item_key] = result
                
                # Return results in original order
                results = []
                for item in items:
                    if batch_key_func:
                        item_key = batch_key_func(item)
                    else:
                        item_key = str(item)
                    results.append(cached_results.get(item_key))
                
                return results
            
            # Return appropriate wrapper
            if inspect.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
        
        return decorator
    
    def invalidate_function_cache(self, cache_name: str) -> None:
        """
        Invalidate all entries for a function cache.
        
        Args:
            cache_name: Name of the cache to invalidate
        """
        if cache_name in self.caches:
            self.caches[cache_name].clear()
            logger.info(f"Invalidated cache for {cache_name}")
    
    def get_function_cache_stats(self, cache_name: str) -> Dict[str, Any]:
        """
        Get statistics for a function cache.
        
        Args:
            cache_name: Name of the cache
            
        Returns:
            Cache statistics
        """
        if cache_name in self.caches:
            return self.caches[cache_name].get_stats()
        return {}
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for all function caches.
        
        Returns:
            Dictionary mapping cache names to statistics
        """
        return {
            name: cache.get_stats()
            for name, cache in self.caches.items()
        }
    
    def clear_all(self) -> None:
        """Clear all function caches."""
        for cache in self.caches.values():
            cache.clear()
        logger.info("Cleared all result caches")
    
    def _generate_cache_key(
        self,
        func: Callable,
        args: tuple,
        kwargs: dict,
        key_prefix: Optional[str],
        ignore_params: Optional[Set[str]],
    ) -> str:
        """Generate cache key for function call."""
        # Get function signature
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        
        # Filter ignored parameters
        if ignore_params:
            for param in ignore_params:
                bound_args.arguments.pop(param, None)
        
        # Build key components
        key_parts = []
        
        if key_prefix:
            key_parts.append(key_prefix)
        
        # Add function identifier
        key_parts.append(f"{func.__module__}.{func.__name__}")
        
        # Add arguments
        for name, value in bound_args.arguments.items():
            if isinstance(value, (str, int, float, bool, type(None))):
                key_parts.append(f"{name}={value}")
            else:
                # Hash complex objects
                value_str = json.dumps(value, sort_keys=True, default=str)
                value_hash = hashlib.md5(value_str.encode()).hexdigest()[:8]
                key_parts.append(f"{name}={value_hash}")
        
        return ":".join(key_parts)


# Global result cache instance
result_cache = ResultCache()

# Export decorator directly
cached = result_cache.cached
batch_cached = result_cache.batch_cached