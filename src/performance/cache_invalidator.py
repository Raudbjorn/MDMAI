"""Cache invalidation logic and strategies."""

import fnmatch
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

from config.logging_config import get_logger

logger = get_logger(__name__)


class InvalidationStrategy(Enum):
    """Cache invalidation strategies."""
    IMMEDIATE = "immediate"  # Invalidate immediately
    LAZY = "lazy"  # Invalidate on next access
    SCHEDULED = "scheduled"  # Invalidate at scheduled time
    CASCADE = "cascade"  # Invalidate related entries
    SMART = "smart"  # Intelligent invalidation based on patterns


@dataclass
class InvalidationRule:
    """Rule for cache invalidation."""
    name: str
    pattern: Optional[str] = None  # Key pattern to match
    tags: Optional[Set[str]] = None  # Tags to match
    max_age: Optional[int] = None  # Maximum age in seconds
    max_accesses: Optional[int] = None  # Maximum number of accesses
    condition: Optional[Callable[[Any], bool]] = None  # Custom condition
    strategy: InvalidationStrategy = InvalidationStrategy.IMMEDIATE
    cascade_tags: Optional[Set[str]] = None  # Tags to cascade invalidation to
    
    def matches(self, key: str, entry_info: Dict[str, Any]) -> bool:
        """Check if rule matches a cache entry."""
        # Check pattern
        if self.pattern and not self._match_pattern(key, self.pattern):
            return False
        
        # Check tags
        if self.tags:
            entry_tags = set(entry_info.get("tags", []))
            if not self.tags.intersection(entry_tags):
                return False
        
        # Check age
        if self.max_age:
            age = entry_info.get("age_seconds", 0)
            if age < self.max_age:
                return False
        
        # Check access count
        if self.max_accesses:
            accesses = entry_info.get("access_count", 0)
            if accesses < self.max_accesses:
                return False
        
        # Check custom condition
        if self.condition:
            if not self.condition(entry_info):
                return False
        
        return True
    
    def _match_pattern(self, key: str, pattern: str) -> bool:
        """Match key against pattern (supports wildcards)."""
        return fnmatch.fnmatch(key, pattern)


class CacheInvalidator:
    """Manages cache invalidation across the system."""
    
    def __init__(self):
        """Initialize cache invalidator."""
        self.rules: Dict[str, InvalidationRule] = {}
        self.cache_systems: Dict[str, Any] = {}  # Reference to cache systems
        self.invalidation_history = deque(maxlen=1000)  # Automatically limited to 1000 entries
        self.scheduled_invalidations: List[Dict[str, Any]] = []
        
        # Default rules
        self._setup_default_rules()
        
        logger.info("Cache invalidator initialized")
    
    def register_cache(self, name: str, cache_system: Any) -> None:
        """
        Register a cache system for invalidation.
        
        Args:
            name: Cache name
            cache_system: Cache system instance
        """
        self.cache_systems[name] = cache_system
        logger.debug(f"Registered cache system: {name}")
    
    def add_rule(self, rule: InvalidationRule) -> None:
        """
        Add an invalidation rule.
        
        Args:
            rule: Invalidation rule to add
        """
        self.rules[rule.name] = rule
        logger.debug(f"Added invalidation rule: {rule.name}")
    
    def invalidate(
        self,
        cache_name: Optional[str] = None,
        key: Optional[str] = None,
        tags: Optional[Set[str]] = None,
        rule_name: Optional[str] = None,
        cascade: bool = True,
    ) -> Dict[str, int]:
        """
        Invalidate cache entries.
        
        Args:
            cache_name: Specific cache to invalidate (None for all)
            key: Specific key to invalidate
            tags: Tags to invalidate
            rule_name: Apply specific rule
            cascade: Whether to cascade invalidation
            
        Returns:
            Dictionary mapping cache names to number of invalidated entries
        """
        results = {}
        
        # Determine which caches to target
        if cache_name:
            target_caches = {cache_name: self.cache_systems.get(cache_name)}
            if not target_caches[cache_name]:
                logger.warning(f"Cache '{cache_name}' not found")
                return results
        else:
            target_caches = self.cache_systems
        
        # Apply rule if specified
        if rule_name:
            rule = self.rules.get(rule_name)
            if not rule:
                logger.warning(f"Rule '{rule_name}' not found")
                return results
            
            for name, cache in target_caches.items():
                if cache:
                    count = self._apply_rule(cache, rule)
                    if count > 0:
                        results[name] = count
        else:
            # Direct invalidation
            for name, cache in target_caches.items():
                if cache:
                    count = cache.invalidate(key=key, tags=tags)
                    if count > 0:
                        results[name] = count
        
        # Handle cascading
        if cascade and tags:
            cascade_results = self._cascade_invalidation(tags)
            for name, count in cascade_results.items():
                results[name] = results.get(name, 0) + count
        
        # Record history
        self._record_invalidation(results, key, tags, rule_name)
        
        return results
    
    def invalidate_by_pattern(
        self,
        pattern: str,
        cache_name: Optional[str] = None,
    ) -> Dict[str, int]:
        """
        Invalidate entries matching a pattern.
        
        Args:
            pattern: Pattern to match (supports wildcards)
            cache_name: Specific cache to target
            
        Returns:
            Dictionary mapping cache names to number of invalidated entries
        """
        results = {}
        
        # Create temporary rule
        rule = InvalidationRule(
            name=f"pattern_{pattern}",
            pattern=pattern,
            strategy=InvalidationStrategy.IMMEDIATE,
        )
        
        # Determine which caches to target
        if cache_name:
            target_caches = {cache_name: self.cache_systems.get(cache_name)}
        else:
            target_caches = self.cache_systems
        
        for name, cache in target_caches.items():
            if cache:
                count = self._apply_rule(cache, rule)
                if count > 0:
                    results[name] = count
        
        return results
    
    def invalidate_stale(
        self,
        max_age_seconds: int = 3600,
        max_idle_seconds: Optional[int] = None,
    ) -> Dict[str, int]:
        """
        Invalidate stale entries across all caches.
        
        Args:
            max_age_seconds: Maximum age for entries
            max_idle_seconds: Maximum idle time for entries
            
        Returns:
            Dictionary mapping cache names to number of invalidated entries
        """
        results = {}
        
        for name, cache in self.cache_systems.items():
            if not cache:
                continue
            
            count = 0
            # Get all entries and check staleness
            with cache.lock:
                keys_to_invalidate = []
                for key, entry in cache.cache.items():
                    if entry.get_age() > max_age_seconds:
                        keys_to_invalidate.append(key)
                    elif max_idle_seconds and entry.get_idle_time() > max_idle_seconds:
                        keys_to_invalidate.append(key)
                
                for key in keys_to_invalidate:
                    cache._remove_entry(key)
                    count += 1
            
            if count > 0:
                results[name] = count
                logger.info(f"Invalidated {count} stale entries in '{name}'")
        
        return results
    
    def schedule_invalidation(
        self,
        when: datetime,
        cache_name: Optional[str] = None,
        key: Optional[str] = None,
        tags: Optional[Set[str]] = None,
        rule_name: Optional[str] = None,
    ) -> str:
        """
        Schedule a future invalidation.
        
        Args:
            when: When to perform invalidation
            cache_name: Cache to target
            key: Key to invalidate
            tags: Tags to invalidate
            rule_name: Rule to apply
            
        Returns:
            Schedule ID
        """
        import uuid
        
        schedule_id = str(uuid.uuid4())
        
        self.scheduled_invalidations.append({
            "id": schedule_id,
            "when": when,
            "cache_name": cache_name,
            "key": key,
            "tags": tags,
            "rule_name": rule_name,
            "status": "pending",
        })
        
        logger.info(f"Scheduled invalidation {schedule_id} for {when}")
        return schedule_id
    
    def process_scheduled_invalidations(self) -> Dict[str, int]:
        """
        Process due scheduled invalidations.
        
        Returns:
            Results of processed invalidations
        """
        now = datetime.now()
        results = {}
        
        for schedule in self.scheduled_invalidations:
            if schedule["status"] == "pending" and schedule["when"] <= now:
                inv_results = self.invalidate(
                    cache_name=schedule["cache_name"],
                    key=schedule["key"],
                    tags=schedule["tags"],
                    rule_name=schedule["rule_name"],
                )
                
                for name, count in inv_results.items():
                    results[name] = results.get(name, 0) + count
                
                schedule["status"] = "completed"
                schedule["completed_at"] = now
        
        # Clean up old completed invalidations
        cutoff = now - timedelta(days=1)
        self.scheduled_invalidations = [
            s for s in self.scheduled_invalidations
            if s["status"] == "pending" or s.get("completed_at", now) > cutoff
        ]
        
        return results
    
    def get_invalidation_stats(self) -> Dict[str, Any]:
        """Get invalidation statistics."""
        total_invalidations = sum(
            h.get("total_count", 0) for h in self.invalidation_history
        )
        
        pending_scheduled = sum(
            1 for s in self.scheduled_invalidations
            if s["status"] == "pending"
        )
        
        return {
            "total_invalidations": total_invalidations,
            "history_entries": len(self.invalidation_history),
            "active_rules": len(self.rules),
            "registered_caches": len(self.cache_systems),
            "pending_scheduled": pending_scheduled,
            "recent_invalidations": list(self.invalidation_history)[-10:],
        }
    
    def _apply_rule(self, cache: Any, rule: InvalidationRule) -> int:
        """Apply invalidation rule to a cache."""
        count = 0
        
        with cache.lock:
            keys_to_invalidate = []
            
            for key in list(cache.cache.keys()):
                entry_info = cache.get_entry_info(key)
                if entry_info and rule.matches(key, entry_info):
                    keys_to_invalidate.append(key)
            
            for key in keys_to_invalidate:
                cache._remove_entry(key)
                count += 1
                cache.stats.total_invalidations += 1
        
        if count > 0:
            logger.debug(f"Rule '{rule.name}' invalidated {count} entries")
        
        return count
    
    def _cascade_invalidation(self, tags: Set[str]) -> Dict[str, int]:
        """Cascade invalidation to related entries."""
        results = {}
        
        # Find rules with cascade tags
        for rule in self.rules.values():
            if rule.cascade_tags and tags.intersection(rule.cascade_tags):
                for name, cache in self.cache_systems.items():
                    if cache:
                        count = cache.invalidate(tags=rule.cascade_tags)
                        if count > 0:
                            results[name] = results.get(name, 0) + count
        
        return results
    
    def _record_invalidation(
        self,
        results: Dict[str, int],
        key: Optional[str],
        tags: Optional[Set[str]],
        rule_name: Optional[str],
    ) -> None:
        """Record invalidation in history."""
        total_count = sum(results.values())
        
        if total_count > 0:
            self.invalidation_history.append({
                "timestamp": datetime.now().isoformat(),
                "results": results,
                "total_count": total_count,
                "key": key,
                "tags": list(tags) if tags else None,
                "rule_name": rule_name,
            })
            # No need to manually trim - deque with maxlen handles it automatically
    
    def _setup_default_rules(self) -> None:
        """Set up default invalidation rules."""
        # Rule for expired entries
        self.add_rule(InvalidationRule(
            name="expired",
            condition=lambda info: info.get("expired", False),
            strategy=InvalidationStrategy.IMMEDIATE,
        ))
        
        # Rule for old entries
        self.add_rule(InvalidationRule(
            name="old_entries",
            max_age=86400,  # 24 hours
            strategy=InvalidationStrategy.LAZY,
        ))
        
        # Rule for rarely accessed entries
        self.add_rule(InvalidationRule(
            name="rarely_accessed",
            max_accesses=2,
            max_age=3600,  # 1 hour old with less than 2 accesses
            strategy=InvalidationStrategy.IMMEDIATE,
        ))
        
        # Rule for source updates
        self.add_rule(InvalidationRule(
            name="source_update",
            tags={"source", "rulebook"},
            strategy=InvalidationStrategy.CASCADE,
            cascade_tags={"search", "embedding"},
        ))
        
        # Rule for campaign updates
        self.add_rule(InvalidationRule(
            name="campaign_update",
            tags={"campaign"},
            strategy=InvalidationStrategy.CASCADE,
            cascade_tags={"session", "character"},
        ))