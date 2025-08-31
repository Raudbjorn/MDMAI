"""Rate limiting implementation for MCP tools."""

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from threading import Lock
from typing import Any, Dict, Optional

from config.logging_config import get_logger

logger = get_logger(__name__)


class RateLimitStrategy(Enum):
    """Rate limiting strategies."""

    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"
    FIXED_WINDOW = "fixed_window"
    LEAKY_BUCKET = "leaky_bucket"


class OperationType(Enum):
    """Types of operations with different rate limits."""

    # Search operations
    SEARCH_BASIC = "search.basic"
    SEARCH_ADVANCED = "search.advanced"
    SEARCH_ANALYTICS = "search.analytics"

    # Campaign operations
    CAMPAIGN_READ = "campaign.read"
    CAMPAIGN_WRITE = "campaign.write"
    CAMPAIGN_DELETE = "campaign.delete"

    # Source operations
    SOURCE_ADD = "source.add"
    SOURCE_READ = "source.read"
    SOURCE_DELETE = "source.delete"

    # Character operations
    CHARACTER_GENERATE = "character.generate"
    CHARACTER_READ = "character.read"
    CHARACTER_UPDATE = "character.update"

    # Session operations
    SESSION_CREATE = "session.create"
    SESSION_UPDATE = "session.update"

    # System operations
    CACHE_CLEAR = "cache.clear"
    INDEX_UPDATE = "index.update"
    SYSTEM_CONFIG = "system.config"

    # Personality operations
    PERSONALITY_CREATE = "personality.create"
    PERSONALITY_APPLY = "personality.apply"


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""

    max_requests: int
    time_window_seconds: int
    burst_size: Optional[int] = None
    strategy: RateLimitStrategy = RateLimitStrategy.SLIDING_WINDOW

    def __post_init__(self):
        """Set burst size if not provided."""
        if self.burst_size is None:
            self.burst_size = self.max_requests * 2


@dataclass
class RateLimitStatus:
    """Status of rate limiting for a client."""

    allowed: bool
    remaining_requests: int
    reset_time: datetime
    retry_after: Optional[int] = None
    message: Optional[str] = None


@dataclass
class TokenBucket:
    """Token bucket for rate limiting."""

    capacity: int
    tokens: float
    refill_rate: float
    last_refill: float = field(default_factory=time.time)
    lock: Lock = field(default_factory=Lock)

    def consume(self, tokens: int = 1) -> bool:
        """
        Consume tokens from the bucket.

        Args:
            tokens: Number of tokens to consume

        Returns:
            True if tokens were available
        """
        with self.lock:
            self._refill()
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill
        tokens_to_add = elapsed * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill = now


@dataclass
class SlidingWindow:
    """Sliding window for rate limiting."""

    max_requests: int
    window_size: int
    requests: deque = field(default_factory=deque)
    lock: Lock = field(default_factory=Lock)

    def add_request(self) -> bool:
        """
        Add a request to the sliding window.

        Returns:
            True if request is allowed
        """
        with self.lock:
            now = time.time()
            # Remove old requests outside the window
            while self.requests and self.requests[0] <= now - self.window_size:
                self.requests.popleft()

            # Check if we can add new request
            if len(self.requests) < self.max_requests:
                self.requests.append(now)
                return True
            return False

    def get_remaining(self) -> int:
        """Get remaining requests in current window."""
        with self.lock:
            now = time.time()
            # Remove old requests
            while self.requests and self.requests[0] <= now - self.window_size:
                self.requests.popleft()
            return self.max_requests - len(self.requests)

    def get_reset_time(self) -> datetime:
        """Get time when the window resets."""
        with self.lock:
            if self.requests:
                oldest_request = self.requests[0]
                reset_timestamp = oldest_request + self.window_size
                return datetime.fromtimestamp(reset_timestamp)
            return datetime.utcnow()


class RateLimiter:
    """Rate limiter for controlling request rates."""

    # Default rate limits per operation type
    DEFAULT_LIMITS = {
        OperationType.SEARCH_BASIC: RateLimitConfig(100, 60),  # 100 requests per minute
        OperationType.SEARCH_ADVANCED: RateLimitConfig(50, 60),  # 50 requests per minute
        OperationType.SEARCH_ANALYTICS: RateLimitConfig(10, 60),  # 10 requests per minute
        OperationType.CAMPAIGN_READ: RateLimitConfig(200, 60),  # 200 requests per minute
        OperationType.CAMPAIGN_WRITE: RateLimitConfig(30, 60),  # 30 requests per minute
        OperationType.CAMPAIGN_DELETE: RateLimitConfig(5, 60),  # 5 requests per minute
        OperationType.SOURCE_ADD: RateLimitConfig(
            5, 300, strategy=RateLimitStrategy.TOKEN_BUCKET
        ),  # 5 per 5 minutes
        OperationType.SOURCE_READ: RateLimitConfig(100, 60),  # 100 requests per minute
        OperationType.SOURCE_DELETE: RateLimitConfig(10, 60),  # 10 requests per minute
        OperationType.CHARACTER_GENERATE: RateLimitConfig(
            20, 60, strategy=RateLimitStrategy.TOKEN_BUCKET
        ),  # 20 per minute
        OperationType.CHARACTER_READ: RateLimitConfig(100, 60),  # 100 requests per minute
        OperationType.CHARACTER_UPDATE: RateLimitConfig(50, 60),  # 50 requests per minute
        OperationType.SESSION_CREATE: RateLimitConfig(20, 60),  # 20 requests per minute
        OperationType.SESSION_UPDATE: RateLimitConfig(50, 60),  # 50 requests per minute
        OperationType.CACHE_CLEAR: RateLimitConfig(5, 300),  # 5 per 5 minutes
        OperationType.INDEX_UPDATE: RateLimitConfig(2, 300),  # 2 per 5 minutes
        OperationType.SYSTEM_CONFIG: RateLimitConfig(10, 300),  # 10 per 5 minutes
        OperationType.PERSONALITY_CREATE: RateLimitConfig(10, 60),  # 10 per minute
        OperationType.PERSONALITY_APPLY: RateLimitConfig(30, 60),  # 30 per minute
    }

    def __init__(
        self,
        custom_limits: Optional[Dict[OperationType, RateLimitConfig]] = None,
        enable_rate_limiting: bool = True,
        global_limit: Optional[RateLimitConfig] = None,
    ):
        """
        Initialize rate limiter.

        Args:
            custom_limits: Custom rate limit configurations
            enable_rate_limiting: Whether to enable rate limiting
            global_limit: Global rate limit across all operations
        """
        self.enabled = enable_rate_limiting
        self.limits = self.DEFAULT_LIMITS.copy()

        # Apply custom limits
        if custom_limits:
            self.limits.update(custom_limits)

        # Global rate limit
        self.global_limit = global_limit or RateLimitConfig(
            1000, 60, strategy=RateLimitStrategy.SLIDING_WINDOW
        )

        # Storage for rate limit tracking
        self._sliding_windows: Dict[str, SlidingWindow] = {}
        self._token_buckets: Dict[str, TokenBucket] = {}
        self._fixed_windows: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"count": 0, "reset_time": time.time()}
        )

        # Global tracking
        self._global_windows: Dict[str, SlidingWindow] = {}

        # Lock for thread safety
        self._lock = Lock()

        logger.info(
            "Rate limiter initialized",
            enabled=self.enabled,
            operation_types=len(self.limits),
        )

    def check_rate_limit(
        self,
        client_id: str,
        operation: OperationType,
        consume: bool = True,
    ) -> RateLimitStatus:
        """
        Check if a request is within rate limits.

        Args:
            client_id: Unique identifier for the client
            operation: Type of operation
            consume: Whether to consume from the limit

        Returns:
            RateLimitStatus indicating if request is allowed
        """
        if not self.enabled:
            return RateLimitStatus(
                allowed=True,
                remaining_requests=999999,
                reset_time=datetime.utcnow() + timedelta(hours=1),
            )

        # Check global rate limit first
        global_status = self._check_global_limit(client_id, consume)
        if not global_status.allowed:
            return global_status

        # Check operation-specific rate limit
        config = self.limits.get(operation)
        if not config:
            # No specific limit for this operation
            return RateLimitStatus(
                allowed=True,
                remaining_requests=999999,
                reset_time=datetime.utcnow() + timedelta(hours=1),
            )

        # Check based on strategy
        if config.strategy == RateLimitStrategy.SLIDING_WINDOW:
            return self._check_sliding_window(client_id, operation, config, consume)
        elif config.strategy == RateLimitStrategy.TOKEN_BUCKET:
            return self._check_token_bucket(client_id, operation, config, consume)
        elif config.strategy == RateLimitStrategy.FIXED_WINDOW:
            return self._check_fixed_window(client_id, operation, config, consume)
        else:
            # Default to sliding window
            return self._check_sliding_window(client_id, operation, config, consume)

    def _check_global_limit(self, client_id: str, consume: bool) -> RateLimitStatus:
        """Check global rate limit."""
        key = f"global:{client_id}"

        with self._lock:
            if key not in self._global_windows:
                self._global_windows[key] = SlidingWindow(
                    self.global_limit.max_requests,
                    self.global_limit.time_window_seconds,
                )

            window = self._global_windows[key]
            remaining = window.get_remaining()

            if consume:
                allowed = window.add_request()
            else:
                allowed = remaining > 0

            if not allowed:
                reset_time = window.get_reset_time()
                retry_after = int((reset_time - datetime.utcnow()).total_seconds())
                return RateLimitStatus(
                    allowed=False,
                    remaining_requests=0,
                    reset_time=reset_time,
                    retry_after=max(1, retry_after),
                    message="Global rate limit exceeded",
                )

            return RateLimitStatus(
                allowed=True,
                remaining_requests=remaining - 1 if consume else remaining,
                reset_time=window.get_reset_time(),
            )

    def _check_sliding_window(
        self,
        client_id: str,
        operation: OperationType,
        config: RateLimitConfig,
        consume: bool,
    ) -> RateLimitStatus:
        """Check rate limit using sliding window strategy."""
        key = f"{operation.value}:{client_id}"

        with self._lock:
            if key not in self._sliding_windows:
                self._sliding_windows[key] = SlidingWindow(
                    config.max_requests, config.time_window_seconds
                )

            window = self._sliding_windows[key]
            remaining = window.get_remaining()

            if consume:
                allowed = window.add_request()
            else:
                allowed = remaining > 0

            reset_time = window.get_reset_time()

            if not allowed:
                retry_after = int((reset_time - datetime.utcnow()).total_seconds())
                return RateLimitStatus(
                    allowed=False,
                    remaining_requests=0,
                    reset_time=reset_time,
                    retry_after=max(1, retry_after),
                    message=f"Rate limit exceeded for {operation.value}",
                )

            return RateLimitStatus(
                allowed=True,
                remaining_requests=remaining - 1 if consume else remaining,
                reset_time=reset_time,
            )

    def _check_token_bucket(
        self,
        client_id: str,
        operation: OperationType,
        config: RateLimitConfig,
        consume: bool,
    ) -> RateLimitStatus:
        """Check rate limit using token bucket strategy."""
        key = f"{operation.value}:{client_id}"

        with self._lock:
            if key not in self._token_buckets:
                refill_rate = config.max_requests / config.time_window_seconds
                self._token_buckets[key] = TokenBucket(
                    capacity=config.burst_size or config.max_requests,
                    tokens=float(config.burst_size or config.max_requests),
                    refill_rate=refill_rate,
                )

            bucket = self._token_buckets[key]

            if consume:
                allowed = bucket.consume(1)
            else:
                bucket._refill()
                allowed = bucket.tokens >= 1

            # Calculate reset time (when bucket will be full)
            tokens_needed = bucket.capacity - bucket.tokens
            seconds_to_full = tokens_needed / bucket.refill_rate
            reset_time = datetime.utcnow() + timedelta(seconds=seconds_to_full)

            if not allowed:
                # Calculate retry after (when at least 1 token will be available)
                seconds_to_token = 1 / bucket.refill_rate
                retry_after = int(seconds_to_token)
                return RateLimitStatus(
                    allowed=False,
                    remaining_requests=int(bucket.tokens),
                    reset_time=reset_time,
                    retry_after=max(1, retry_after),
                    message=f"Rate limit exceeded for {operation.value}",
                )

            return RateLimitStatus(
                allowed=True,
                remaining_requests=int(bucket.tokens),
                reset_time=reset_time,
            )

    def _check_fixed_window(
        self,
        client_id: str,
        operation: OperationType,
        config: RateLimitConfig,
        consume: bool,
    ) -> RateLimitStatus:
        """Check rate limit using fixed window strategy."""
        key = f"{operation.value}:{client_id}"
        now = time.time()

        with self._lock:
            window = self._fixed_windows[key]

            # Check if window has expired
            if now >= window["reset_time"]:
                window["count"] = 0
                window["reset_time"] = now + config.time_window_seconds

            remaining = config.max_requests - window["count"]
            allowed = remaining > 0

            if consume and allowed:
                window["count"] += 1
                remaining -= 1

            reset_time = datetime.fromtimestamp(window["reset_time"])

            if not allowed:
                retry_after = int(window["reset_time"] - now)
                return RateLimitStatus(
                    allowed=False,
                    remaining_requests=0,
                    reset_time=reset_time,
                    retry_after=max(1, retry_after),
                    message=f"Rate limit exceeded for {operation.value}",
                )

            return RateLimitStatus(
                allowed=True,
                remaining_requests=remaining,
                reset_time=reset_time,
            )

    def reset_client_limits(self, client_id: str) -> None:
        """
        Reset all rate limits for a specific client.

        Args:
            client_id: Client identifier
        """
        with self._lock:
            # Clear sliding windows
            keys_to_remove = [k for k in self._sliding_windows if k.endswith(f":{client_id}")]
            for key in keys_to_remove:
                del self._sliding_windows[key]

            # Clear token buckets
            keys_to_remove = [k for k in self._token_buckets if k.endswith(f":{client_id}")]
            for key in keys_to_remove:
                del self._token_buckets[key]

            # Clear fixed windows
            keys_to_remove = [k for k in self._fixed_windows if k.endswith(f":{client_id}")]
            for key in keys_to_remove:
                del self._fixed_windows[key]

            # Clear global windows
            global_key = f"global:{client_id}"
            if global_key in self._global_windows:
                del self._global_windows[global_key]

        logger.info(f"Reset rate limits for client: {client_id}")

    def get_client_status(
        self, client_id: str
    ) -> Dict[OperationType, RateLimitStatus]:
        """
        Get rate limit status for all operations for a client.

        Args:
            client_id: Client identifier

        Returns:
            Dictionary of operation types to their rate limit status
        """
        status = {}
        for operation in OperationType:
            status[operation] = self.check_rate_limit(client_id, operation, consume=False)
        return status

    def update_limit(
        self, operation: OperationType, config: RateLimitConfig
    ) -> None:
        """
        Update rate limit configuration for an operation.

        Args:
            operation: Operation type
            config: New rate limit configuration
        """
        self.limits[operation] = config
        # Clear existing tracking for this operation to apply new limits
        with self._lock:
            for key in list(self._sliding_windows.keys()):
                if key.startswith(f"{operation.value}:"):
                    del self._sliding_windows[key]
            for key in list(self._token_buckets.keys()):
                if key.startswith(f"{operation.value}:"):
                    del self._token_buckets[key]
            for key in list(self._fixed_windows.keys()):
                if key.startswith(f"{operation.value}:"):
                    del self._fixed_windows[key]

        logger.info(f"Updated rate limit for {operation.value}: {config}")

    def cleanup_old_entries(self, inactive_threshold_minutes: int = 60) -> int:
        """
        Clean up old rate limit entries for inactive clients.

        Args:
            inactive_threshold_minutes: Minutes of inactivity before cleanup

        Returns:
            Number of entries cleaned up
        """
        threshold = time.time() - (inactive_threshold_minutes * 60)
        cleaned = 0

        with self._lock:
            # Clean sliding windows
            for key in list(self._sliding_windows.keys()):
                window = self._sliding_windows[key]
                if window.requests and window.requests[-1] < threshold:
                    del self._sliding_windows[key]
                    cleaned += 1

            # Clean token buckets
            for key in list(self._token_buckets.keys()):
                bucket = self._token_buckets[key]
                if bucket.last_refill < threshold:
                    del self._token_buckets[key]
                    cleaned += 1

            # Clean fixed windows
            for key in list(self._fixed_windows.keys()):
                window = self._fixed_windows[key]
                if window["reset_time"] < threshold:
                    del self._fixed_windows[key]
                    cleaned += 1

        if cleaned > 0:
            logger.info(f"Cleaned up {cleaned} old rate limit entries")

        return cleaned