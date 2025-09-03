"""
Rate limiter with exponential backoff for MDMAI TTRPG Assistant.

This module provides sophisticated rate limiting capabilities with
exponential backoff, provider-specific limits, and adaptive behavior.
"""

import asyncio
import random
import time
import logging
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

from .base_provider import ProviderType

logger = logging.getLogger(__name__)


class BackoffStrategy(Enum):
    """Backoff strategies for rate limiting."""
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    FIBONACCI = "fibonacci"
    CUSTOM = "custom"


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    requests_per_minute: int = 60
    requests_per_hour: int = 3600
    burst_limit: int = 10  # Max requests in burst window
    burst_window_seconds: int = 10  # Window for burst detection
    backoff_strategy: BackoffStrategy = BackoffStrategy.EXPONENTIAL
    initial_delay: float = 1.0  # Initial delay in seconds
    max_delay: float = 300.0  # Maximum delay in seconds
    backoff_multiplier: float = 2.0  # Multiplier for exponential backoff
    jitter: bool = True  # Add randomization to delays
    adaptive: bool = True  # Adapt limits based on success rates


@dataclass
class RateLimitState:
    """State tracking for rate limiting."""
    requests_this_minute: int = 0
    requests_this_hour: int = 0
    burst_requests: int = 0
    minute_reset_time: datetime = None
    hour_reset_time: datetime = None
    burst_reset_time: datetime = None
    consecutive_failures: int = 0
    current_delay: float = 0.0
    last_request_time: float = 0.0
    
    def __post_init__(self):
        now = datetime.utcnow()
        if self.minute_reset_time is None:
            self.minute_reset_time = now + timedelta(minutes=1)
        if self.hour_reset_time is None:
            self.hour_reset_time = now + timedelta(hours=1)
        if self.burst_reset_time is None:
            self.burst_reset_time = now + timedelta(seconds=10)


class RateLimiter:
    """
    Advanced rate limiter with exponential backoff.
    
    Features:
    - Multiple time windows (minute, hour, burst)
    - Provider-specific rate limits
    - Multiple backoff strategies
    - Adaptive rate limiting based on success rates
    - Jitter to prevent thundering herd
    - Comprehensive metrics and monitoring
    """
    
    def __init__(self):
        """Initialize rate limiter."""
        # Provider-specific configurations
        self.provider_configs = {
            ProviderType.ANTHROPIC: RateLimitConfig(
                requests_per_minute=50,
                requests_per_hour=1000,
                burst_limit=5,
                burst_window_seconds=10,
                initial_delay=0.5,
                max_delay=60.0
            ),
            ProviderType.OPENAI: RateLimitConfig(
                requests_per_minute=500,  # OpenAI has higher limits
                requests_per_hour=10000,
                burst_limit=20,
                burst_window_seconds=5,
                initial_delay=0.1,
                max_delay=30.0
            ),
            ProviderType.GOOGLE: RateLimitConfig(
                requests_per_minute=60,
                requests_per_hour=1000,
                burst_limit=10,
                burst_window_seconds=10,
                initial_delay=1.0,
                max_delay=120.0
            )
        }
        
        # State tracking per provider per user
        self.provider_states: Dict[str, RateLimitState] = {}
        
        # Metrics tracking
        self.metrics = {
            'total_requests': 0,
            'rate_limited_requests': 0,
            'total_delay_time': 0.0,
            'adaptive_adjustments': 0
        }
        
        logger.info("RateLimiter initialized with provider-specific configurations")
    
    async def acquire(
        self, 
        provider: ProviderType, 
        user_id: str,
        priority: int = 5
    ) -> float:
        """
        Acquire permission to make a request with rate limiting.
        
        Args:
            provider: The AI provider
            user_id: User identifier
            priority: Request priority (1-10, higher is more important)
            
        Returns:
            float: Actual delay time in seconds
            
        Raises:
            asyncio.TimeoutError: If maximum delay is reached
        """
        state_key = f"{user_id}:{provider.value}"
        config = self.provider_configs.get(provider, RateLimitConfig())
        
        # Get or create state
        if state_key not in self.provider_states:
            self.provider_states[state_key] = RateLimitState()
        
        state = self.provider_states[state_key]
        
        # Update counters and check limits
        self._update_counters(state)
        delay = self._calculate_delay(config, state, priority)
        
        if delay > 0:
            logger.debug(f"Rate limiting {provider.value} for {user_id}: {delay:.2f}s delay")
            self.metrics['rate_limited_requests'] += 1
            self.metrics['total_delay_time'] += delay
            
            # Apply the delay
            await asyncio.sleep(delay)
        
        # Record the request
        state.last_request_time = time.time()
        self._increment_counters(state)
        self.metrics['total_requests'] += 1
        
        return delay
    
    def record_success(self, provider: ProviderType, user_id: str):
        """
        Record a successful request for adaptive rate limiting.
        
        Args:
            provider: The AI provider
            user_id: User identifier
        """
        state_key = f"{user_id}:{provider.value}"
        if state_key not in self.provider_states:
            return
        
        state = self.provider_states[state_key]
        config = self.provider_configs.get(provider, RateLimitConfig())
        
        # Reset failure tracking
        if state.consecutive_failures > 0:
            logger.debug(f"Resetting failure count for {provider.value} user {user_id}")
            state.consecutive_failures = 0
            state.current_delay = config.initial_delay
        
        # Adaptive adjustment - reduce delays on success
        if config.adaptive and state.current_delay > config.initial_delay:
            state.current_delay = max(
                config.initial_delay,
                state.current_delay * 0.8  # Reduce delay by 20%
            )
            self.metrics['adaptive_adjustments'] += 1
    
    def record_failure(
        self, 
        provider: ProviderType, 
        user_id: str,
        is_rate_limit_error: bool = False
    ):
        """
        Record a failed request for adaptive rate limiting.
        
        Args:
            provider: The AI provider
            user_id: User identifier
            is_rate_limit_error: Whether the failure was due to rate limiting
        """
        state_key = f"{user_id}:{provider.value}"
        if state_key not in self.provider_states:
            self.provider_states[state_key] = RateLimitState()
        
        state = self.provider_states[state_key]
        config = self.provider_configs.get(provider, RateLimitConfig())
        
        state.consecutive_failures += 1
        
        # Increase delay based on backoff strategy
        if config.backoff_strategy == BackoffStrategy.EXPONENTIAL:
            state.current_delay = min(
                config.max_delay,
                config.initial_delay * (config.backoff_multiplier ** state.consecutive_failures)
            )
        elif config.backoff_strategy == BackoffStrategy.LINEAR:
            state.current_delay = min(
                config.max_delay,
                config.initial_delay + (state.consecutive_failures * config.initial_delay)
            )
        elif config.backoff_strategy == BackoffStrategy.FIBONACCI:
            state.current_delay = min(
                config.max_delay,
                config.initial_delay * self._fibonacci(state.consecutive_failures)
            )
        
        # Extra penalty for rate limit errors
        if is_rate_limit_error:
            state.current_delay = min(config.max_delay, state.current_delay * 2.0)
            logger.warning(f"Rate limit error for {provider.value} user {user_id}, increasing delay to {state.current_delay:.2f}s")
        
        self.metrics['adaptive_adjustments'] += 1
    
    def get_status(self, provider: ProviderType, user_id: str) -> Dict:
        """
        Get current rate limiting status.
        
        Args:
            provider: The AI provider
            user_id: User identifier
            
        Returns:
            Dict: Status information
        """
        state_key = f"{user_id}:{provider.value}"
        config = self.provider_configs.get(provider, RateLimitConfig())
        
        if state_key not in self.provider_states:
            return {
                'requests_per_minute_remaining': config.requests_per_minute,
                'requests_per_hour_remaining': config.requests_per_hour,
                'burst_requests_remaining': config.burst_limit,
                'current_delay': 0.0,
                'consecutive_failures': 0,
                'next_reset_minute': None,
                'next_reset_hour': None
            }
        
        state = self.provider_states[state_key]
        self._update_counters(state)
        
        return {
            'requests_per_minute_remaining': max(0, config.requests_per_minute - state.requests_this_minute),
            'requests_per_hour_remaining': max(0, config.requests_per_hour - state.requests_this_hour),
            'burst_requests_remaining': max(0, config.burst_limit - state.burst_requests),
            'current_delay': state.current_delay,
            'consecutive_failures': state.consecutive_failures,
            'next_reset_minute': state.minute_reset_time.isoformat(),
            'next_reset_hour': state.hour_reset_time.isoformat()
        }
    
    def get_metrics(self) -> Dict:
        """
        Get rate limiter metrics.
        
        Returns:
            Dict: Metrics information
        """
        total_requests = self.metrics['total_requests']
        rate_limited = self.metrics['rate_limited_requests']
        
        return {
            'total_requests': total_requests,
            'rate_limited_requests': rate_limited,
            'rate_limit_percentage': (rate_limited / max(total_requests, 1)) * 100,
            'total_delay_time_seconds': self.metrics['total_delay_time'],
            'average_delay_per_limited_request': (
                self.metrics['total_delay_time'] / max(rate_limited, 1)
            ),
            'adaptive_adjustments': self.metrics['adaptive_adjustments'],
            'active_states': len(self.provider_states)
        }
    
    def reset_user_state(self, provider: ProviderType, user_id: str):
        """
        Reset rate limiting state for a user.
        
        Args:
            provider: The AI provider
            user_id: User identifier
        """
        state_key = f"{user_id}:{provider.value}"
        if state_key in self.provider_states:
            del self.provider_states[state_key]
            logger.info(f"Reset rate limit state for {provider.value} user {user_id}")
    
    def update_config(self, provider: ProviderType, config: RateLimitConfig):
        """
        Update rate limiting configuration for a provider.
        
        Args:
            provider: The AI provider
            config: New configuration
        """
        self.provider_configs[provider] = config
        logger.info(f"Updated rate limit config for {provider.value}")
    
    def _update_counters(self, state: RateLimitState):
        """Update time-based counters."""
        now = datetime.utcnow()
        
        # Reset minute counter
        if now >= state.minute_reset_time:
            state.requests_this_minute = 0
            state.minute_reset_time = now + timedelta(minutes=1)
        
        # Reset hour counter
        if now >= state.hour_reset_time:
            state.requests_this_hour = 0
            state.hour_reset_time = now + timedelta(hours=1)
        
        # Reset burst counter
        if now >= state.burst_reset_time:
            state.burst_requests = 0
            state.burst_reset_time = now + timedelta(seconds=10)  # Configurable burst window
    
    def _increment_counters(self, state: RateLimitState):
        """Increment request counters."""
        state.requests_this_minute += 1
        state.requests_this_hour += 1
        state.burst_requests += 1
    
    def _calculate_delay(
        self, 
        config: RateLimitConfig, 
        state: RateLimitState,
        priority: int
    ) -> float:
        """
        Calculate delay needed before making request.
        
        Args:
            config: Rate limit configuration
            state: Current state
            priority: Request priority (1-10)
            
        Returns:
            float: Delay in seconds
        """
        delays = []
        
        # Check minute limit
        if state.requests_this_minute >= config.requests_per_minute:
            time_until_reset = (state.minute_reset_time - datetime.utcnow()).total_seconds()
            delays.append(max(0, time_until_reset))
        
        # Check hour limit
        if state.requests_this_hour >= config.requests_per_hour:
            time_until_reset = (state.hour_reset_time - datetime.utcnow()).total_seconds()
            delays.append(max(0, time_until_reset))
        
        # Check burst limit
        if state.burst_requests >= config.burst_limit:
            time_until_reset = (state.burst_reset_time - datetime.utcnow()).total_seconds()
            delays.append(max(0, time_until_reset))
        
        # Check minimum time between requests
        if state.last_request_time > 0:
            time_since_last = time.time() - state.last_request_time
            min_interval = 60.0 / config.requests_per_minute  # Minimum interval based on per-minute limit
            if time_since_last < min_interval:
                delays.append(min_interval - time_since_last)
        
        # Add backoff delay for failures
        if state.consecutive_failures > 0:
            backoff_delay = state.current_delay
            
            # Apply jitter to prevent thundering herd
            if config.jitter:
                jitter_factor = random.uniform(0.8, 1.2)
                backoff_delay *= jitter_factor
            
            delays.append(backoff_delay)
        
        # Use the maximum delay
        base_delay = max(delays) if delays else 0.0
        
        # Apply priority adjustment (higher priority = less delay)
        priority_factor = max(0.1, (11 - priority) / 10.0)  # Priority 10 = 10% delay, Priority 1 = 100% delay
        adjusted_delay = base_delay * priority_factor
        
        # Cap at maximum delay
        return min(adjusted_delay, config.max_delay)
    
    def _fibonacci(self, n: int) -> int:
        """Calculate nth Fibonacci number for backoff strategy."""
        if n <= 1:
            return n
        elif n <= 10:  # Cache first 10 for performance
            fib_cache = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
            return fib_cache[n]
        else:
            # For larger n, use iterative approach
            a, b = 0, 1
            for _ in range(2, n + 1):
                a, b = b, a + b
            return b
    
    def cleanup_old_states(self, max_age_hours: int = 24):
        """
        Clean up old rate limiting states.
        
        Args:
            max_age_hours: Maximum age of states to keep
        """
        cutoff_time = time.time() - (max_age_hours * 3600)
        old_keys = [
            key for key, state in self.provider_states.items()
            if state.last_request_time < cutoff_time
        ]
        
        for key in old_keys:
            del self.provider_states[key]
        
        if old_keys:
            logger.info(f"Cleaned up {len(old_keys)} old rate limiting states")