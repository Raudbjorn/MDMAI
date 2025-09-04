"""
Intelligent Budget Enforcement System with Multi-tier Limits and Emergency Brakes.

This module provides sophisticated budget enforcement mechanisms:
- Multi-tier budget limits (soft, hard, emergency)
- Adaptive limit enforcement based on usage patterns
- Emergency circuit breakers for cost protection
- Graceful degradation strategies
- Smart spending velocity monitoring
"""

import asyncio
import time
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from structlog import get_logger

from ..usage_tracking.storage.models import SpendingLimit, UserProfile, ProviderType

logger = get_logger(__name__)


class BudgetLimitType(Enum):
    """Types of budget limits."""
    SOFT = "soft"          # Warning only
    HARD = "hard"          # Block requests
    EMERGENCY = "emergency"  # Emergency brake
    ADAPTIVE = "adaptive"    # Dynamic based on patterns


class VelocityAlert(Enum):
    """Spending velocity alert levels."""
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    CRITICAL = "critical"


class BudgetAction(Enum):
    """Actions to take when budget limits are exceeded."""
    ALLOW = "allow"
    WARN = "warn" 
    THROTTLE = "throttle"
    DOWNGRADE = "downgrade"
    BLOCK = "block"
    EMERGENCY_STOP = "emergency_stop"


class DegradationStrategy:
    """Graceful degradation strategies for budget enforcement."""
    
    # Model downgrades: expensive -> cheaper alternatives
    MODEL_DOWNGRADES = {
        "gpt-4": "gpt-3.5-turbo",
        "gpt-4-turbo": "gpt-3.5-turbo", 
        "claude-3-opus": "claude-3-sonnet",
        "claude-3-sonnet": "claude-3-haiku",
        "gemini-pro": "gemini-pro",  # No downgrade available
    }
    
    # Provider cost order (most to least expensive typically)
    PROVIDER_COST_ORDER = [
        ProviderType.ANTHROPIC,
        ProviderType.OPENAI,
        ProviderType.GOOGLE,
        ProviderType.OLLAMA,
        ProviderType.LOCAL
    ]
    
    @classmethod
    def get_model_downgrade(cls, model: str) -> Optional[str]:
        """Get cheaper alternative model."""
        return cls.MODEL_DOWNGRADES.get(model)
    
    @classmethod
    def get_cheaper_provider(cls, current_provider: ProviderType) -> Optional[ProviderType]:
        """Get cheaper provider alternative."""
        try:
            current_index = cls.PROVIDER_COST_ORDER.index(current_provider)
            if current_index < len(cls.PROVIDER_COST_ORDER) - 1:
                return cls.PROVIDER_COST_ORDER[current_index + 1]
        except ValueError:
            pass
        return None
    
    @classmethod
    def reduce_max_tokens(cls, current_max_tokens: int, reduction_factor: float = 0.5) -> int:
        """Reduce max_tokens to save cost."""
        return max(50, int(current_max_tokens * reduction_factor))


class BudgetLimit:
    """Enhanced budget limit with multi-tier enforcement."""
    
    def __init__(
        self,
        limit_id: str,
        name: str,
        limit_type: BudgetLimitType,
        amount: Decimal,
        period: str,  # "hourly", "daily", "weekly", "monthly"
        action: BudgetAction = BudgetAction.WARN,
        reset_time: Optional[datetime] = None
    ):
        self.limit_id = limit_id
        self.name = name
        self.limit_type = limit_type
        self.amount = amount
        self.period = period
        self.action = action
        self.reset_time = reset_time or self._calculate_next_reset()
        
        self.current_spent = Decimal("0.00")
        self.request_count = 0
        self.last_reset = datetime.utcnow()
        self.violations = []
        self.enabled = True
        
        # Adaptive parameters
        self.adjustment_factor = Decimal("1.0")
        self.min_adjustment = Decimal("0.5")
        self.max_adjustment = Decimal("2.0")
    
    def _calculate_next_reset(self) -> datetime:
        """Calculate next reset time based on period."""
        now = datetime.utcnow()
        
        if self.period == "hourly":
            return now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        elif self.period == "daily":
            return now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        elif self.period == "weekly":
            days_ahead = 6 - now.weekday()  # Monday = 0
            return now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=days_ahead)
        elif self.period == "monthly":
            if now.month == 12:
                return now.replace(year=now.year + 1, month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
            else:
                return now.replace(month=now.month + 1, day=1, hour=0, minute=0, second=0, microsecond=0)
        else:
            return now + timedelta(days=1)
    
    def reset_if_needed(self) -> bool:
        """Reset limit if reset time has passed."""
        now = datetime.utcnow()
        if now >= self.reset_time:
            self.current_spent = Decimal("0.00")
            self.request_count = 0
            self.last_reset = now
            self.reset_time = self._calculate_next_reset()
            self.violations.clear()
            logger.info(f"Budget limit {self.limit_id} reset for period {self.period}")
            return True
        return False
    
    def add_spending(self, amount: Decimal) -> None:
        """Add spending to current total."""
        self.reset_if_needed()
        self.current_spent += amount
        self.request_count += 1
    
    def get_effective_limit(self) -> Decimal:
        """Get effective limit with adaptive adjustments."""
        return self.amount * self.adjustment_factor
    
    def is_exceeded(self, additional_cost: Decimal = Decimal("0.00")) -> bool:
        """Check if limit is exceeded."""
        self.reset_if_needed()
        effective_limit = self.get_effective_limit()
        return (self.current_spent + additional_cost) > effective_limit
    
    def get_remaining(self) -> Decimal:
        """Get remaining budget."""
        self.reset_if_needed()
        effective_limit = self.get_effective_limit()
        return max(Decimal("0.00"), effective_limit - self.current_spent)
    
    def get_utilization_percentage(self) -> float:
        """Get budget utilization as percentage."""
        self.reset_if_needed()
        effective_limit = self.get_effective_limit()
        if effective_limit == 0:
            return 0.0
        return float((self.current_spent / effective_limit) * 100)
    
    def adjust_adaptive_limit(self, usage_pattern_score: float) -> None:
        """Adjust limit based on usage patterns for adaptive limits."""
        if self.limit_type != BudgetLimitType.ADAPTIVE:
            return
        
        # Adjust based on usage pattern (0.0 = very low usage, 1.0 = very high usage)
        if usage_pattern_score < 0.3:
            # Low usage - can reduce limit to save money
            self.adjustment_factor = max(self.min_adjustment, self.adjustment_factor * Decimal("0.9"))
        elif usage_pattern_score > 0.8:
            # High usage - may need to increase limit
            self.adjustment_factor = min(self.max_adjustment, self.adjustment_factor * Decimal("1.1"))
        
        logger.debug(f"Adjusted adaptive limit {self.limit_id} by factor {self.adjustment_factor}")


class SpendingVelocityMonitor:
    """Monitor spending velocity and predict budget exhaustion."""
    
    def __init__(self, window_minutes: int = 60):
        self.window_minutes = window_minutes
        self.spending_history = []  # (timestamp, amount) pairs
        self.velocity_alerts = {}  # user_id -> alert_level
    
    def add_spending(self, user_id: str, amount: Decimal, timestamp: Optional[datetime] = None) -> None:
        """Add spending event."""
        timestamp = timestamp or datetime.utcnow()
        self.spending_history.append((timestamp, float(amount), user_id))
        
        # Clean old entries
        self._clean_history(timestamp)
    
    def _clean_history(self, current_time: datetime) -> None:
        """Remove entries outside the monitoring window."""
        cutoff = current_time - timedelta(minutes=self.window_minutes)
        self.spending_history = [
            (ts, amount, user_id) for ts, amount, user_id in self.spending_history
            if ts >= cutoff
        ]
    
    def calculate_velocity(self, user_id: str, window_minutes: Optional[int] = None) -> float:
        """Calculate spending velocity (cost per minute) for a user."""
        window_minutes = window_minutes or self.window_minutes
        current_time = datetime.utcnow()
        cutoff = current_time - timedelta(minutes=window_minutes)
        
        user_spending = [
            amount for ts, amount, uid in self.spending_history
            if uid == user_id and ts >= cutoff
        ]
        
        if not user_spending:
            return 0.0
        
        return sum(user_spending) / window_minutes
    
    def predict_time_to_limit(self, user_id: str, remaining_budget: Decimal) -> Optional[int]:
        """Predict minutes until budget limit is reached at current velocity."""
        velocity = self.calculate_velocity(user_id)
        
        if velocity <= 0:
            return None
        
        return int(float(remaining_budget) / velocity)
    
    def get_velocity_alert_level(self, user_id: str, daily_budget: Decimal) -> VelocityAlert:
        """Get velocity alert level based on spending rate."""
        velocity_per_minute = self.calculate_velocity(user_id)
        velocity_per_day = velocity_per_minute * 24 * 60
        
        if daily_budget == 0:
            return VelocityAlert.LOW
        
        velocity_ratio = velocity_per_day / float(daily_budget)
        
        if velocity_ratio < 0.5:
            return VelocityAlert.LOW
        elif velocity_ratio < 0.8:
            return VelocityAlert.MEDIUM
        elif velocity_ratio < 1.2:
            return VelocityAlert.HIGH
        else:
            return VelocityAlert.CRITICAL


class BudgetEnforcer:
    """Intelligent budget enforcement system with multi-tier limits."""
    
    def __init__(self):
        self.budget_limits = {}  # user_id -> list of BudgetLimit
        self.velocity_monitor = SpendingVelocityMonitor()
        self.degradation_strategy = DegradationStrategy()
        self.emergency_brakes = {}  # user_id -> timestamp when activated
        self.usage_patterns = {}  # user_id -> pattern analysis
        
        # Circuit breaker configuration
        self.circuit_breaker_threshold = 10  # violations per hour
        self.circuit_breaker_window = 3600  # 1 hour
        self.circuit_violations = {}  # user_id -> (count, reset_time)
        
        logger.info("Budget Enforcer initialized")
    
    def add_budget_limit(self, user_id: str, budget_limit: BudgetLimit) -> None:
        """Add budget limit for a user."""
        if user_id not in self.budget_limits:
            self.budget_limits[user_id] = []
        
        self.budget_limits[user_id].append(budget_limit)
        logger.info(f"Added {budget_limit.limit_type.value} budget limit for user {user_id}: ${budget_limit.amount}")
    
    def remove_budget_limit(self, user_id: str, limit_id: str) -> bool:
        """Remove budget limit."""
        if user_id not in self.budget_limits:
            return False
        
        original_count = len(self.budget_limits[user_id])
        self.budget_limits[user_id] = [
            limit for limit in self.budget_limits[user_id]
            if limit.limit_id != limit_id
        ]
        
        removed = len(self.budget_limits[user_id]) < original_count
        if removed:
            logger.info(f"Removed budget limit {limit_id} for user {user_id}")
        
        return removed
    
    def check_emergency_brake(self, user_id: str) -> bool:
        """Check if emergency brake is active."""
        if user_id in self.emergency_brakes:
            brake_time = self.emergency_brakes[user_id]
            # Emergency brake lasts 1 hour
            if datetime.utcnow() - brake_time < timedelta(hours=1):
                return True
            else:
                del self.emergency_brakes[user_id]
        return False
    
    def activate_emergency_brake(self, user_id: str, reason: str) -> None:
        """Activate emergency brake for a user."""
        self.emergency_brakes[user_id] = datetime.utcnow()
        logger.critical(f"Emergency brake activated for user {user_id}: {reason}")
    
    def check_circuit_breaker(self, user_id: str) -> bool:
        """Check if circuit breaker is active for user."""
        if user_id not in self.circuit_violations:
            return False
        
        count, reset_time = self.circuit_violations[user_id]
        current_time = time.time()
        
        if current_time >= reset_time:
            # Reset circuit breaker
            del self.circuit_violations[user_id]
            return False
        
        return count >= self.circuit_breaker_threshold
    
    def record_violation(self, user_id: str) -> None:
        """Record a budget violation for circuit breaker tracking."""
        current_time = time.time()
        
        if user_id not in self.circuit_violations:
            self.circuit_violations[user_id] = (1, current_time + self.circuit_breaker_window)
        else:
            count, reset_time = self.circuit_violations[user_id]
            if current_time >= reset_time:
                # Reset window
                self.circuit_violations[user_id] = (1, current_time + self.circuit_breaker_window)
            else:
                # Increment violations
                self.circuit_violations[user_id] = (count + 1, reset_time)
    
    async def check_budget_approval(
        self,
        user_id: str,
        estimated_cost: Decimal,
        request_metadata: Dict[str, Any]
    ) -> Tuple[BudgetAction, Optional[Dict[str, Any]], List[str]]:
        """Check budget approval for a request.
        
        Returns:
            Tuple of (action, modification_suggestions, reasons)
        """
        # Check emergency brake first
        if self.check_emergency_brake(user_id):
            return BudgetAction.EMERGENCY_STOP, None, ["Emergency brake is active"]
        
        # Check circuit breaker
        if self.check_circuit_breaker(user_id):
            return BudgetAction.BLOCK, None, ["Circuit breaker is active due to repeated violations"]
        
        if user_id not in self.budget_limits:
            # No limits configured - allow but monitor
            self.velocity_monitor.add_spending(user_id, estimated_cost)
            return BudgetAction.ALLOW, None, []
        
        violations = []
        most_restrictive_action = BudgetAction.ALLOW
        modification_suggestions = {}
        
        # Check all budget limits
        for budget_limit in self.budget_limits[user_id]:
            if not budget_limit.enabled:
                continue
            
            budget_limit.reset_if_needed()
            
            if budget_limit.is_exceeded(estimated_cost):
                violation_msg = f"{budget_limit.name} limit exceeded: ${budget_limit.current_spent + estimated_cost} > ${budget_limit.get_effective_limit()}"
                violations.append(violation_msg)
                
                # Record violation for circuit breaker
                self.record_violation(user_id)
                
                # Determine action based on limit type
                if budget_limit.action.value == BudgetAction.EMERGENCY_STOP.value:
                    self.activate_emergency_brake(user_id, violation_msg)
                    return BudgetAction.EMERGENCY_STOP, None, violations
                
                # Track most restrictive action
                action_priority = {
                    BudgetAction.ALLOW: 0,
                    BudgetAction.WARN: 1,
                    BudgetAction.THROTTLE: 2,
                    BudgetAction.DOWNGRADE: 3,
                    BudgetAction.BLOCK: 4,
                    BudgetAction.EMERGENCY_STOP: 5
                }
                
                if action_priority[budget_limit.action] > action_priority[most_restrictive_action]:
                    most_restrictive_action = budget_limit.action
        
        # Generate modification suggestions for downgrades
        if most_restrictive_action == BudgetAction.DOWNGRADE:
            current_model = request_metadata.get('model', '')
            current_provider = request_metadata.get('provider')
            current_max_tokens = request_metadata.get('max_tokens', 1024)
            
            # Suggest model downgrade
            downgraded_model = self.degradation_strategy.get_model_downgrade(current_model)
            if downgraded_model:
                modification_suggestions['model'] = downgraded_model
            
            # Suggest provider downgrade
            if current_provider:
                cheaper_provider = self.degradation_strategy.get_cheaper_provider(current_provider)
                if cheaper_provider:
                    modification_suggestions['provider'] = cheaper_provider
            
            # Suggest token reduction
            reduced_tokens = self.degradation_strategy.reduce_max_tokens(current_max_tokens)
            modification_suggestions['max_tokens'] = reduced_tokens
        
        # Check spending velocity
        velocity_alert = self.velocity_monitor.get_velocity_alert_level(
            user_id, 
            sum(limit.amount for limit in self.budget_limits[user_id] if limit.period == "daily")
        )
        
        if velocity_alert == VelocityAlert.CRITICAL and most_restrictive_action == BudgetAction.ALLOW:
            most_restrictive_action = BudgetAction.WARN
            violations.append("Critical spending velocity detected")
        
        # Record spending if approved
        if most_restrictive_action in [BudgetAction.ALLOW, BudgetAction.WARN, BudgetAction.DOWNGRADE]:
            self.velocity_monitor.add_spending(user_id, estimated_cost)
            
            # Update budget limits
            for budget_limit in self.budget_limits[user_id]:
                if budget_limit.enabled:
                    budget_limit.add_spending(estimated_cost)
        
        return most_restrictive_action, modification_suggestions or None, violations
    
    def get_budget_status(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive budget status for a user."""
        if user_id not in self.budget_limits:
            return {
                'limits': [],
                'emergency_brake_active': False,
                'circuit_breaker_active': False,
                'spending_velocity': 0.0,
                'velocity_alert': VelocityAlert.LOW.value
            }
        
        status = {
            'limits': [],
            'emergency_brake_active': self.check_emergency_brake(user_id),
            'circuit_breaker_active': self.check_circuit_breaker(user_id),
            'spending_velocity': self.velocity_monitor.calculate_velocity(user_id),
            'velocity_alert': self.velocity_monitor.get_velocity_alert_level(
                user_id,
                sum(limit.amount for limit in self.budget_limits[user_id] if limit.period == "daily")
            ).value
        }
        
        for budget_limit in self.budget_limits[user_id]:
            budget_limit.reset_if_needed()
            status['limits'].append({
                'id': budget_limit.limit_id,
                'name': budget_limit.name,
                'type': budget_limit.limit_type.value,
                'period': budget_limit.period,
                'amount': float(budget_limit.amount),
                'spent': float(budget_limit.current_spent),
                'remaining': float(budget_limit.get_remaining()),
                'utilization_percentage': budget_limit.get_utilization_percentage(),
                'action': budget_limit.action.value,
                'enabled': budget_limit.enabled,
                'reset_time': budget_limit.reset_time.isoformat()
            })
        
        return status
    
    def optimize_adaptive_limits(self, user_id: str, usage_pattern_score: float) -> None:
        """Optimize adaptive limits based on usage patterns."""
        if user_id not in self.budget_limits:
            return
        
        for budget_limit in self.budget_limits[user_id]:
            budget_limit.adjust_adaptive_limit(usage_pattern_score)
    
    def get_spending_forecast(self, user_id: str, days_ahead: int = 30) -> Dict[str, Any]:
        """Generate spending forecast based on current velocity."""
        current_velocity = self.velocity_monitor.calculate_velocity(user_id)  # per minute
        daily_velocity = current_velocity * 24 * 60  # per day
        
        forecast = {
            'current_velocity_per_day': daily_velocity,
            'projected_spending': {},
            'budget_exhaustion_warnings': []
        }
        
        # Project spending for different time periods
        for days in [1, 7, 30, days_ahead]:
            forecast['projected_spending'][f'{days}_days'] = daily_velocity * days
        
        # Check budget exhaustion warnings
        if user_id in self.budget_limits:
            for budget_limit in self.budget_limits[user_id]:
                remaining = budget_limit.get_remaining()
                if remaining > 0:
                    time_to_exhaustion = self.velocity_monitor.predict_time_to_limit(user_id, remaining)
                    if time_to_exhaustion and time_to_exhaustion < 1440:  # Less than 24 hours
                        forecast['budget_exhaustion_warnings'].append({
                            'limit_name': budget_limit.name,
                            'time_to_exhaustion_minutes': time_to_exhaustion,
                            'time_to_exhaustion_hours': time_to_exhaustion / 60
                        })
        
        return forecast
    
    def create_standard_budget_limits(
        self,
        user_id: str,
        daily_soft: Optional[Decimal] = None,
        daily_hard: Optional[Decimal] = None,
        monthly_soft: Optional[Decimal] = None,
        monthly_hard: Optional[Decimal] = None,
        emergency_daily: Optional[Decimal] = None
    ) -> None:
        """Create standard set of budget limits for a user."""
        limits = []
        
        if daily_soft:
            limits.append(BudgetLimit(
                f"{user_id}_daily_soft",
                "Daily Soft Limit",
                BudgetLimitType.SOFT,
                daily_soft,
                "daily",
                BudgetAction.WARN
            ))
        
        if daily_hard:
            limits.append(BudgetLimit(
                f"{user_id}_daily_hard",
                "Daily Hard Limit",
                BudgetLimitType.HARD,
                daily_hard,
                "daily",
                BudgetAction.BLOCK
            ))
        
        if monthly_soft:
            limits.append(BudgetLimit(
                f"{user_id}_monthly_soft", 
                "Monthly Soft Limit",
                BudgetLimitType.SOFT,
                monthly_soft,
                "monthly",
                BudgetAction.WARN
            ))
        
        if monthly_hard:
            limits.append(BudgetLimit(
                f"{user_id}_monthly_hard",
                "Monthly Hard Limit", 
                BudgetLimitType.HARD,
                monthly_hard,
                "monthly",
                BudgetAction.DOWNGRADE
            ))
        
        if emergency_daily:
            limits.append(BudgetLimit(
                f"{user_id}_emergency",
                "Emergency Brake",
                BudgetLimitType.EMERGENCY,
                emergency_daily,
                "daily",
                BudgetAction.EMERGENCY_STOP
            ))
        
        for limit in limits:
            self.add_budget_limit(user_id, limit)
        
        logger.info(f"Created {len(limits)} standard budget limits for user {user_id}")