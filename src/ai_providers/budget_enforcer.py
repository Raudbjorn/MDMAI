"""Budget enforcement system with graceful degradation and spending limits."""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from structlog import get_logger

from .models import ProviderType, AIRequest, CostTier
from .user_usage_tracker import UserUsageTracker, UserSpendingLimits
from .pricing_engine import PricingEngine, DynamicPricingFactor

logger = get_logger(__name__)


class EnforcementAction(Enum):
    """Actions that can be taken when budget limits are approached or exceeded."""
    
    ALLOW = "allow"  # Allow request to proceed
    WARN = "warn"  # Allow but issue warning
    THROTTLE = "throttle"  # Slow down requests
    DOWNGRADE = "downgrade"  # Switch to cheaper model
    CACHE_ONLY = "cache_only"  # Only serve cached responses
    QUEUE = "queue"  # Queue request for later processing
    DENY = "deny"  # Deny the request
    EMERGENCY_STOP = "emergency_stop"  # Stop all requests for user


@dataclass
class BudgetPolicy:
    """Budget enforcement policy configuration."""
    
    policy_id: str
    name: str
    description: str
    user_tier: str  # free, premium, enterprise, etc.
    
    # Enforcement thresholds (percentage of limit)
    warning_threshold: float = 0.8  # 80% of limit
    throttle_threshold: float = 0.9  # 90% of limit  
    downgrade_threshold: float = 0.95  # 95% of limit
    hard_limit_threshold: float = 1.0  # 100% of limit
    
    # Grace periods and cooldowns
    grace_period_hours: int = 24  # Hours to allow temporary overages
    cooldown_minutes: int = 60  # Minutes to wait before re-evaluation
    
    # Degradation settings
    allow_model_downgrade: bool = True
    allow_provider_switching: bool = True
    allow_caching_fallback: bool = True
    allow_queuing: bool = False
    
    # Emergency settings
    emergency_threshold: float = 1.5  # 150% - emergency stop
    max_overage_amount: float = 10.0  # USD - maximum overage allowed
    
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    enabled: bool = True


@dataclass
class EnforcementResult:
    """Result of budget enforcement check."""
    
    action: EnforcementAction
    allowed: bool
    reason: str
    
    # Alternative options if request was modified
    modified_request: Optional[AIRequest] = None
    suggested_provider: Optional[ProviderType] = None
    suggested_model: Optional[str] = None
    estimated_cost: Optional[float] = None
    
    # Warning and limit information
    warning_message: Optional[str] = None
    usage_percentage: float = 0.0
    remaining_budget: float = 0.0
    
    # Metadata for logging and debugging
    metadata: Dict[str, Any] = field(default_factory=dict)


class BudgetEnforcer:
    """Budget enforcement system with graceful degradation capabilities."""
    
    def __init__(
        self,
        usage_tracker: UserUsageTracker,
        pricing_engine: PricingEngine,
        policies_config_path: Optional[str] = None
    ):
        self.usage_tracker = usage_tracker
        self.pricing_engine = pricing_engine
        
        # Budget policies by user tier
        self.policies: Dict[str, BudgetPolicy] = {}
        self._load_default_policies()
        
        # Enforcement state tracking
        self.user_enforcement_state: Dict[str, Dict[str, Any]] = {}
        self.throttle_queues: Dict[str, List[Tuple[AIRequest, float]]] = {}  # user_id -> queued requests
        self.cooldown_timers: Dict[str, datetime] = {}  # user_id -> cooldown end time
        
        # Model alternatives for downgrading
        self.model_alternatives = self._initialize_model_alternatives()
        
        # Statistics
        self.enforcement_stats = {
            "total_checks": 0,
            "actions_taken": {action.value: 0 for action in EnforcementAction},
            "cost_savings": 0.0,
            "users_affected": set(),
        }
        
        logger.info("Budget enforcer initialized")
    
    def _load_default_policies(self) -> None:
        """Load default budget policies for different user tiers."""
        default_policies = [
            BudgetPolicy(
                policy_id="free_tier",
                name="Free Tier Policy",
                description="Conservative limits for free tier users",
                user_tier="free",
                warning_threshold=0.7,
                throttle_threshold=0.8,
                downgrade_threshold=0.9,
                hard_limit_threshold=1.0,
                grace_period_hours=6,
                cooldown_minutes=30,
                allow_queuing=True,
                emergency_threshold=1.2,
                max_overage_amount=2.0
            ),
            BudgetPolicy(
                policy_id="premium_tier",
                name="Premium Tier Policy", 
                description="Balanced limits for premium users",
                user_tier="premium",
                warning_threshold=0.8,
                throttle_threshold=0.9,
                downgrade_threshold=0.95,
                hard_limit_threshold=1.0,
                grace_period_hours=12,
                cooldown_minutes=15,
                allow_queuing=False,
                emergency_threshold=1.3,
                max_overage_amount=25.0
            ),
            BudgetPolicy(
                policy_id="enterprise_tier",
                name="Enterprise Tier Policy",
                description="Flexible limits for enterprise users",
                user_tier="enterprise",
                warning_threshold=0.9,
                throttle_threshold=0.95,
                downgrade_threshold=0.98,
                hard_limit_threshold=1.05,  # Allow 5% overage
                grace_period_hours=24,
                cooldown_minutes=5,
                allow_queuing=False,
                emergency_threshold=1.5,
                max_overage_amount=100.0
            ),
        ]
        
        for policy in default_policies:
            self.policies[policy.user_tier] = policy
    
    def _initialize_model_alternatives(self) -> Dict[str, List[Tuple[str, ProviderType, float]]]:
        """Initialize model alternatives for cost optimization."""
        # This maps expensive models to cheaper alternatives
        # Format: original_model -> [(alternative_model, provider, cost_multiplier), ...]
        return {
            # OpenAI models
            "gpt-4": [
                ("gpt-4-turbo", ProviderType.OPENAI, 0.33),  # Cheaper GPT-4 variant
                ("gpt-3.5-turbo", ProviderType.OPENAI, 0.05),  # Much cheaper
                ("claude-3-haiku", ProviderType.ANTHROPIC, 0.02),  # Cross-provider alternative
            ],
            "gpt-4-turbo": [
                ("gpt-3.5-turbo", ProviderType.OPENAI, 0.15),
                ("claude-3-haiku", ProviderType.ANTHROPIC, 0.08),
                ("gemini-pro", ProviderType.GOOGLE, 0.05),
            ],
            
            # Anthropic models
            "claude-3-opus": [
                ("claude-3-sonnet", ProviderType.ANTHROPIC, 0.2),
                ("claude-3-haiku", ProviderType.ANTHROPIC, 0.02),
                ("gpt-3.5-turbo", ProviderType.OPENAI, 0.1),
            ],
            "claude-3-sonnet": [
                ("claude-3-haiku", ProviderType.ANTHROPIC, 0.1),
                ("gpt-3.5-turbo", ProviderType.OPENAI, 0.5),
                ("gemini-pro", ProviderType.GOOGLE, 0.17),
            ],
            
            # Google models - typically already cost-effective
            "gemini-pro": [
                ("claude-3-haiku", ProviderType.ANTHROPIC, 0.5),
                ("gpt-3.5-turbo", ProviderType.OPENAI, 3.0),  # Actually more expensive
            ],
        }
    
    async def check_budget_enforcement(
        self,
        user_id: str,
        request: AIRequest,
        estimated_cost: float,
        provider_type: ProviderType
    ) -> EnforcementResult:
        """
        Check if a request should be allowed, modified, or denied based on budget limits.
        
        Returns:
            EnforcementResult with action to take and any modifications
        """
        self.enforcement_stats["total_checks"] += 1
        
        # Get user profile and limits
        user_profile = self.usage_tracker.user_profiles.get(user_id)
        if not user_profile:
            # Unknown user - create default profile and allow with warning
            return EnforcementResult(
                action=EnforcementAction.WARN,
                allowed=True,
                reason="Unknown user - creating default profile",
                warning_message="User profile not found. Please register for better cost management."
            )
        
        user_limits = self.usage_tracker.user_limits.get(user_id)
        if not user_limits or not user_limits.enabled:
            # No limits configured - allow
            return EnforcementResult(
                action=EnforcementAction.ALLOW,
                allowed=True,
                reason="No budget limits configured"
            )
        
        # Get policy for user tier
        policy = self.policies.get(user_profile.user_tier)
        if not policy or not policy.enabled:
            # No policy - allow with warning
            return EnforcementResult(
                action=EnforcementAction.WARN,
                allowed=True,
                reason="No budget policy found for user tier",
                warning_message=f"No budget policy configured for tier: {user_profile.user_tier}"
            )
        
        # Check if user is in cooldown period
        if user_id in self.cooldown_timers:
            if datetime.now() < self.cooldown_timers[user_id]:
                return EnforcementResult(
                    action=EnforcementAction.QUEUE if policy.allow_queuing else EnforcementAction.DENY,
                    allowed=policy.allow_queuing,
                    reason=f"User in cooldown period until {self.cooldown_timers[user_id]}"
                )
            else:
                # Cooldown expired
                del self.cooldown_timers[user_id]
        
        # Check per-request limits first
        if user_limits.per_request_limit and estimated_cost > user_limits.per_request_limit:
            # Try to find a cheaper alternative
            alternative = await self._find_cheaper_alternative(request, provider_type, user_limits.per_request_limit)
            if alternative:
                return alternative
            
            return EnforcementResult(
                action=EnforcementAction.DENY,
                allowed=False,
                reason=f"Request cost ${estimated_cost:.4f} exceeds per-request limit ${user_limits.per_request_limit:.4f}",
                metadata={"estimated_cost": estimated_cost, "limit": user_limits.per_request_limit}
            )
        
        # Check session limits
        if user_limits.per_session_limit and request.session_id:
            session_usage = self.usage_tracker.get_user_session_usage(user_id, request.session_id)
            if session_usage + estimated_cost > user_limits.per_session_limit:
                return EnforcementResult(
                    action=EnforcementAction.DENY,
                    allowed=False,
                    reason=f"Session cost would exceed limit: ${session_usage + estimated_cost:.4f} > ${user_limits.per_session_limit:.4f}"
                )
        
        # Check daily limits
        daily_result = await self._check_daily_limits(user_id, estimated_cost, user_limits, policy)
        if daily_result.action in [EnforcementAction.DENY, EnforcementAction.EMERGENCY_STOP]:
            return daily_result
        
        # Check monthly limits
        monthly_result = await self._check_monthly_limits(user_id, estimated_cost, user_limits, policy)
        if monthly_result.action in [EnforcementAction.DENY, EnforcementAction.EMERGENCY_STOP]:
            return monthly_result
        
        # Determine the most restrictive action
        actions_severity = [
            EnforcementAction.ALLOW,
            EnforcementAction.WARN, 
            EnforcementAction.THROTTLE,
            EnforcementAction.DOWNGRADE,
            EnforcementAction.CACHE_ONLY,
            EnforcementAction.QUEUE,
            EnforcementAction.DENY,
            EnforcementAction.EMERGENCY_STOP
        ]
        
        most_severe_action = max([daily_result.action, monthly_result.action], key=lambda x: actions_severity.index(x))
        
        # Execute the action
        final_result = daily_result if daily_result.action == most_severe_action else monthly_result
        
        # Apply additional logic based on action
        if final_result.action == EnforcementAction.DOWNGRADE:
            # Try to find a cheaper model
            alternative = await self._find_cheaper_alternative(request, provider_type, estimated_cost * 0.7)  # 30% cheaper
            if alternative:
                return alternative
            else:
                # Fallback to denial if no alternative
                final_result.action = EnforcementAction.DENY
                final_result.allowed = False
                final_result.reason += " (no cheaper alternative available)"
        
        elif final_result.action == EnforcementAction.THROTTLE:
            # Implement throttling delay
            await self._apply_throttling(user_id, policy)
        
        elif final_result.action == EnforcementAction.QUEUE:
            # Add to queue for later processing
            await self._queue_request(user_id, request, estimated_cost)
            final_result.allowed = False
            final_result.reason = "Request queued for later processing due to budget constraints"
        
        # Update enforcement statistics
        self.enforcement_stats["actions_taken"][final_result.action.value] += 1
        if final_result.action != EnforcementAction.ALLOW:
            self.enforcement_stats["users_affected"].add(user_id)
        
        if final_result.estimated_cost and final_result.estimated_cost < estimated_cost:
            self.enforcement_stats["cost_savings"] += (estimated_cost - final_result.estimated_cost)
        
        return final_result
    
    async def _check_daily_limits(
        self,
        user_id: str,
        estimated_cost: float,
        limits: UserSpendingLimits,
        policy: BudgetPolicy
    ) -> EnforcementResult:
        """Check daily budget limits and determine action."""
        if not limits.daily_limit:
            return EnforcementResult(action=EnforcementAction.ALLOW, allowed=True, reason="No daily limit set")
        
        current_usage = self.usage_tracker.get_user_daily_usage(user_id)
        projected_usage = current_usage + estimated_cost
        usage_percentage = projected_usage / limits.daily_limit
        remaining_budget = limits.daily_limit - current_usage
        
        result = EnforcementResult(
            action=EnforcementAction.ALLOW,
            allowed=True,
            reason="Within daily budget",
            usage_percentage=usage_percentage,
            remaining_budget=remaining_budget
        )
        
        # Check emergency threshold
        if usage_percentage >= policy.emergency_threshold:
            result.action = EnforcementAction.EMERGENCY_STOP
            result.allowed = False
            result.reason = f"Emergency stop: daily usage would be {usage_percentage:.1%} of limit"
            await self._set_cooldown(user_id, policy.cooldown_minutes)
            return result
        
        # Check hard limit
        if usage_percentage >= policy.hard_limit_threshold:
            # Check if within grace period and overage limit
            if await self._check_grace_period(user_id, projected_usage - limits.daily_limit, policy):
                result.action = EnforcementAction.WARN
                result.warning_message = f"Exceeding daily budget (${projected_usage:.2f}/${limits.daily_limit:.2f}) - grace period active"
            else:
                result.action = EnforcementAction.DENY
                result.allowed = False
                result.reason = f"Daily budget exceeded: ${projected_usage:.2f} > ${limits.daily_limit:.2f}"
                return result
        
        # Check downgrade threshold
        elif usage_percentage >= policy.downgrade_threshold:
            result.action = EnforcementAction.DOWNGRADE
            result.allowed = True
            result.reason = f"Approaching daily limit ({usage_percentage:.1%}) - suggesting cheaper alternatives"
        
        # Check throttle threshold
        elif usage_percentage >= policy.throttle_threshold:
            result.action = EnforcementAction.THROTTLE
            result.allowed = True
            result.reason = f"High daily usage ({usage_percentage:.1%}) - applying throttling"
        
        # Check warning threshold
        elif usage_percentage >= policy.warning_threshold:
            result.action = EnforcementAction.WARN
            result.allowed = True
            result.reason = f"Daily budget warning: {usage_percentage:.1%} of limit used"
            result.warning_message = f"You've used ${current_usage:.2f} of your ${limits.daily_limit:.2f} daily budget"
        
        return result
    
    async def _check_monthly_limits(
        self,
        user_id: str,
        estimated_cost: float,
        limits: UserSpendingLimits,
        policy: BudgetPolicy
    ) -> EnforcementResult:
        """Check monthly budget limits and determine action."""
        if not limits.monthly_limit:
            return EnforcementResult(action=EnforcementAction.ALLOW, allowed=True, reason="No monthly limit set")
        
        current_usage = self.usage_tracker.get_user_monthly_usage(user_id)
        projected_usage = current_usage + estimated_cost
        usage_percentage = projected_usage / limits.monthly_limit
        remaining_budget = limits.monthly_limit - current_usage
        
        result = EnforcementResult(
            action=EnforcementAction.ALLOW,
            allowed=True,
            reason="Within monthly budget",
            usage_percentage=usage_percentage,
            remaining_budget=remaining_budget
        )
        
        # Similar logic to daily limits but for monthly
        if usage_percentage >= policy.emergency_threshold:
            result.action = EnforcementAction.EMERGENCY_STOP
            result.allowed = False
            result.reason = f"Emergency stop: monthly usage would be {usage_percentage:.1%} of limit"
            await self._set_cooldown(user_id, policy.cooldown_minutes * 2)  # Longer cooldown for monthly
            return result
        
        if usage_percentage >= policy.hard_limit_threshold:
            if await self._check_grace_period(user_id, projected_usage - limits.monthly_limit, policy):
                result.action = EnforcementAction.WARN
                result.warning_message = f"Exceeding monthly budget (${projected_usage:.2f}/${limits.monthly_limit:.2f}) - grace period active"
            else:
                result.action = EnforcementAction.DENY
                result.allowed = False
                result.reason = f"Monthly budget exceeded: ${projected_usage:.2f} > ${limits.monthly_limit:.2f}"
                return result
        
        elif usage_percentage >= policy.downgrade_threshold:
            result.action = EnforcementAction.DOWNGRADE
            result.reason = f"Approaching monthly limit ({usage_percentage:.1%}) - suggesting cheaper alternatives"
        
        elif usage_percentage >= policy.throttle_threshold:
            result.action = EnforcementAction.THROTTLE
            result.reason = f"High monthly usage ({usage_percentage:.1%}) - applying throttling"
        
        elif usage_percentage >= policy.warning_threshold:
            result.action = EnforcementAction.WARN
            result.warning_message = f"You've used ${current_usage:.2f} of your ${limits.monthly_limit:.2f} monthly budget"
        
        return result
    
    async def _find_cheaper_alternative(
        self,
        request: AIRequest,
        current_provider: ProviderType,
        target_cost: float
    ) -> Optional[EnforcementResult]:
        """Find a cheaper alternative model for the request."""
        current_model = request.model
        
        # Check if we have alternatives for this model
        alternatives = self.model_alternatives.get(current_model, [])
        
        for alt_model, alt_provider, cost_multiplier in alternatives:
            # Estimate cost with alternative
            alt_cost, _ = self.pricing_engine.estimate_request_cost(
                alt_provider, alt_model, request.messages, request.tools, request.max_tokens
            )
            
            if alt_cost <= target_cost:
                # Create modified request
                modified_request = AIRequest(
                    request_id=request.request_id,
                    session_id=request.session_id,
                    model=alt_model,
                    messages=request.messages,
                    tools=request.tools,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    stream=request.stream,
                    metadata=request.metadata,
                    budget_limit=request.budget_limit
                )
                
                return EnforcementResult(
                    action=EnforcementAction.DOWNGRADE,
                    allowed=True,
                    reason=f"Downgraded from {current_model} to {alt_model} to meet budget",
                    modified_request=modified_request,
                    suggested_provider=alt_provider,
                    suggested_model=alt_model,
                    estimated_cost=alt_cost,
                    warning_message=f"Request modified to use cheaper model: {alt_model} (${alt_cost:.4f} vs ${target_cost:.4f})"
                )
        
        return None
    
    async def _apply_throttling(self, user_id: str, policy: BudgetPolicy) -> None:
        """Apply throttling delay for user."""
        # Simple throttling - increase delay with repeated throttling
        if user_id not in self.user_enforcement_state:
            self.user_enforcement_state[user_id] = {"throttle_count": 0}
        
        self.user_enforcement_state[user_id]["throttle_count"] += 1
        throttle_count = self.user_enforcement_state[user_id]["throttle_count"]
        
        # Exponential backoff: 1s, 2s, 4s, 8s, max 30s
        delay = min(2 ** (throttle_count - 1), 30)
        
        logger.info("Applying throttling", user_id=user_id, delay_seconds=delay)
        await asyncio.sleep(delay)
    
    async def _queue_request(self, user_id: str, request: AIRequest, estimated_cost: float) -> None:
        """Queue a request for later processing."""
        if user_id not in self.throttle_queues:
            self.throttle_queues[user_id] = []
        
        self.throttle_queues[user_id].append((request, estimated_cost))
        
        # Limit queue size to prevent memory issues
        if len(self.throttle_queues[user_id]) > 50:
            self.throttle_queues[user_id] = self.throttle_queues[user_id][-25:]  # Keep last 25
        
        logger.info("Request queued", user_id=user_id, queue_size=len(self.throttle_queues[user_id]))
    
    async def _check_grace_period(self, user_id: str, overage_amount: float, policy: BudgetPolicy) -> bool:
        """Check if user is within grace period for budget overage."""
        if overage_amount <= 0 or overage_amount > policy.max_overage_amount:
            return False
        
        # Initialize enforcement state if needed
        if user_id not in self.user_enforcement_state:
            self.user_enforcement_state[user_id] = {}
        
        state = self.user_enforcement_state[user_id]
        
        # Check if this is first overage or if grace period has expired
        if "grace_period_start" not in state:
            # Start grace period
            state["grace_period_start"] = datetime.now()
            state["grace_overage_total"] = overage_amount
            return True
        
        # Check if grace period has expired
        grace_end = state["grace_period_start"] + timedelta(hours=policy.grace_period_hours)
        if datetime.now() > grace_end:
            # Grace period expired - reset
            state["grace_period_start"] = datetime.now()
            state["grace_overage_total"] = overage_amount
            return overage_amount <= policy.max_overage_amount
        
        # Check total overage during grace period
        state["grace_overage_total"] += overage_amount
        return state["grace_overage_total"] <= policy.max_overage_amount
    
    async def _set_cooldown(self, user_id: str, cooldown_minutes: int) -> None:
        """Set cooldown period for user."""
        cooldown_end = datetime.now() + timedelta(minutes=cooldown_minutes)
        self.cooldown_timers[user_id] = cooldown_end
        
        logger.warning("User placed in cooldown", user_id=user_id, cooldown_end=cooldown_end)
    
    async def process_queued_requests(self, user_id: str, max_requests: int = 5) -> List[Tuple[AIRequest, float]]:
        """Process queued requests for a user when budget allows."""
        if user_id not in self.throttle_queues or not self.throttle_queues[user_id]:
            return []
        
        # Check current budget status
        user_limits = self.usage_tracker.user_limits.get(user_id)
        if not user_limits:
            return []
        
        current_daily = self.usage_tracker.get_user_daily_usage(user_id)
        current_monthly = self.usage_tracker.get_user_monthly_usage(user_id)
        
        processable_requests = []
        remaining_requests = []
        
        for request, estimated_cost in self.throttle_queues[user_id][:max_requests]:
            can_process = True
            
            # Check if adding this request would exceed limits
            if user_limits.daily_limit and (current_daily + estimated_cost) > user_limits.daily_limit:
                can_process = False
            
            if user_limits.monthly_limit and (current_monthly + estimated_cost) > user_limits.monthly_limit:
                can_process = False
            
            if can_process:
                processable_requests.append((request, estimated_cost))
                # Update local tracking AND immediately reserve in usage tracker to prevent race conditions
                current_daily += estimated_cost
                current_monthly += estimated_cost
                # Reserve the cost immediately to prevent other threads from overcommitting
                await self.usage_tracker.reserve_budget(user_id, estimated_cost)
            else:
                remaining_requests.append((request, estimated_cost))
        
        # Keep unprocessed requests in queue
        self.throttle_queues[user_id] = remaining_requests + self.throttle_queues[user_id][max_requests:]
        
        if processable_requests:
            logger.info("Processing queued requests", 
                       user_id=user_id, 
                       processed=len(processable_requests),
                       remaining=len(self.throttle_queues[user_id]))
        
        return processable_requests
    
    def get_enforcement_stats(self) -> Dict[str, Any]:
        """Get comprehensive enforcement statistics."""
        return {
            "total_checks": self.enforcement_stats["total_checks"],
            "actions_taken": dict(self.enforcement_stats["actions_taken"]),
            "cost_savings": self.enforcement_stats["cost_savings"],
            "users_affected": len(self.enforcement_stats["users_affected"]),
            "active_cooldowns": len(self.cooldown_timers),
            "queued_requests": {
                user_id: len(requests) 
                for user_id, requests in self.throttle_queues.items()
                if requests
            },
            "policies_active": len([p for p in self.policies.values() if p.enabled]),
        }
    
    def update_budget_policy(self, policy: BudgetPolicy) -> None:
        """Update a budget policy."""
        policy.updated_at = datetime.now()
        self.policies[policy.user_tier] = policy
        
        logger.info("Budget policy updated", policy_id=policy.policy_id, user_tier=policy.user_tier)
    
    async def cleanup(self) -> None:
        """Clean up enforcement state and save any pending data."""
        # Clear expired cooldowns
        current_time = datetime.now()
        expired_cooldowns = [
            user_id for user_id, end_time in self.cooldown_timers.items()
            if current_time >= end_time
        ]
        
        for user_id in expired_cooldowns:
            del self.cooldown_timers[user_id]
        
        # Clear old enforcement states
        cutoff_time = current_time - timedelta(hours=24)
        users_to_clean = []
        
        for user_id, state in self.user_enforcement_state.items():
            if "grace_period_start" in state:
                if state["grace_period_start"] < cutoff_time:
                    users_to_clean.append(user_id)
        
        for user_id in users_to_clean:
            if "grace_period_start" in self.user_enforcement_state[user_id]:
                del self.user_enforcement_state[user_id]["grace_period_start"]
                del self.user_enforcement_state[user_id]["grace_overage_total"]
        
        logger.info("Budget enforcer cleanup completed", 
                   expired_cooldowns=len(expired_cooldowns),
                   cleaned_states=len(users_to_clean))