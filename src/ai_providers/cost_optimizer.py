"""Cost optimization system with usage tracking and budget enforcement."""

import asyncio
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from structlog import get_logger

from .models import (
    AIRequest,
    AIResponse,
    CostBudget,
    CostTier,
    ModelSpec,
    ProviderType,
    UsageRecord,
)

logger = get_logger(__name__)


class UsageTracker:
    """Tracks usage and costs across AI providers."""
    
    def __init__(self):
        self._usage_history: List[UsageRecord] = []
        self._daily_usage: Dict[str, float] = {}  # date -> cost
        self._monthly_usage: Dict[str, float] = {}  # month -> cost
        self._provider_usage: Dict[ProviderType, float] = defaultdict(float)
        self._session_usage: Dict[str, float] = defaultdict(float)
        self._lock = asyncio.Lock()
    
    async def record_usage(
        self,
        request: AIRequest,
        response: AIResponse,
        provider_type: ProviderType,
        success: bool = True,
        error_message: Optional[str] = None,
    ) -> None:
        """Record usage for a request/response pair.
        
        Args:
            request: The AI request
            response: The AI response (may be None for errors)
            provider_type: Provider that handled the request
            success: Whether the request was successful
            error_message: Error message if request failed
        """
        async with self._lock:
            usage_record = UsageRecord(
                request_id=request.request_id,
                session_id=request.session_id,
                provider_type=provider_type,
                model=request.model,
                input_tokens=response.usage.get("input_tokens", 0) if response and response.usage else 0,
                output_tokens=response.usage.get("output_tokens", 0) if response and response.usage else 0,
                cost=response.cost if response else 0.0,
                latency_ms=response.latency_ms if response else 0.0,
                success=success,
                error_message=error_message,
                metadata={
                    "temperature": request.temperature,
                    "max_tokens": request.max_tokens,
                    "has_tools": bool(request.tools),
                    "streaming": request.stream,
                }
            )
            
            self._usage_history.append(usage_record)
            
            # Update aggregated statistics
            cost = usage_record.cost
            today = datetime.now().date().isoformat()
            current_month = datetime.now().strftime("%Y-%m")
            
            self._daily_usage[today] = self._daily_usage.get(today, 0) + cost
            self._monthly_usage[current_month] = self._monthly_usage.get(current_month, 0) + cost
            self._provider_usage[provider_type] += cost
            
            if request.session_id:
                self._session_usage[request.session_id] += cost
            
            logger.debug(
                "Recorded usage",
                request_id=request.request_id,
                provider=provider_type.value,
                cost=cost,
                success=success,
            )
    
    def get_daily_usage(self, date: Optional[str] = None) -> float:
        """Get total usage for a specific date.
        
        Args:
            date: Date in YYYY-MM-DD format (default: today)
            
        Returns:
            Total cost for the date
        """
        if date is None:
            date = datetime.now().date().isoformat()
        return self._daily_usage.get(date, 0.0)
    
    def get_monthly_usage(self, month: Optional[str] = None) -> float:
        """Get total usage for a specific month.
        
        Args:
            month: Month in YYYY-MM format (default: current month)
            
        Returns:
            Total cost for the month
        """
        if month is None:
            month = datetime.now().strftime("%Y-%m")
        return self._monthly_usage.get(month, 0.0)
    
    def get_provider_usage(self, provider_type: ProviderType) -> float:
        """Get total usage for a specific provider.
        
        Args:
            provider_type: Provider to get usage for
            
        Returns:
            Total cost for the provider
        """
        return self._provider_usage.get(provider_type, 0.0)
    
    def get_session_usage(self, session_id: str) -> float:
        """Get total usage for a specific session.
        
        Args:
            session_id: Session ID to get usage for
            
        Returns:
            Total cost for the session
        """
        return self._session_usage.get(session_id, 0.0)
    
    def get_usage_history(
        self,
        limit: Optional[int] = None,
        provider_type: Optional[ProviderType] = None,
        session_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[UsageRecord]:
        """Get filtered usage history.
        
        Args:
            limit: Maximum number of records to return
            provider_type: Filter by provider type
            session_id: Filter by session ID
            start_date: Filter records after this date
            end_date: Filter records before this date
            
        Returns:
            List of filtered usage records
        """
        filtered_records = []
        
        for record in reversed(self._usage_history):  # Most recent first
            # Apply filters
            if provider_type and record.provider_type != provider_type:
                continue
            if session_id and record.session_id != session_id:
                continue
            if start_date and record.timestamp < start_date:
                continue
            if end_date and record.timestamp > end_date:
                continue
            
            filtered_records.append(record)
            
            # Apply limit
            if limit and len(filtered_records) >= limit:
                break
        
        return filtered_records
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get comprehensive usage statistics.
        
        Returns:
            Dictionary with usage statistics
        """
        total_requests = len(self._usage_history)
        successful_requests = sum(1 for r in self._usage_history if r.success)
        failed_requests = total_requests - successful_requests
        
        total_cost = sum(r.cost for r in self._usage_history)
        total_tokens = sum(r.input_tokens + r.output_tokens for r in self._usage_history)
        
        avg_latency = 0.0
        if self._usage_history:
            avg_latency = sum(r.latency_ms for r in self._usage_history) / len(self._usage_history)
        
        # Cost by provider
        provider_costs = {}
        for provider_type, cost in self._provider_usage.items():
            provider_costs[provider_type.value] = cost
        
        return {
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "failed_requests": failed_requests,
            "success_rate": successful_requests / total_requests if total_requests > 0 else 0.0,
            "total_cost": total_cost,
            "total_tokens": total_tokens,
            "avg_cost_per_request": total_cost / total_requests if total_requests > 0 else 0.0,
            "avg_latency_ms": avg_latency,
            "daily_usage": dict(self._daily_usage),
            "monthly_usage": dict(self._monthly_usage),
            "provider_usage": provider_costs,
            "cost_per_token": total_cost / total_tokens if total_tokens > 0 else 0.0,
        }


class CostOptimizer:
    """Cost optimization system with budget enforcement and smart routing."""
    
    def __init__(self, usage_tracker: UsageTracker):
        self.usage_tracker = usage_tracker
        self._budgets: Dict[str, CostBudget] = {}
        self._provider_models: Dict[ProviderType, Dict[str, ModelSpec]] = {}
        self._cost_cache: Dict[str, float] = {}  # Cache for cost calculations
        
    def add_budget(self, budget: CostBudget) -> None:
        """Add a cost budget for monitoring.
        
        Args:
            budget: Cost budget configuration
        """
        self._budgets[budget.budget_id] = budget
        logger.info(
            "Added cost budget",
            budget_id=budget.budget_id,
            daily_limit=budget.daily_limit,
            monthly_limit=budget.monthly_limit,
        )
    
    def remove_budget(self, budget_id: str) -> None:
        """Remove a cost budget.
        
        Args:
            budget_id: ID of budget to remove
        """
        if budget_id in self._budgets:
            del self._budgets[budget_id]
            logger.info("Removed cost budget", budget_id=budget_id)
    
    def register_provider_models(self, provider_type: ProviderType, models: Dict[str, ModelSpec]) -> None:
        """Register models for a provider for cost optimization.
        
        Args:
            provider_type: Provider type
            models: Dictionary of model_id -> ModelSpec
        """
        self._provider_models[provider_type] = models
        logger.info(
            "Registered provider models for cost optimization",
            provider=provider_type.value,
            models=len(models),
        )
    
    async def check_budget_limits(
        self,
        request: AIRequest,
        estimated_cost: float,
        provider_type: ProviderType,
    ) -> Tuple[bool, List[str]]:
        """Check if a request would exceed budget limits.
        
        Args:
            request: The AI request to check
            estimated_cost: Estimated cost of the request
            provider_type: Provider that would handle the request
            
        Returns:
            Tuple of (allowed, reasons) where allowed is True if within budget
        """
        violations = []
        
        # Check per-request budget limit
        if request.budget_limit and estimated_cost > request.budget_limit:
            violations.append(f"Request cost ${estimated_cost:.4f} exceeds limit ${request.budget_limit:.4f}")
        
        # Check all configured budgets
        for budget in self._budgets.values():
            if not budget.enabled:
                continue
            
            # Check daily limit
            if budget.daily_limit:
                current_daily = self.usage_tracker.get_daily_usage()
                if current_daily + estimated_cost > budget.daily_limit:
                    violations.append(
                        f"Daily budget exceeded: ${current_daily + estimated_cost:.4f} > ${budget.daily_limit:.4f}"
                    )
            
            # Check monthly limit
            if budget.monthly_limit:
                current_monthly = self.usage_tracker.get_monthly_usage()
                if current_monthly + estimated_cost > budget.monthly_limit:
                    violations.append(
                        f"Monthly budget exceeded: ${current_monthly + estimated_cost:.4f} > ${budget.monthly_limit:.4f}"
                    )
            
            # Check provider-specific limits
            if provider_type in budget.provider_limits:
                provider_limit = budget.provider_limits[provider_type]
                current_provider = self.usage_tracker.get_provider_usage(provider_type)
                if current_provider + estimated_cost > provider_limit:
                    violations.append(
                        f"Provider {provider_type.value} budget exceeded: "
                        f"${current_provider + estimated_cost:.4f} > ${provider_limit:.4f}"
                    )
        
        return len(violations) == 0, violations
    
    def estimate_request_cost(
        self,
        request: AIRequest,
        provider_type: ProviderType,
    ) -> float:
        """Estimate the cost of a request for a specific provider.
        
        Args:
            request: The AI request
            provider_type: Provider that would handle the request
            
        Returns:
            Estimated cost in USD
        """
        # Check cache first
        cache_key = f"{provider_type.value}:{request.model}:{len(str(request.messages))}"
        if cache_key in self._cost_cache:
            return self._cost_cache[cache_key]
        
        if provider_type not in self._provider_models:
            logger.warning("No models registered for provider", provider=provider_type.value)
            return 0.0
        
        models = self._provider_models[provider_type]
        if request.model not in models:
            logger.warning(
                "Model not found for cost estimation",
                provider=provider_type.value,
                model=request.model,
            )
            return 0.0
        
        model_spec = models[request.model]
        
        # Estimate input tokens
        input_tokens = self._estimate_input_tokens(request.messages)
        
        # Estimate output tokens (use max_tokens or model default)
        output_tokens = request.max_tokens or model_spec.max_output_tokens
        output_tokens = min(output_tokens, model_spec.max_output_tokens)
        
        # Calculate cost
        input_cost = (input_tokens / 1000) * model_spec.cost_per_input_token
        output_cost = (output_tokens / 1000) * model_spec.cost_per_output_token
        total_cost = input_cost + output_cost
        
        # Cache the result
        self._cost_cache[cache_key] = total_cost
        
        return total_cost
    
    def find_cheapest_provider(
        self,
        request: AIRequest,
        available_providers: List[ProviderType],
        required_capabilities: Optional[List[str]] = None,
    ) -> Optional[Tuple[ProviderType, str, float]]:
        """Find the cheapest provider that can handle the request.
        
        Args:
            request: The AI request
            available_providers: List of available provider types
            required_capabilities: Required capabilities (e.g., "streaming", "tools")
            
        Returns:
            Tuple of (provider_type, model_id, estimated_cost) or None if no suitable provider
        """
        candidates = []
        
        for provider_type in available_providers:
            if provider_type not in self._provider_models:
                continue
            
            models = self._provider_models[provider_type]
            
            for model_id, model_spec in models.items():
                if not model_spec.is_available:
                    continue
                
                # Check capability requirements
                if required_capabilities:
                    if "streaming" in required_capabilities and not model_spec.supports_streaming:
                        continue
                    if "tools" in required_capabilities and not model_spec.supports_tools:
                        continue
                    if "vision" in required_capabilities and not model_spec.supports_vision:
                        continue
                
                # Check context length
                estimated_tokens = self._estimate_input_tokens(request.messages)
                if estimated_tokens > model_spec.context_length:
                    continue
                
                # Calculate cost
                cost = self.estimate_request_cost(
                    AIRequest(
                        model=model_id,
                        messages=request.messages,
                        max_tokens=request.max_tokens,
                    ),
                    provider_type,
                )
                
                candidates.append((provider_type, model_id, cost, model_spec.cost_tier))
        
        if not candidates:
            return None
        
        # Sort by cost (ascending) and then by cost tier preference
        candidates.sort(key=lambda x: (x[2], x[3].value))
        
        return candidates[0][:3]  # Return provider, model, cost
    
    def get_cost_efficient_models(
        self,
        provider_type: ProviderType,
        max_cost_tier: CostTier = CostTier.PREMIUM,
    ) -> List[Tuple[str, ModelSpec, float]]:
        """Get cost-efficient models for a provider, ranked by cost effectiveness.
        
        Args:
            provider_type: Provider type
            max_cost_tier: Maximum allowed cost tier
            
        Returns:
            List of (model_id, model_spec, cost_per_1k_tokens) sorted by efficiency
        """
        if provider_type not in self._provider_models:
            return []
        
        models = self._provider_models[provider_type]
        candidates = []
        
        # Define cost tier ordering
        tier_order = {
            CostTier.FREE: 0,
            CostTier.LOW: 1, 
            CostTier.MEDIUM: 2,
            CostTier.HIGH: 3,
            CostTier.PREMIUM: 4,
        }
        
        max_tier_value = tier_order[max_cost_tier]
        
        for model_id, model_spec in models.items():
            if not model_spec.is_available:
                continue
            
            if tier_order[model_spec.cost_tier] > max_tier_value:
                continue
            
            # Calculate average cost per 1K tokens (input + output)
            avg_cost = (model_spec.cost_per_input_token + model_spec.cost_per_output_token) / 2
            
            candidates.append((model_id, model_spec, avg_cost))
        
        # Sort by cost per token
        candidates.sort(key=lambda x: x[2])
        
        return candidates
    
    def get_budget_alerts(self) -> List[Dict[str, Any]]:
        """Get budget alerts for budgets approaching limits.
        
        Returns:
            List of alert dictionaries
        """
        alerts = []
        
        for budget in self._budgets.values():
            if not budget.enabled:
                continue
            
            # Check daily budget
            if budget.daily_limit:
                current_daily = self.usage_tracker.get_daily_usage()
                usage_percentage = (current_daily / budget.daily_limit) * 100
                
                for threshold in budget.alert_thresholds:
                    if usage_percentage >= threshold * 100:
                        alerts.append({
                            "type": "daily_budget_alert",
                            "budget_id": budget.budget_id,
                            "budget_name": budget.name,
                            "threshold": threshold,
                            "current_usage": current_daily,
                            "limit": budget.daily_limit,
                            "percentage": usage_percentage,
                        })
                        break
            
            # Check monthly budget
            if budget.monthly_limit:
                current_monthly = self.usage_tracker.get_monthly_usage()
                usage_percentage = (current_monthly / budget.monthly_limit) * 100
                
                for threshold in budget.alert_thresholds:
                    if usage_percentage >= threshold * 100:
                        alerts.append({
                            "type": "monthly_budget_alert",
                            "budget_id": budget.budget_id,
                            "budget_name": budget.name,
                            "threshold": threshold,
                            "current_usage": current_monthly,
                            "limit": budget.monthly_limit,
                            "percentage": usage_percentage,
                        })
                        break
        
        return alerts
    
    def _estimate_input_tokens(self, messages: List[Dict[str, Any]]) -> int:
        """Estimate input token count for messages.
        
        Args:
            messages: List of messages
            
        Returns:
            Estimated token count
        """
        total_chars = 0
        
        for message in messages:
            content = message.get("content", "")
            if isinstance(content, str):
                total_chars += len(content)
            elif isinstance(content, list):
                # Handle multimodal content
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        total_chars += len(item.get("text", ""))
        
        # Add overhead for message structure (role, formatting, etc.)
        total_chars += len(messages) * 20
        
        # Rough estimation: 4 characters per token
        return total_chars // 4
    
    def optimize_request_routing(
        self,
        request: AIRequest,
        available_providers: List[ProviderType],
        optimization_strategy: str = "cost",
    ) -> Optional[Dict[str, Any]]:
        """Optimize request routing based on strategy.
        
        Args:
            request: The AI request
            available_providers: Available providers
            optimization_strategy: "cost", "speed", "quality", or "balanced"
            
        Returns:
            Routing recommendation dictionary or None
        """
        if optimization_strategy == "cost":
            result = self.find_cheapest_provider(request, available_providers)
            if result:
                provider, model, cost = result
                return {
                    "provider": provider,
                    "model": model,
                    "estimated_cost": cost,
                    "strategy": "cost",
                    "reason": "Cheapest available option",
                }
        
        elif optimization_strategy == "speed":
            # For speed optimization, prefer providers with lower latency models
            fastest_option = None
            min_latency = float('inf')
            
            for provider_type in available_providers:
                if provider_type not in self._provider_models:
                    continue
                
                models = self._provider_models[provider_type]
                for model_id, model_spec in models.items():
                    if not model_spec.is_available:
                        continue
                    
                    # Estimate latency (smaller models are typically faster)
                    estimated_latency = model_spec.max_output_tokens / 100  # Rough heuristic
                    
                    if estimated_latency < min_latency:
                        min_latency = estimated_latency
                        cost = self.estimate_request_cost(
                            AIRequest(model=model_id, messages=request.messages), provider_type
                        )
                        fastest_option = {
                            "provider": provider_type,
                            "model": model_id,
                            "estimated_cost": cost,
                            "estimated_latency": estimated_latency,
                            "strategy": "speed",
                            "reason": "Fastest available option",
                        }
            
            return fastest_option
        
        elif optimization_strategy == "balanced":
            # Balanced optimization considers both cost and capabilities
            candidates = []
            
            for provider_type in available_providers:
                if provider_type not in self._provider_models:
                    continue
                
                models = self._provider_models[provider_type]
                for model_id, model_spec in models.items():
                    if not model_spec.is_available:
                        continue
                    
                    cost = self.estimate_request_cost(
                        AIRequest(model=model_id, messages=request.messages), provider_type
                    )
                    
                    # Score based on cost tier (lower is better) and capabilities
                    tier_score = {
                        CostTier.FREE: 1,
                        CostTier.LOW: 2,
                        CostTier.MEDIUM: 3,
                        CostTier.HIGH: 4,
                        CostTier.PREMIUM: 5,
                    }[model_spec.cost_tier]
                    
                    capability_score = len(model_spec.capabilities)
                    
                    # Balanced score: lower cost tier + more capabilities = better
                    balance_score = tier_score - (capability_score * 0.5)
                    
                    candidates.append((provider_type, model_id, cost, balance_score))
            
            if candidates:
                candidates.sort(key=lambda x: x[3])  # Sort by balance score
                provider, model, cost, _ = candidates[0]
                
                return {
                    "provider": provider,
                    "model": model,
                    "estimated_cost": cost,
                    "strategy": "balanced",
                    "reason": "Best balance of cost and capabilities",
                }
        
        return None