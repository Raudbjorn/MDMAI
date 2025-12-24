"""Real-time cost calculation engine with dynamic pricing models."""

import json
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import asyncio
from structlog import get_logger

from .models import ProviderType
from .enhanced_token_estimator import EnhancedTokenEstimator

logger = get_logger(__name__)


class PricingModel(Enum):
    """Pricing model types."""
    
    TOKEN_BASED = "token_based"  # Per token pricing
    REQUEST_BASED = "request_based"  # Per request pricing
    TIME_BASED = "time_based"  # Per minute/hour pricing
    TIERED = "tiered"  # Volume-based tiers
    DYNAMIC = "dynamic"  # Dynamic pricing based on demand


@dataclass
class PricingTier:
    """Pricing tier for volume-based pricing."""
    
    min_tokens: int
    max_tokens: Optional[int]  # None for highest tier
    input_rate: float  # USD per 1K tokens
    output_rate: float  # USD per 1K tokens
    discount_percentage: float = 0.0


@dataclass
class ModelPricing:
    """Pricing configuration for a specific model."""
    
    provider_type: ProviderType
    model_id: str
    pricing_model: PricingModel
    input_token_rate: float = 0.0  # USD per 1K input tokens
    output_token_rate: float = 0.0  # USD per 1K output tokens
    request_rate: float = 0.0  # USD per request
    time_rate: float = 0.0  # USD per minute
    minimum_charge: float = 0.0  # Minimum charge per request
    currency: str = "USD"
    effective_date: datetime = field(default_factory=datetime.now)
    expires_date: Optional[datetime] = None
    tiers: List[PricingTier] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DynamicPricingFactor:
    """Factors affecting dynamic pricing."""
    
    base_multiplier: float = 1.0
    demand_multiplier: float = 1.0
    time_of_day_multiplier: float = 1.0
    provider_health_multiplier: float = 1.0
    volume_discount: float = 0.0
    effective_multiplier: float = 1.0


class PricingEngine:
    """Real-time cost calculation engine with dynamic pricing support."""
    
    def __init__(self, pricing_data_path: Optional[str] = None):
        self.pricing_data_path = Path(pricing_data_path) if pricing_data_path else Path("./data/pricing")
        self.pricing_data_path.mkdir(parents=True, exist_ok=True)
        
        self.token_estimator = EnhancedTokenEstimator()
        self.pricing_models: Dict[str, ModelPricing] = {}
        self.pricing_cache: Dict[str, Tuple[float, float]] = {}  # Cache for cost calculations
        self.usage_stats: Dict[str, Dict[str, Any]] = {}  # Track usage for dynamic pricing
        
        # Rate limiting and caching
        self._last_pricing_update = 0
        self._pricing_update_interval = 300  # 5 minutes
        self._cache_ttl = 60  # 1 minute cache TTL
        
        # Load initial pricing data
        self._load_pricing_data()
        
        logger.info("Pricing engine initialized", models_loaded=len(self.pricing_models))
    
    def _load_pricing_data(self) -> None:
        """Load pricing data from JSON files."""
        try:
            pricing_file = self.pricing_data_path / "model_pricing.json"
            if pricing_file.exists():
                with open(pricing_file, 'r') as f:
                    pricing_data = json.load(f)
                
                for model_data in pricing_data.get("models", []):
                    pricing = ModelPricing(
                        provider_type=ProviderType(model_data["provider_type"]),
                        model_id=model_data["model_id"],
                        pricing_model=PricingModel(model_data["pricing_model"]),
                        input_token_rate=model_data.get("input_token_rate", 0.0),
                        output_token_rate=model_data.get("output_token_rate", 0.0),
                        request_rate=model_data.get("request_rate", 0.0),
                        time_rate=model_data.get("time_rate", 0.0),
                        minimum_charge=model_data.get("minimum_charge", 0.0),
                        effective_date=datetime.fromisoformat(model_data.get("effective_date", datetime.now().isoformat())),
                        expires_date=datetime.fromisoformat(model_data["expires_date"]) if model_data.get("expires_date") else None,
                        tiers=[
                            PricingTier(**tier) for tier in model_data.get("tiers", [])
                        ],
                        metadata=model_data.get("metadata", {})
                    )
                    
                    key = f"{pricing.provider_type.value}:{pricing.model_id}"
                    self.pricing_models[key] = pricing
                
                logger.info("Pricing data loaded", models=len(self.pricing_models))
            else:
                # Create default pricing data
                self._create_default_pricing()
                
        except Exception as e:
            logger.error("Failed to load pricing data", error=str(e))
            self._create_default_pricing()
    
    def _create_default_pricing(self) -> None:
        """Create default pricing models for supported providers."""
        default_models = [
            # OpenAI Models (as of 2024)
            {
                "provider_type": "openai",
                "model_id": "gpt-4",
                "pricing_model": "token_based",
                "input_token_rate": 0.03,  # $0.03 per 1K input tokens
                "output_token_rate": 0.06,  # $0.06 per 1K output tokens
                "minimum_charge": 0.0001,
            },
            {
                "provider_type": "openai", 
                "model_id": "gpt-4-turbo",
                "pricing_model": "token_based",
                "input_token_rate": 0.01,
                "output_token_rate": 0.03,
                "minimum_charge": 0.0001,
            },
            {
                "provider_type": "openai",
                "model_id": "gpt-3.5-turbo",
                "pricing_model": "token_based", 
                "input_token_rate": 0.0015,
                "output_token_rate": 0.002,
                "minimum_charge": 0.00005,
            },
            
            # Anthropic Models (as of 2024)
            {
                "provider_type": "anthropic",
                "model_id": "claude-3-opus",
                "pricing_model": "token_based",
                "input_token_rate": 0.015,
                "output_token_rate": 0.075,
                "minimum_charge": 0.0001,
            },
            {
                "provider_type": "anthropic",
                "model_id": "claude-3-sonnet", 
                "pricing_model": "token_based",
                "input_token_rate": 0.003,
                "output_token_rate": 0.015,
                "minimum_charge": 0.00005,
            },
            {
                "provider_type": "anthropic",
                "model_id": "claude-3-haiku",
                "pricing_model": "token_based",
                "input_token_rate": 0.00025,
                "output_token_rate": 0.00125,
                "minimum_charge": 0.00001,
            },
            
            # Google Models (as of 2024)
            {
                "provider_type": "google",
                "model_id": "gemini-pro",
                "pricing_model": "token_based",
                "input_token_rate": 0.0005,
                "output_token_rate": 0.0015,
                "minimum_charge": 0.00001,
            },
            {
                "provider_type": "google",
                "model_id": "gemini-pro-vision",
                "pricing_model": "token_based",
                "input_token_rate": 0.0005,
                "output_token_rate": 0.0015,
                "minimum_charge": 0.00001,
            },
        ]
        
        for model_data in default_models:
            pricing = ModelPricing(
                provider_type=ProviderType(model_data["provider_type"]),
                model_id=model_data["model_id"],
                pricing_model=PricingModel(model_data["pricing_model"]),
                input_token_rate=model_data["input_token_rate"],
                output_token_rate=model_data["output_token_rate"],
                minimum_charge=model_data["minimum_charge"],
            )
            
            key = f"{pricing.provider_type.value}:{pricing.model_id}"
            self.pricing_models[key] = pricing
        
        # Save default pricing to file
        self._save_pricing_data()
        logger.info("Default pricing models created", count=len(default_models))
    
    def _save_pricing_data(self) -> None:
        """Save current pricing data to JSON file."""
        try:
            pricing_data = {
                "last_updated": datetime.now().isoformat(),
                "models": []
            }
            
            for pricing in self.pricing_models.values():
                model_data = {
                    "provider_type": pricing.provider_type.value,
                    "model_id": pricing.model_id,
                    "pricing_model": pricing.pricing_model.value,
                    "input_token_rate": pricing.input_token_rate,
                    "output_token_rate": pricing.output_token_rate,
                    "request_rate": pricing.request_rate,
                    "time_rate": pricing.time_rate,
                    "minimum_charge": pricing.minimum_charge,
                    "effective_date": pricing.effective_date.isoformat(),
                    "expires_date": pricing.expires_date.isoformat() if pricing.expires_date else None,
                    "tiers": [
                        {
                            "min_tokens": tier.min_tokens,
                            "max_tokens": tier.max_tokens,
                            "input_rate": tier.input_rate,
                            "output_rate": tier.output_rate,
                            "discount_percentage": tier.discount_percentage,
                        }
                        for tier in pricing.tiers
                    ],
                    "metadata": pricing.metadata,
                }
                pricing_data["models"].append(model_data)
            
            pricing_file = self.pricing_data_path / "model_pricing.json"
            with open(pricing_file, 'w') as f:
                json.dump(pricing_data, f, indent=2)
                
        except Exception as e:
            logger.error("Failed to save pricing data", error=str(e))
    
    def get_model_pricing(self, provider_type: ProviderType, model_id: str) -> Optional[ModelPricing]:
        """Get pricing model for a specific provider and model."""
        key = f"{provider_type.value}:{model_id}"
        return self.pricing_models.get(key)
    
    def calculate_request_cost(
        self,
        provider_type: ProviderType,
        model_id: str,
        input_tokens: int,
        output_tokens: int,
        processing_time_ms: Optional[float] = None,
        request_count: int = 1,
        dynamic_factors: Optional[DynamicPricingFactor] = None
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate the cost of a request with detailed breakdown.
        
        Returns:
            Tuple of (total_cost, cost_breakdown)
        """
        pricing = self.get_model_pricing(provider_type, model_id)
        
        if not pricing:
            logger.warning("No pricing model found", provider=provider_type.value, model=model_id)
            return 0.0, {"error": "No pricing model found"}
        
        # Check cache
        cache_key = f"{provider_type.value}:{model_id}:{input_tokens}:{output_tokens}"
        cache_entry = self.pricing_cache.get(cache_key)
        if cache_entry and (time.time() - cache_entry[0]) < self._cache_ttl:
            return cache_entry[1], {"cached": True}
        
        cost_breakdown = {
            "provider": provider_type.value,
            "model": model_id,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "pricing_model": pricing.pricing_model.value,
        }
        
        base_cost = 0.0
        
        if pricing.pricing_model == PricingModel.TOKEN_BASED:
            # Calculate token-based cost
            input_cost = (input_tokens / 1000) * pricing.input_token_rate
            output_cost = (output_tokens / 1000) * pricing.output_token_rate
            base_cost = input_cost + output_cost
            
            cost_breakdown.update({
                "input_cost": input_cost,
                "output_cost": output_cost,
                "input_rate_per_1k": pricing.input_token_rate,
                "output_rate_per_1k": pricing.output_token_rate,
            })
            
            # Apply tiered pricing if configured
            if pricing.tiers:
                total_tokens = input_tokens + output_tokens
                tier = self._get_pricing_tier(pricing.tiers, total_tokens)
                if tier:
                    # Recalculate with tiered rates
                    input_cost = (input_tokens / 1000) * tier.input_rate
                    output_cost = (output_tokens / 1000) * tier.output_rate
                    base_cost = input_cost + output_cost
                    
                    # Apply discount
                    if tier.discount_percentage > 0:
                        discount = base_cost * (tier.discount_percentage / 100)
                        base_cost -= discount
                        cost_breakdown["tier_discount"] = discount
                    
                    cost_breakdown.update({
                        "tier": {
                            "min_tokens": tier.min_tokens,
                            "max_tokens": tier.max_tokens,
                            "discount_percentage": tier.discount_percentage,
                        }
                    })
        
        elif pricing.pricing_model == PricingModel.REQUEST_BASED:
            base_cost = pricing.request_rate * request_count
            cost_breakdown["request_rate"] = pricing.request_rate
            cost_breakdown["request_count"] = request_count
        
        elif pricing.pricing_model == PricingModel.TIME_BASED:
            if processing_time_ms:
                minutes = processing_time_ms / (1000 * 60)
                base_cost = minutes * pricing.time_rate
                cost_breakdown["processing_minutes"] = minutes
                cost_breakdown["time_rate"] = pricing.time_rate
            else:
                # Fallback to estimated time based on tokens
                estimated_minutes = (input_tokens + output_tokens) / 10000  # Rough estimate
                base_cost = estimated_minutes * pricing.time_rate
                cost_breakdown["estimated_minutes"] = estimated_minutes
                cost_breakdown["time_rate"] = pricing.time_rate
        
        # Apply minimum charge
        if base_cost < pricing.minimum_charge:
            cost_breakdown["minimum_charge_applied"] = pricing.minimum_charge - base_cost
            base_cost = pricing.minimum_charge
        
        # Apply dynamic pricing factors
        if dynamic_factors:
            base_cost *= dynamic_factors.effective_multiplier
            cost_breakdown["dynamic_factors"] = {
                "base_multiplier": dynamic_factors.base_multiplier,
                "demand_multiplier": dynamic_factors.demand_multiplier,
                "time_of_day_multiplier": dynamic_factors.time_of_day_multiplier,
                "provider_health_multiplier": dynamic_factors.provider_health_multiplier,
                "volume_discount": dynamic_factors.volume_discount,
                "effective_multiplier": dynamic_factors.effective_multiplier,
            }
        
        cost_breakdown["base_cost"] = base_cost
        cost_breakdown["total_cost"] = base_cost
        
        # Cache the result
        self.pricing_cache[cache_key] = (time.time(), base_cost)
        
        return base_cost, cost_breakdown
    
    def estimate_request_cost(
        self,
        provider_type: ProviderType,
        model_id: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        max_output_tokens: Optional[int] = None,
        dynamic_factors: Optional[DynamicPricingFactor] = None
    ) -> Tuple[float, Dict[str, Any]]:
        """Estimate cost before making a request."""
        # Estimate tokens
        input_tokens, estimated_output_tokens = self.token_estimator.estimate_request_tokens(
            provider_type, model_id, messages, tools, max_output_tokens
        )
        
        # Calculate estimated cost
        return self.calculate_request_cost(
            provider_type,
            model_id,
            input_tokens,
            estimated_output_tokens,
            dynamic_factors=dynamic_factors
        )
    
    def _get_pricing_tier(self, tiers: List[PricingTier], total_tokens: int) -> Optional[PricingTier]:
        """Get the appropriate pricing tier for token volume."""
        for tier in sorted(tiers, key=lambda t: t.min_tokens):
            if tier.min_tokens <= total_tokens and (tier.max_tokens is None or total_tokens <= tier.max_tokens):
                return tier
        return None
    
    def calculate_dynamic_factors(
        self,
        provider_type: ProviderType,
        model_id: str,
        current_usage: Dict[str, Any],
        provider_health: Optional[Dict[str, Any]] = None
    ) -> DynamicPricingFactor:
        """Calculate dynamic pricing factors based on current conditions."""
        factors = DynamicPricingFactor()
        
        # Time-of-day pricing (peak hours cost more)
        current_hour = datetime.now().hour
        if 9 <= current_hour <= 17:  # Business hours
            factors.time_of_day_multiplier = 1.2
        elif 18 <= current_hour <= 22:  # Evening hours
            factors.time_of_day_multiplier = 1.1
        else:  # Off-peak hours
            factors.time_of_day_multiplier = 0.9
        
        # Demand-based pricing
        recent_requests = current_usage.get("recent_requests", 0)
        if recent_requests > 100:  # High demand
            factors.demand_multiplier = 1.3
        elif recent_requests > 50:  # Medium demand
            factors.demand_multiplier = 1.1
        else:  # Low demand
            factors.demand_multiplier = 0.95
        
        # Provider health impact
        if provider_health:
            error_rate = provider_health.get("error_rate", 0)
            latency = provider_health.get("avg_latency_ms", 1000)
            
            if error_rate > 0.1 or latency > 5000:  # Poor health
                factors.provider_health_multiplier = 1.5
            elif error_rate < 0.01 and latency < 1000:  # Excellent health
                factors.provider_health_multiplier = 0.9
        
        # Volume discounts
        monthly_tokens = current_usage.get("monthly_tokens", 0)
        if monthly_tokens > 1_000_000:  # High volume
            factors.volume_discount = 0.15
        elif monthly_tokens > 100_000:  # Medium volume
            factors.volume_discount = 0.05
        
        # Calculate effective multiplier
        factors.effective_multiplier = (
            factors.base_multiplier *
            factors.demand_multiplier *
            factors.time_of_day_multiplier *
            factors.provider_health_multiplier *
            (1 - factors.volume_discount)
        )
        
        return factors
    
    def update_pricing_model(self, pricing: ModelPricing) -> None:
        """Update or add a pricing model."""
        key = f"{pricing.provider_type.value}:{pricing.model_id}"
        self.pricing_models[key] = pricing
        
        # Clear cache for this model
        cache_keys_to_remove = [k for k in self.pricing_cache.keys() if k.startswith(key)]
        for cache_key in cache_keys_to_remove:
            del self.pricing_cache[cache_key]
        
        # Save to disk
        self._save_pricing_data()
        
        logger.info("Pricing model updated", provider=pricing.provider_type.value, model=pricing.model_id)
    
    async def fetch_latest_pricing(self) -> None:
        """Fetch latest pricing from provider APIs (if available)."""
        # This would be implemented to fetch real-time pricing from provider APIs
        # For now, we'll just update the timestamp
        current_time = time.time()
        if current_time - self._last_pricing_update > self._pricing_update_interval:
            self._last_pricing_update = current_time
            logger.debug("Pricing data refresh check completed")
    
    def get_pricing_summary(self) -> Dict[str, Any]:
        """Get a summary of all pricing models."""
        summary = {
            "total_models": len(self.pricing_models),
            "providers": {},
            "pricing_types": {},
            "cache_stats": {
                "size": len(self.pricing_cache),
                "hit_rate": 0.0,  # Would need to track hits/misses
            }
        }
        
        for pricing in self.pricing_models.values():
            provider = pricing.provider_type.value
            if provider not in summary["providers"]:
                summary["providers"][provider] = {"models": [], "avg_input_rate": 0, "avg_output_rate": 0}
            
            summary["providers"][provider]["models"].append({
                "model": pricing.model_id,
                "input_rate": pricing.input_token_rate,
                "output_rate": pricing.output_token_rate,
                "minimum_charge": pricing.minimum_charge,
            })
            
            pricing_type = pricing.pricing_model.value
            summary["pricing_types"][pricing_type] = summary["pricing_types"].get(pricing_type, 0) + 1
        
        # Calculate average rates per provider
        for provider_data in summary["providers"].values():
            models = provider_data["models"]
            if models:
                provider_data["avg_input_rate"] = sum(m["input_rate"] for m in models) / len(models)
                provider_data["avg_output_rate"] = sum(m["output_rate"] for m in models) / len(models)
        
        return summary
    
    def clear_cache(self) -> None:
        """Clear the pricing cache."""
        self.pricing_cache.clear()
        logger.info("Pricing cache cleared")