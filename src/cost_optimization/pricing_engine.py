"""
Provider Pricing Engine with Real-time Cost Formulas and Volume Discounts.

This module provides sophisticated pricing models for different AI providers:
- Real-time pricing updates from provider APIs
- Volume-based discount calculations
- Time-based pricing (peak/off-peak)
- Model-specific cost formulas
- Enterprise tier pricing
- Currency conversion and regional pricing
- Cost optimization recommendations
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from decimal import Decimal, getcontext
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import aiohttp
from structlog import get_logger

from ..ai_providers.models import ProviderType
from ..usage_tracking.storage.models import ProviderType as TrackingProviderType

# Set decimal precision for financial calculations
getcontext().prec = 10

logger = get_logger(__name__)


class PricingTier(Enum):
    """Pricing tiers for different usage levels."""
    FREE = "free"
    STARTER = "starter"
    PRO = "pro"
    ENTERPRISE = "enterprise"
    CUSTOM = "custom"


class PricingModel(Enum):
    """Different pricing model types."""
    PAY_PER_TOKEN = "pay_per_token"
    PAY_PER_REQUEST = "pay_per_request"
    SUBSCRIPTION = "subscription"
    CREDITS = "credits"
    HYBRID = "hybrid"


class CostComponent(Enum):
    """Different components of API costs."""
    INPUT_TOKENS = "input_tokens"
    OUTPUT_TOKENS = "output_tokens"
    TOOL_CALLS = "tool_calls"
    IMAGE_PROCESSING = "image_processing"
    AUDIO_PROCESSING = "audio_processing"
    EMBEDDING_GENERATION = "embedding_generation"
    FINE_TUNING = "fine_tuning"
    BATCH_PROCESSING = "batch_processing"


class ModelPricingInfo:
    """Pricing information for a specific model."""
    
    def __init__(
        self,
        model_id: str,
        provider: ProviderType,
        pricing_model: PricingModel = PricingModel.PAY_PER_TOKEN
    ):
        self.model_id = model_id
        self.provider = provider
        self.pricing_model = pricing_model
        
        # Base pricing (per 1K tokens unless specified)
        self.base_prices = {}  # CostComponent -> Decimal
        
        # Tier-based pricing
        self.tier_multipliers = {}  # PricingTier -> float
        
        # Volume discounts (usage_threshold -> discount_factor)
        self.volume_discounts = {}  # Decimal -> float
        
        # Time-based pricing
        self.peak_hours = set()  # Hours when pricing is higher
        self.peak_multiplier = Decimal("1.0")
        self.off_peak_multiplier = Decimal("0.8")
        
        # Regional pricing
        self.regional_multipliers = {}  # region_code -> float
        
        # Currency
        self.base_currency = "USD"
        
        # Metadata
        self.last_updated = datetime.utcnow()
        self.pricing_source = "manual"
    
    def set_base_price(self, component: CostComponent, price: Union[Decimal, float]) -> None:
        """Set base price for a cost component."""
        self.base_prices[component] = Decimal(str(price))
        self.last_updated = datetime.utcnow()
    
    def set_tier_multiplier(self, tier: PricingTier, multiplier: float) -> None:
        """Set pricing multiplier for a tier."""
        self.tier_multipliers[tier] = multiplier
    
    def add_volume_discount(self, usage_threshold: Union[Decimal, float], discount_factor: float) -> None:
        """Add volume discount (discount_factor < 1.0 means discount)."""
        self.volume_discounts[Decimal(str(usage_threshold))] = discount_factor
    
    def set_time_based_pricing(
        self,
        peak_hours: List[int],
        peak_multiplier: float = 1.2,
        off_peak_multiplier: float = 0.8
    ) -> None:
        """Configure time-based pricing."""
        self.peak_hours = set(peak_hours)
        self.peak_multiplier = Decimal(str(peak_multiplier))
        self.off_peak_multiplier = Decimal(str(off_peak_multiplier))
    
    def calculate_cost(
        self,
        usage_amounts: Dict[CostComponent, int],
        tier: PricingTier = PricingTier.PRO,
        monthly_usage: Decimal = Decimal("0"),
        request_time: Optional[datetime] = None,
        region: str = "us"
    ) -> Dict[str, Any]:
        """Calculate cost with all pricing factors."""
        
        request_time = request_time or datetime.utcnow()
        total_cost = Decimal("0")
        cost_breakdown = {}
        
        # Base cost calculation
        for component, amount in usage_amounts.items():
            if component not in self.base_prices:
                continue
            
            base_price = self.base_prices[component]
            
            if self.pricing_model == PricingModel.PAY_PER_TOKEN:
                # Price per 1K tokens
                component_cost = (Decimal(str(amount)) / Decimal("1000")) * base_price
            else:
                # Price per unit
                component_cost = Decimal(str(amount)) * base_price
            
            cost_breakdown[component.value] = {
                'base_cost': float(component_cost),
                'units': amount,
                'rate': float(base_price)
            }
            
            total_cost += component_cost
        
        # Apply tier multiplier
        tier_multiplier = self.tier_multipliers.get(tier, 1.0)
        if tier_multiplier != 1.0:
            total_cost *= Decimal(str(tier_multiplier))
        
        # Apply volume discount
        volume_discount_factor = self._get_volume_discount(monthly_usage)
        if volume_discount_factor != 1.0:
            total_cost *= Decimal(str(volume_discount_factor))
        
        # Apply time-based pricing
        time_multiplier = self._get_time_multiplier(request_time)
        if time_multiplier != 1.0:
            total_cost *= time_multiplier
        
        # Apply regional multiplier
        regional_multiplier = self.regional_multipliers.get(region, 1.0)
        if regional_multiplier != 1.0:
            total_cost *= Decimal(str(regional_multiplier))
        
        return {
            'total_cost': float(total_cost),
            'cost_breakdown': cost_breakdown,
            'applied_factors': {
                'tier': tier.value,
                'tier_multiplier': tier_multiplier,
                'volume_discount_factor': volume_discount_factor,
                'time_multiplier': float(time_multiplier),
                'regional_multiplier': regional_multiplier,
                'currency': self.base_currency
            },
            'pricing_timestamp': request_time.isoformat(),
            'pricing_model': self.pricing_model.value
        }
    
    def _get_volume_discount(self, monthly_usage: Decimal) -> float:
        """Get volume discount factor based on monthly usage."""
        if not self.volume_discounts:
            return 1.0
        
        # Find applicable discount tier
        applicable_discount = 1.0
        for threshold in sorted(self.volume_discounts.keys()):
            if monthly_usage >= threshold:
                applicable_discount = self.volume_discounts[threshold]
            else:
                break
        
        return applicable_discount
    
    def _get_time_multiplier(self, request_time: datetime) -> Decimal:
        """Get time-based pricing multiplier."""
        if not self.peak_hours:
            return Decimal("1.0")
        
        hour = request_time.hour
        if hour in self.peak_hours:
            return self.peak_multiplier
        else:
            return self.off_peak_multiplier


class ProviderPricingConfig:
    """Pricing configuration for a provider."""
    
    def __init__(self, provider: ProviderType):
        self.provider = provider
        self.models = {}  # model_id -> ModelPricingInfo
        self.api_endpoint = None  # For real-time pricing updates
        self.api_key = None
        self.update_frequency_hours = 24
        self.last_api_update = None
        
        # Provider-wide settings
        self.supports_volume_discounts = True
        self.supports_enterprise_pricing = True
        self.supports_time_based_pricing = False
        self.minimum_charge = Decimal("0.0001")  # Minimum charge per request
    
    def add_model(self, model_pricing: ModelPricingInfo) -> None:
        """Add model pricing information."""
        self.models[model_pricing.model_id] = model_pricing
        logger.info(f"Added pricing for {self.provider.value}:{model_pricing.model_id}")
    
    def get_model_pricing(self, model_id: str) -> Optional[ModelPricingInfo]:
        """Get pricing info for a model."""
        return self.models.get(model_id)
    
    async def update_from_api(self) -> bool:
        """Update pricing from provider API (if available)."""
        if not self.api_endpoint:
            return False
        
        try:
            headers = {}
            if self.api_key:
                headers['Authorization'] = f'Bearer {self.api_key}'
            
            async with aiohttp.ClientSession() as session:
                async with session.get(self.api_endpoint, headers=headers) as response:
                    if response.status == 200:
                        pricing_data = await response.json()
                        self._process_api_pricing_data(pricing_data)
                        self.last_api_update = datetime.utcnow()
                        logger.info(f"Updated pricing from API for {self.provider.value}")
                        return True
                    else:
                        logger.warning(f"API pricing update failed for {self.provider.value}: {response.status}")
                        return False
        
        except Exception as e:
            logger.error(f"Error updating pricing from API for {self.provider.value}: {e}")
            return False
    
    def _process_api_pricing_data(self, data: Dict[str, Any]) -> None:
        """Process pricing data from API response."""
        # This would be implemented based on each provider's API format
        # For now, this is a placeholder
        logger.debug(f"Processing API pricing data for {self.provider.value}")


class PricingEngine:
    """Main pricing engine coordinating all providers."""
    
    def __init__(self):
        self.provider_configs = {}  # ProviderType -> ProviderPricingConfig
        self.currency_rates = {'USD': Decimal('1.0')}  # Base currency rates
        self.pricing_cache = {}  # Cache for calculated prices
        self.cache_ttl = 300  # 5 minutes cache TTL
        
        # Initialize with default pricing
        self._initialize_default_pricing()
        
        logger.info("Pricing Engine initialized")
    
    def _initialize_default_pricing(self) -> None:
        """Initialize with default pricing for all providers."""
        
        # OpenAI Pricing
        openai_config = ProviderPricingConfig(ProviderType.OPENAI)
        
        # GPT-4 Turbo
        gpt4_turbo = ModelPricingInfo("gpt-4-turbo", ProviderType.OPENAI)
        gpt4_turbo.set_base_price(CostComponent.INPUT_TOKENS, Decimal("0.01"))
        gpt4_turbo.set_base_price(CostComponent.OUTPUT_TOKENS, Decimal("0.03"))
        gpt4_turbo.add_volume_discount(Decimal("1000"), 0.95)  # 5% discount after $1000
        gpt4_turbo.add_volume_discount(Decimal("5000"), 0.9)   # 10% discount after $5000
        openai_config.add_model(gpt4_turbo)
        
        # GPT-4
        gpt4 = ModelPricingInfo("gpt-4", ProviderType.OPENAI)
        gpt4.set_base_price(CostComponent.INPUT_TOKENS, Decimal("0.03"))
        gpt4.set_base_price(CostComponent.OUTPUT_TOKENS, Decimal("0.06"))
        gpt4.add_volume_discount(Decimal("1000"), 0.95)
        openai_config.add_model(gpt4)
        
        # GPT-3.5 Turbo
        gpt35_turbo = ModelPricingInfo("gpt-3.5-turbo", ProviderType.OPENAI)
        gpt35_turbo.set_base_price(CostComponent.INPUT_TOKENS, Decimal("0.0015"))
        gpt35_turbo.set_base_price(CostComponent.OUTPUT_TOKENS, Decimal("0.002"))
        gpt35_turbo.add_volume_discount(Decimal("500"), 0.9)   # 10% discount after $500
        openai_config.add_model(gpt35_turbo)
        
        self.provider_configs[ProviderType.OPENAI] = openai_config
        
        # Anthropic Pricing
        anthropic_config = ProviderPricingConfig(ProviderType.ANTHROPIC)
        
        # Claude 3 Opus
        claude_opus = ModelPricingInfo("claude-3-opus-20240229", ProviderType.ANTHROPIC)
        claude_opus.set_base_price(CostComponent.INPUT_TOKENS, Decimal("0.015"))
        claude_opus.set_base_price(CostComponent.OUTPUT_TOKENS, Decimal("0.075"))
        claude_opus.add_volume_discount(Decimal("2000"), 0.92)  # 8% discount after $2000
        anthropic_config.add_model(claude_opus)
        
        # Claude 3 Sonnet
        claude_sonnet = ModelPricingInfo("claude-3-sonnet-20240229", ProviderType.ANTHROPIC)
        claude_sonnet.set_base_price(CostComponent.INPUT_TOKENS, Decimal("0.003"))
        claude_sonnet.set_base_price(CostComponent.OUTPUT_TOKENS, Decimal("0.015"))
        claude_sonnet.add_volume_discount(Decimal("1000"), 0.95)
        anthropic_config.add_model(claude_sonnet)
        
        # Claude 3 Haiku
        claude_haiku = ModelPricingInfo("claude-3-haiku-20240307", ProviderType.ANTHROPIC)
        claude_haiku.set_base_price(CostComponent.INPUT_TOKENS, Decimal("0.00025"))
        claude_haiku.set_base_price(CostComponent.OUTPUT_TOKENS, Decimal("0.00125"))
        claude_haiku.add_volume_discount(Decimal("500"), 0.9)
        anthropic_config.add_model(claude_haiku)
        
        self.provider_configs[ProviderType.ANTHROPIC] = anthropic_config
        
        # Google Pricing
        google_config = ProviderPricingConfig(ProviderType.GOOGLE)
        
        # Gemini Pro
        gemini_pro = ModelPricingInfo("gemini-pro", ProviderType.GOOGLE)
        gemini_pro.set_base_price(CostComponent.INPUT_TOKENS, Decimal("0.0005"))
        gemini_pro.set_base_price(CostComponent.OUTPUT_TOKENS, Decimal("0.0015"))
        gemini_pro.add_volume_discount(Decimal("1000"), 0.9)   # 10% discount after $1000
        google_config.add_model(gemini_pro)
        
        # Gemini Pro Vision
        gemini_pro_vision = ModelPricingInfo("gemini-pro-vision", ProviderType.GOOGLE)
        gemini_pro_vision.set_base_price(CostComponent.INPUT_TOKENS, Decimal("0.00025"))
        gemini_pro_vision.set_base_price(CostComponent.OUTPUT_TOKENS, Decimal("0.0005"))
        gemini_pro_vision.set_base_price(CostComponent.IMAGE_PROCESSING, Decimal("0.0025"))  # per image
        google_config.add_model(gemini_pro_vision)
        
        self.provider_configs[ProviderType.GOOGLE] = google_config
    
    def register_provider_config(self, config: ProviderPricingConfig) -> None:
        """Register a provider pricing configuration."""
        self.provider_configs[config.provider] = config
        logger.info(f"Registered pricing config for {config.provider.value}")
    
    def calculate_cost(
        self,
        provider: ProviderType,
        model_id: str,
        usage_amounts: Dict[CostComponent, int],
        tier: PricingTier = PricingTier.PRO,
        monthly_usage: Decimal = Decimal("0"),
        request_time: Optional[datetime] = None,
        region: str = "us",
        currency: str = "USD"
    ) -> Optional[Dict[str, Any]]:
        """Calculate cost for a specific request."""
        
        # Create cache key
        cache_key = self._create_cache_key(
            provider, model_id, usage_amounts, tier, monthly_usage, request_time, region, currency
        )
        
        # Check cache
        if cache_key in self.pricing_cache:
            cached_result, timestamp = self.pricing_cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                return cached_result
        
        # Get provider config
        if provider not in self.provider_configs:
            logger.warning(f"No pricing config for provider {provider.value}")
            return None
        
        provider_config = self.provider_configs[provider]
        model_pricing = provider_config.get_model_pricing(model_id)
        
        if not model_pricing:
            logger.warning(f"No pricing info for model {model_id} from {provider.value}")
            return None
        
        # Calculate base cost
        result = model_pricing.calculate_cost(
            usage_amounts, tier, monthly_usage, request_time, region
        )
        
        # Apply currency conversion if needed
        if currency != model_pricing.base_currency:
            conversion_rate = self._get_currency_rate(model_pricing.base_currency, currency)
            if conversion_rate:
                result['total_cost'] *= float(conversion_rate)
                result['applied_factors']['currency_conversion'] = {
                    'from': model_pricing.base_currency,
                    'to': currency,
                    'rate': float(conversion_rate)
                }
        
        # Apply minimum charge
        if result['total_cost'] < float(provider_config.minimum_charge):
            result['total_cost'] = float(provider_config.minimum_charge)
            result['applied_factors']['minimum_charge'] = float(provider_config.minimum_charge)
        
        # Cache result
        self.pricing_cache[cache_key] = (result, time.time())
        
        return result
    
    def _create_cache_key(
        self,
        provider: ProviderType,
        model_id: str,
        usage_amounts: Dict[CostComponent, int],
        tier: PricingTier,
        monthly_usage: Decimal,
        request_time: Optional[datetime],
        region: str,
        currency: str
    ) -> str:
        """Create cache key for pricing calculation."""
        
        # Round request_time to nearest hour for better caching
        if request_time:
            cache_time = request_time.replace(minute=0, second=0, microsecond=0)
        else:
            cache_time = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
        
        cache_data = {
            'provider': provider.value,
            'model': model_id,
            'usage': {comp.value: amount for comp, amount in usage_amounts.items()},
            'tier': tier.value,
            'monthly_usage': str(monthly_usage),
            'time': cache_time.isoformat(),
            'region': region,
            'currency': currency
        }
        
        return str(hash(json.dumps(cache_data, sort_keys=True)))
    
    def _get_currency_rate(self, from_currency: str, to_currency: str) -> Optional[Decimal]:
        """Get currency conversion rate."""
        if from_currency == to_currency:
            return Decimal('1.0')
        
        # This would typically fetch from a currency API
        # For now, return None if conversion not available
        return self.currency_rates.get(to_currency)
    
    async def update_all_pricing(self) -> None:
        """Update pricing from all provider APIs."""
        tasks = []
        for provider_config in self.provider_configs.values():
            if provider_config.api_endpoint:
                tasks.append(provider_config.update_from_api())
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            successful_updates = sum(1 for result in results if result is True)
            logger.info(f"Updated pricing for {successful_updates} providers")
    
    def get_cost_comparison(
        self,
        usage_amounts: Dict[CostComponent, int],
        tier: PricingTier = PricingTier.PRO,
        monthly_usage: Decimal = Decimal("0")
    ) -> Dict[str, Any]:
        """Compare costs across all providers and models."""
        
        comparison = {
            'providers': {},
            'cheapest_option': None,
            'most_expensive_option': None,
            'cost_range': {},
            'analysis_timestamp': datetime.utcnow().isoformat()
        }
        
        all_costs = []
        
        for provider, config in self.provider_configs.items():
            provider_results = {}
            
            for model_id, model_pricing in config.models.items():
                cost_result = self.calculate_cost(
                    provider, model_id, usage_amounts, tier, monthly_usage
                )
                
                if cost_result:
                    provider_results[model_id] = cost_result
                    all_costs.append({
                        'provider': provider.value,
                        'model': model_id,
                        'cost': cost_result['total_cost'],
                        'details': cost_result
                    })
            
            if provider_results:
                # Find cheapest model for this provider
                cheapest_model = min(provider_results.items(), key=lambda x: x[1]['total_cost'])
                provider_results['cheapest_model'] = {
                    'model': cheapest_model[0],
                    'cost': cheapest_model[1]['total_cost']
                }
                
                comparison['providers'][provider.value] = provider_results
        
        # Global analysis
        if all_costs:
            all_costs.sort(key=lambda x: x['cost'])
            
            comparison['cheapest_option'] = all_costs[0]
            comparison['most_expensive_option'] = all_costs[-1]
            comparison['cost_range'] = {
                'min': all_costs[0]['cost'],
                'max': all_costs[-1]['cost'],
                'difference': all_costs[-1]['cost'] - all_costs[0]['cost'],
                'savings_percentage': ((all_costs[-1]['cost'] - all_costs[0]['cost']) / all_costs[-1]['cost'] * 100) if all_costs[-1]['cost'] > 0 else 0
            }
        
        return comparison
    
    def get_volume_discount_recommendations(
        self,
        provider: ProviderType,
        model_id: str,
        current_monthly_usage: Decimal
    ) -> Dict[str, Any]:
        """Get recommendations for achieving volume discounts."""
        
        if provider not in self.provider_configs:
            return {'error': 'Provider not configured'}
        
        model_pricing = self.provider_configs[provider].get_model_pricing(model_id)
        if not model_pricing:
            return {'error': 'Model not configured'}
        
        recommendations = {
            'current_monthly_usage': float(current_monthly_usage),
            'current_discount_factor': model_pricing._get_volume_discount(current_monthly_usage),
            'next_discount_tiers': [],
            'potential_savings': {}
        }
        
        # Find next discount tiers
        sorted_thresholds = sorted(model_pricing.volume_discounts.keys())
        
        for threshold in sorted_thresholds:
            if threshold > current_monthly_usage:
                discount_factor = model_pricing.volume_discounts[threshold]
                additional_usage_needed = threshold - current_monthly_usage
                
                recommendations['next_discount_tiers'].append({
                    'threshold': float(threshold),
                    'discount_factor': discount_factor,
                    'discount_percentage': (1 - discount_factor) * 100,
                    'additional_usage_needed': float(additional_usage_needed),
                    'time_to_reach': self._estimate_time_to_reach(current_monthly_usage, additional_usage_needed)
                })
        
        # Calculate potential savings
        if recommendations['next_discount_tiers']:
            next_tier = recommendations['next_discount_tiers'][0]
            current_cost_per_unit = float(model_pricing.base_prices.get(CostComponent.INPUT_TOKENS, Decimal('0')))
            
            current_monthly_cost = current_cost_per_unit * float(current_monthly_usage)
            discounted_monthly_cost = current_cost_per_unit * float(current_monthly_usage) * next_tier['discount_factor']
            
            recommendations['potential_savings'] = {
                'monthly_savings': current_monthly_cost - discounted_monthly_cost,
                'annual_savings': (current_monthly_cost - discounted_monthly_cost) * 12,
                'breakeven_months': self._calculate_breakeven_months(
                    float(next_tier['additional_usage_needed']),
                    current_monthly_cost - discounted_monthly_cost
                )
            }
        
        return recommendations
    
    def _estimate_time_to_reach(self, current_usage: Decimal, additional_needed: Decimal) -> str:
        """Estimate time to reach next discount tier."""
        if current_usage == 0:
            return "Unable to estimate"
        
        # Assume linear growth rate based on current usage
        months_to_reach = float(additional_needed / current_usage)
        
        if months_to_reach < 1:
            return f"{months_to_reach * 30:.0f} days"
        elif months_to_reach < 12:
            return f"{months_to_reach:.1f} months"
        else:
            return f"{months_to_reach / 12:.1f} years"
    
    def _calculate_breakeven_months(self, additional_usage_cost: float, monthly_savings: float) -> float:
        """Calculate months to break even on increased usage."""
        if monthly_savings <= 0:
            return float('inf')
        
        return additional_usage_cost / monthly_savings
    
    def get_pricing_analytics(self) -> Dict[str, Any]:
        """Get analytics about pricing and usage patterns."""
        
        analytics = {
            'providers_configured': len(self.provider_configs),
            'models_configured': sum(len(config.models) for config in self.provider_configs.values()),
            'cache_hit_ratio': self._calculate_cache_hit_ratio(),
            'last_pricing_updates': {},
            'pricing_coverage': {}
        }
        
        # Pricing coverage analysis
        for provider, config in self.provider_configs.items():
            analytics['last_pricing_updates'][provider.value] = {
                'last_update': config.last_api_update.isoformat() if config.last_api_update else None,
                'update_frequency_hours': config.update_frequency_hours,
                'has_api_endpoint': config.api_endpoint is not None
            }
            
            # Model pricing coverage
            model_coverage = {}
            for model_id, model_pricing in config.models.items():
                components_covered = list(model_pricing.base_prices.keys())
                model_coverage[model_id] = {
                    'components_covered': [comp.value for comp in components_covered],
                    'has_volume_discounts': len(model_pricing.volume_discounts) > 0,
                    'has_time_based_pricing': len(model_pricing.peak_hours) > 0,
                    'last_updated': model_pricing.last_updated.isoformat()
                }
            
            analytics['pricing_coverage'][provider.value] = model_coverage
        
        return analytics
    
    def _calculate_cache_hit_ratio(self) -> float:
        """Calculate cache hit ratio (simplified)."""
        if not hasattr(self, '_total_requests'):
            self._total_requests = 0
            self._cache_hits = 0
        
        if self._total_requests == 0:
            return 0.0
        
        return self._cache_hits / self._total_requests
    
    def clear_pricing_cache(self) -> None:
        """Clear pricing cache."""
        self.pricing_cache.clear()
        logger.info("Cleared pricing cache")
    
    def export_pricing_config(self) -> Dict[str, Any]:
        """Export pricing configuration for backup/sharing."""
        config_export = {
            'export_timestamp': datetime.utcnow().isoformat(),
            'providers': {}
        }
        
        for provider, config in self.provider_configs.items():
            provider_data = {
                'models': {},
                'settings': {
                    'supports_volume_discounts': config.supports_volume_discounts,
                    'supports_enterprise_pricing': config.supports_enterprise_pricing,
                    'minimum_charge': float(config.minimum_charge)
                }
            }
            
            for model_id, model_pricing in config.models.items():
                model_data = {
                    'pricing_model': model_pricing.pricing_model.value,
                    'base_prices': {comp.value: float(price) for comp, price in model_pricing.base_prices.items()},
                    'volume_discounts': {str(threshold): factor for threshold, factor in model_pricing.volume_discounts.items()},
                    'tier_multipliers': {tier.value: mult for tier, mult in model_pricing.tier_multipliers.items()},
                    'last_updated': model_pricing.last_updated.isoformat()
                }
                
                provider_data['models'][model_id] = model_data
            
            config_export['providers'][provider.value] = provider_data
        
        return config_export