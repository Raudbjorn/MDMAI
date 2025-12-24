"""
Pricing configuration management for MDMAI TTRPG Assistant.

This module provides loading, validation, and management of AI provider
pricing configurations with automatic updates and caching.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime, timedelta
import json

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    yaml = None

from .base_provider import ProviderType

logger = logging.getLogger(__name__)

# Minimum segment overlap ratio for fuzzy model matching (e.g., "gpt-4-turbo" to "gpt-4-turbo-preview")
MODEL_MATCH_OVERLAP_THRESHOLD = 0.75


def _extract_pricing(model_info: Dict[str, Any]) -> Dict[str, float]:
    """Extract pricing information from model config.

    Args:
        model_info: Model configuration dictionary

    Returns:
        Dictionary with input_price and output_price
    """
    return {
        'input_price': model_info.get('input_price', 0.0),
        'output_price': model_info.get('output_price', 0.0)
    }


class PricingConfigError(Exception):
    """Exception raised for pricing configuration errors."""
    pass


class PricingConfigManager:
    """
    Manager for AI provider pricing configuration.
    
    Features:
    - YAML configuration loading
    - Fallback to JSON if YAML not available
    - Configuration validation
    - Automatic updates
    - Cost calculation helpers
    - Spending recommendations
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize pricing configuration manager.
        
        Args:
            config_path: Path to configuration file
        """
        if config_path:
            self.config_path = Path(config_path)
        else:
            # Default to config/pricing.yaml in project root
            self.config_path = Path(__file__).parent.parent.parent / 'config' / 'pricing.yaml'
        
        self.config: Dict[str, Any] = {}
        self.last_loaded: Optional[datetime] = None
        self.cache_duration = timedelta(hours=1)  # Cache config for 1 hour
        
        # Load initial configuration
        self.load_config()
        
        logger.info(f"PricingConfigManager initialized with config from {self.config_path}")
    
    def load_config(self, force_reload: bool = False) -> bool:
        """
        Load pricing configuration from file.
        
        Args:
            force_reload: Force reload even if cached
            
        Returns:
            bool: True if configuration was loaded successfully
            
        Raises:
            PricingConfigError: If configuration cannot be loaded or is invalid
        """
        # Check if we need to reload
        if (not force_reload and 
            self.last_loaded and 
            datetime.now() - self.last_loaded < self.cache_duration and
            self.config):
            return True
        
        try:
            if not self.config_path.exists():
                logger.warning(f"Pricing config file not found: {self.config_path}")
                self.config = self._get_default_config()
                self._save_default_config()
                return True
            
            # Load configuration
            with open(self.config_path, 'r', encoding='utf-8') as f:
                if YAML_AVAILABLE and self.config_path.suffix.lower() in ['.yaml', '.yml']:
                    self.config = yaml.safe_load(f)
                else:
                    self.config = json.load(f)
            
            # Validate configuration
            self._validate_config()
            
            self.last_loaded = datetime.now()
            logger.info("Pricing configuration loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load pricing configuration: {e}")
            if not self.config:  # Fallback to default if no config loaded
                self.config = self._get_default_config()
            raise PricingConfigError(f"Failed to load pricing configuration: {e}")
    
    def get_model_pricing(
        self, 
        provider: ProviderType, 
        model: str
    ) -> Optional[Dict[str, float]]:
        """
        Get pricing information for a specific model.
        
        Args:
            provider: AI provider
            model: Model name
            
        Returns:
            Optional[Dict[str, float]]: Pricing info with 'input_price' and 'output_price'
        """
        provider_config = self.config.get('providers', {}).get(provider.value, {})
        models_config = provider_config.get('models', {})
        
        # Try exact match first
        if model in models_config:
            return _extract_pricing(models_config[model])

        # Try partial matches with improved logic to avoid false positives
        # 1. Try normalized exact match (handle case and hyphens)
        normalized_model = model.lower().replace('_', '-')
        for config_model, model_info in models_config.items():
            normalized_config = config_model.lower().replace('_', '-')
            if normalized_model == normalized_config:
                return _extract_pricing(model_info)

        # 2. Try prefix match for versioned models with strict boundaries
        # Only match if one is a true prefix of the other with version/date suffix
        for config_model, model_info in models_config.items():
            normalized_config = config_model.lower().replace('_', '-')

            # Check for versioned model patterns (e.g., model-20240307 or model-v2)
            # The prefix should be followed by a version separator like -, _, or digit
            if normalized_model.startswith(normalized_config):
                # Check if the next character after the prefix is a version separator
                if len(normalized_model) > len(normalized_config):
                    next_char = normalized_model[len(normalized_config)]
                    if next_char in ['-', '_'] or next_char.isdigit():
                        return _extract_pricing(model_info)
            elif normalized_config.startswith(normalized_model):
                # Check the reverse case
                if len(normalized_config) > len(normalized_model):
                    next_char = normalized_config[len(normalized_model)]
                    if next_char in ['-', '_'] or next_char.isdigit():
                        return _extract_pricing(model_info)

        # 3. Try word-boundary based substring match (safer than general substring)
        # Only match complete words/segments separated by common delimiters
        for config_model, model_info in models_config.items():
            normalized_config = config_model.lower().replace('_', '-')

            # Split both strings into segments
            model_segments = set(normalized_model.split('-'))
            config_segments = set(normalized_config.split('-'))

            # Check if all config segments are present in model segments (subset match)
            # This helps match "gpt-4-turbo" to "gpt-4-turbo-preview" but not "gpt-4" to "gpt-4o"
            if len(config_segments) >= 2 and config_segments.issubset(model_segments):
                # Additional check: ensure significant overlap
                overlap_ratio = len(config_segments) / len(model_segments)
                if overlap_ratio >= MODEL_MATCH_OVERLAP_THRESHOLD:
                    return _extract_pricing(model_info)
        
        logger.warning(f"No pricing found for {provider.value} model {model}")
        return None
    
    def calculate_cost(
        self, 
        provider: ProviderType, 
        model: str, 
        input_tokens: int, 
        output_tokens: int
    ) -> float:
        """
        Calculate cost for API usage.
        
        Args:
            provider: AI provider
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            
        Returns:
            float: Cost in USD
        """
        pricing = self.get_model_pricing(provider, model)
        if not pricing:
            return 0.0
        
        per_tokens = self.config.get('per_tokens', 1_000_000)
        
        input_cost = (input_tokens / per_tokens) * pricing['input_price']
        output_cost = (output_tokens / per_tokens) * pricing['output_price']
        
        return input_cost + output_cost
    
    def get_recommended_models(
        self, 
        use_case: str,
        budget_tier: str = "regular"
    ) -> List[Dict[str, Any]]:
        """
        Get recommended models for a specific use case.
        
        Args:
            use_case: Use case (e.g., 'quick_responses', 'main_gameplay')
            budget_tier: Budget tier (e.g., 'casual', 'regular', 'power')
            
        Returns:
            List[Dict[str, Any]]: List of recommended models with details
        """
        optimization = self.config.get('optimization', {})
        use_case_config = optimization.get('ttrpg_use_cases', {}).get(use_case, {})
        
        if 'recommended_models' not in use_case_config:
            logger.warning(f"No recommendations found for use case: {use_case}")
            return []
        
        recommendations = []
        for model_rec in use_case_config['recommended_models']:
            provider_str = model_rec.get('provider', '')
            model_name = model_rec.get('model', '')
            
            try:
                provider = ProviderType(provider_str)
                pricing = self.get_model_pricing(provider, model_name)
                
                recommendations.append({
                    'provider': provider_str,
                    'model': model_name,
                    'reason': model_rec.get('reason', ''),
                    'pricing': pricing,
                    'cost_per_1k_tokens': self._calculate_avg_cost_per_1k(pricing) if pricing else 0.0
                })
            except ValueError:
                logger.warning(f"Invalid provider in recommendation: {provider_str}")
                continue
        
        # Sort by cost if budget tier is specified
        if budget_tier in ['casual', 'regular']:
            recommendations.sort(key=lambda x: x['cost_per_1k_tokens'])
        
        return recommendations
    
    def get_spending_guidelines(self, user_tier: str = "regular") -> Dict[str, float]:
        """
        Get spending guidelines for a user tier.
        
        Args:
            user_tier: User tier (casual, regular, power_user, dm_heavy_usage)
            
        Returns:
            Dict[str, float]: Spending guidelines
        """
        guidelines = self.config.get('spending_guidelines', {})
        
        if user_tier not in guidelines:
            logger.warning(f"No spending guidelines for tier: {user_tier}")
            user_tier = "regular"  # Fallback to regular
        
        return guidelines.get(user_tier, {
            'daily_budget': 3.0,
            'weekly_budget': 15.0,
            'monthly_budget': 50.0
        })
    
    def get_cost_alerts_config(self) -> Dict[str, Any]:
        """
        Get cost alerts configuration.
        
        Returns:
            Dict[str, Any]: Cost alerts configuration
        """
        return self.config.get('cost_alerts', {
            'warning_threshold': 0.80,
            'critical_threshold': 0.95,
            'daily_limits': {'regular': 3.0},
            'weekly_limits': {'regular': 15.0},
            'monthly_limits': {'regular': 50.0}
        })
    
    def get_rate_limits(self, provider: ProviderType, tier: str = "tier_1") -> int:
        """
        Get rate limits for a provider and tier.
        
        Args:
            provider: AI provider
            tier: Rate limit tier
            
        Returns:
            int: Requests per minute limit
        """
        rate_limits = self.config.get('rate_limits', {})
        provider_limits = rate_limits.get(provider.value, {})
        
        return provider_limits.get(tier, 60)  # Default to 60 RPM
    
    def get_provider_info(self, provider: ProviderType) -> Dict[str, Any]:
        """
        Get detailed information about a provider.
        
        Args:
            provider: AI provider
            
        Returns:
            Dict[str, Any]: Provider information
        """
        providers = self.config.get('providers', {})
        return providers.get(provider.value, {})
    
    def get_model_info(self, provider: ProviderType, model: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific model.
        
        Args:
            provider: AI provider
            model: Model name
            
        Returns:
            Dict[str, Any]: Model information
        """
        provider_info = self.get_provider_info(provider)
        models = provider_info.get('models', {})
        
        # Try exact match first
        if model in models:
            return models[model]
        
        # Try improved partial match logic
        # 1. Normalized exact match
        normalized_model = model.lower().replace('_', '-')
        for config_model, model_info in models.items():
            normalized_config = config_model.lower().replace('_', '-')
            if normalized_model == normalized_config:
                return model_info
        
        # 2. Prefix match for versioned models
        for config_model, model_info in models.items():
            normalized_config = config_model.lower().replace('_', '-')
            if normalized_model.startswith(normalized_config) or normalized_config.startswith(normalized_model):
                if len(normalized_config) >= 3 and len(normalized_model) >= 3:
                    return model_info
        
        # 3. Substring match as last resort
        for config_model, model_info in models.items():
            normalized_config = config_model.lower().replace('_', '-')
            if len(normalized_config) >= 6:
                if normalized_config in normalized_model or normalized_model in normalized_config:
                    return model_info
        
        return {}
    
    def is_model_deprecated(self, provider: ProviderType, model: str) -> bool:
        """
        Check if a model is deprecated.
        
        Args:
            provider: AI provider
            model: Model name
            
        Returns:
            bool: True if model is deprecated
        """
        model_info = self.get_model_info(provider, model)
        return model_info.get('deprecated', False)
    
    def get_context_window(self, provider: ProviderType, model: str) -> int:
        """
        Get context window size for a model.
        
        Args:
            provider: AI provider
            model: Model name
            
        Returns:
            int: Context window size in tokens
        """
        model_info = self.get_model_info(provider, model)
        return model_info.get('context_window', 4096)  # Default fallback
    
    def _calculate_avg_cost_per_1k(self, pricing: Dict[str, float]) -> float:
        """Calculate average cost per 1k tokens (assuming equal input/output)."""
        if not pricing:
            return 0.0
        
        per_tokens = self.config.get('per_tokens', 1_000_000)
        input_cost_per_1k = (1000 / per_tokens) * pricing['input_price']
        output_cost_per_1k = (1000 / per_tokens) * pricing['output_price']
        
        # Average assuming 50/50 input/output split
        return (input_cost_per_1k + output_cost_per_1k) / 2
    
    def _validate_config(self):
        """Validate the loaded configuration structure."""
        required_keys = ['providers', 'per_tokens', 'currency']
        
        for key in required_keys:
            if key not in self.config:
                raise PricingConfigError(f"Missing required key in config: {key}")
        
        # Validate providers structure
        providers = self.config.get('providers', {})
        for provider_name, provider_config in providers.items():
            if 'models' not in provider_config:
                logger.warning(f"Provider {provider_name} has no models defined")
                continue
            
            for model_name, model_config in provider_config['models'].items():
                if not isinstance(model_config, dict):
                    continue
                
                required_pricing = ['input_price', 'output_price']
                for pricing_key in required_pricing:
                    if pricing_key not in model_config:
                        logger.warning(f"Model {provider_name}/{model_name} missing {pricing_key}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration if none exists."""
        return {
            'updated_at': datetime.now().isoformat(),
            'currency': 'USD',
            'per_tokens': 1_000_000,
            'version': '1.0',
            'providers': {
                'anthropic': {
                    'models': {
                        'claude-3-haiku': {'input_price': 0.25, 'output_price': 1.25},
                        'claude-3-5-sonnet': {'input_price': 3.0, 'output_price': 15.0},
                        'claude-3-opus': {'input_price': 15.0, 'output_price': 75.0}
                    }
                },
                'openai': {
                    'models': {
                        'gpt-4o-mini': {'input_price': 0.15, 'output_price': 0.60},
                        'gpt-4o': {'input_price': 2.50, 'output_price': 10.00},
                        'gpt-4-turbo': {'input_price': 10.00, 'output_price': 30.00}
                    }
                },
                'google': {
                    'models': {
                        'gemini-1.5-flash': {'input_price': 0.075, 'output_price': 0.30},
                        'gemini-1.5-pro': {'input_price': 1.25, 'output_price': 5.00}
                    }
                }
            },
            'spending_guidelines': {
                'casual': {'daily_budget': 1.0, 'weekly_budget': 5.0, 'monthly_budget': 15.0},
                'regular': {'daily_budget': 3.0, 'weekly_budget': 15.0, 'monthly_budget': 50.0},
                'power_user': {'daily_budget': 10.0, 'weekly_budget': 50.0, 'monthly_budget': 150.0}
            },
            'cost_alerts': {
                'warning_threshold': 0.80,
                'critical_threshold': 0.95
            }
        }
    
    def _save_default_config(self):
        """Save default configuration to file."""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.config_path, 'w', encoding='utf-8') as f:
                if YAML_AVAILABLE and self.config_path.suffix.lower() in ['.yaml', '.yml']:
                    yaml.dump(self.config, f, default_flow_style=False, indent=2)
                else:
                    json.dump(self.config, f, indent=2)
            
            logger.info(f"Saved default pricing configuration to {self.config_path}")
            
        except Exception as e:
            logger.error(f"Failed to save default configuration: {e}")
    
    def update_model_pricing(
        self, 
        provider: ProviderType, 
        model: str, 
        input_price: float, 
        output_price: float,
        save_immediately: bool = True
    ) -> bool:
        """
        Update pricing for a specific model.
        
        Args:
            provider: AI provider
            model: Model name
            input_price: New input price per million tokens
            output_price: New output price per million tokens
            save_immediately: Whether to save configuration immediately
            
        Returns:
            bool: True if update was successful
        """
        try:
            # Navigate to the model configuration
            if 'providers' not in self.config:
                self.config['providers'] = {}
            
            if provider.value not in self.config['providers']:
                self.config['providers'][provider.value] = {'models': {}}
            
            if 'models' not in self.config['providers'][provider.value]:
                self.config['providers'][provider.value]['models'] = {}
            
            # Update pricing
            self.config['providers'][provider.value]['models'][model] = {
                **self.config['providers'][provider.value]['models'].get(model, {}),
                'input_price': input_price,
                'output_price': output_price
            }
            
            # Update timestamp
            self.config['updated_at'] = datetime.now().isoformat()
            
            if save_immediately:
                self._save_config()
            
            logger.info(f"Updated pricing for {provider.value}/{model}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update pricing for {provider.value}/{model}: {e}")
            return False
    
    def _save_config(self):
        """Save current configuration to file."""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                if YAML_AVAILABLE and self.config_path.suffix.lower() in ['.yaml', '.yml']:
                    yaml.dump(self.config, f, default_flow_style=False, indent=2)
                else:
                    json.dump(self.config, f, indent=2)
            
            logger.info("Pricing configuration saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save pricing configuration: {e}")
            raise PricingConfigError(f"Failed to save configuration: {e}")


# Global instance for easy access
_pricing_manager: Optional[PricingConfigManager] = None


def get_pricing_manager() -> PricingConfigManager:
    """Get the global pricing configuration manager instance."""
    global _pricing_manager
    if _pricing_manager is None:
        _pricing_manager = PricingConfigManager()
    return _pricing_manager