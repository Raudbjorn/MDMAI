"""Configuration management for AI providers."""

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field, SecretStr, validator
from structlog import get_logger

from .models import CostBudget, ProviderConfig, ProviderType

logger = get_logger(__name__)


class AIProviderSettings(BaseModel):
    """Settings for AI provider integration."""
    
    # API Keys (from environment variables or config)
    anthropic_api_key: Optional[SecretStr] = Field(None, env="ANTHROPIC_API_KEY")
    openai_api_key: Optional[SecretStr] = Field(None, env="OPENAI_API_KEY")
    google_api_key: Optional[SecretStr] = Field(None, env="GOOGLE_API_KEY")
    
    # Provider configurations
    anthropic_enabled: bool = Field(True, env="AI_ANTHROPIC_ENABLED")
    openai_enabled: bool = Field(True, env="AI_OPENAI_ENABLED")
    google_enabled: bool = Field(True, env="AI_GOOGLE_ENABLED")
    
    # Provider priorities (higher = preferred)
    anthropic_priority: int = Field(10, env="AI_ANTHROPIC_PRIORITY")
    openai_priority: int = Field(5, env="AI_OPENAI_PRIORITY")
    google_priority: int = Field(3, env="AI_GOOGLE_PRIORITY")
    
    # Rate limits (requests per minute)
    anthropic_rate_limit_rpm: int = Field(1000, env="AI_ANTHROPIC_RATE_LIMIT")
    openai_rate_limit_rpm: int = Field(500, env="AI_OPENAI_RATE_LIMIT")
    google_rate_limit_rpm: int = Field(300, env="AI_GOOGLE_RATE_LIMIT")
    
    # Budget limits (USD)
    daily_budget_limit: Optional[float] = Field(None, env="AI_DAILY_BUDGET")
    monthly_budget_limit: Optional[float] = Field(None, env="AI_MONTHLY_BUDGET")
    per_request_budget_limit: Optional[float] = Field(None, env="AI_REQUEST_BUDGET")
    
    # Budget alert thresholds (percentages as decimals)
    budget_alert_thresholds: List[float] = Field(
        default_factory=lambda: [0.5, 0.8, 0.95],
        env="AI_BUDGET_ALERTS",
    )
    
    # Provider-specific budget limits
    anthropic_daily_limit: Optional[float] = Field(None, env="AI_ANTHROPIC_DAILY_LIMIT")
    openai_daily_limit: Optional[float] = Field(None, env="AI_OPENAI_DAILY_LIMIT")
    google_daily_limit: Optional[float] = Field(None, env="AI_GOOGLE_DAILY_LIMIT")
    
    # Request settings
    default_max_tokens: int = Field(2048, env="AI_DEFAULT_MAX_TOKENS")
    default_temperature: float = Field(0.7, env="AI_DEFAULT_TEMPERATURE")
    request_timeout: float = Field(30.0, env="AI_REQUEST_TIMEOUT")
    max_retries: int = Field(3, env="AI_MAX_RETRIES")
    retry_delay: float = Field(1.0, env="AI_RETRY_DELAY")
    
    # Selection strategy
    default_selection_strategy: str = Field("cost", env="AI_SELECTION_STRATEGY")
    
    # Health monitoring
    health_check_enabled: bool = Field(True, env="AI_HEALTH_CHECK_ENABLED")
    health_check_interval: int = Field(300, env="AI_HEALTH_CHECK_INTERVAL")
    
    # Circuit breaker settings
    circuit_breaker_failure_threshold: int = Field(5, env="AI_CIRCUIT_BREAKER_THRESHOLD")
    circuit_breaker_recovery_timeout: float = Field(60.0, env="AI_CIRCUIT_BREAKER_TIMEOUT")
    
    # Streaming settings
    streaming_enabled: bool = Field(True, env="AI_STREAMING_ENABLED")
    streaming_buffer_size: int = Field(5, env="AI_STREAMING_BUFFER_SIZE")
    streaming_session_timeout: int = Field(3600, env="AI_STREAMING_TIMEOUT")
    
    # Caching settings
    cache_enabled: bool = Field(True, env="AI_CACHE_ENABLED")
    cache_ttl: int = Field(3600, env="AI_CACHE_TTL")
    cache_max_size: int = Field(1000, env="AI_CACHE_MAX_SIZE")
    
    # Logging and monitoring
    log_requests: bool = Field(False, env="AI_LOG_REQUESTS")
    log_responses: bool = Field(False, env="AI_LOG_RESPONSES")
    metrics_enabled: bool = Field(True, env="AI_METRICS_ENABLED")
    
    @validator("budget_alert_thresholds", pre=True)
    def parse_alert_thresholds(cls, v):
        """Parse alert thresholds from string if needed."""
        if isinstance(v, str):
            return [float(x) for x in v.split(",")]
        return v
    
    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        case_sensitive = False


class AIProviderConfigManager:
    """Manages AI provider configurations."""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path("config/ai_providers.yaml")
        self.settings = AIProviderSettings()
        self._provider_configs: Dict[ProviderType, ProviderConfig] = {}
        self._budgets: List[CostBudget] = []
        
        # Load configuration
        self.load_configuration()
    
    def load_configuration(self) -> None:
        """Load configuration from file and environment."""
        # Load from YAML if exists
        if self.config_path.exists():
            self._load_yaml_config()
        
        # Build provider configs
        self._build_provider_configs()
        
        # Build budget configs
        self._build_budget_configs()
        
        logger.info(
            "Loaded AI provider configuration",
            providers=len(self._provider_configs),
            budgets=len(self._budgets),
        )
    
    def _load_yaml_config(self) -> None:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, "r") as f:
                config_data = yaml.safe_load(f)
            
            # Update settings with YAML data
            if config_data:
                for key, value in config_data.items():
                    if hasattr(self.settings, key):
                        setattr(self.settings, key, value)
            
            logger.info("Loaded configuration from YAML", path=str(self.config_path))
            
        except Exception as e:
            logger.warning(
                "Failed to load YAML configuration",
                path=str(self.config_path),
                error=str(e),
            )
    
    def _build_provider_configs(self) -> None:
        """Build provider configurations from settings."""
        # Anthropic configuration
        if self.settings.anthropic_api_key and self.settings.anthropic_enabled:
            self._provider_configs[ProviderType.ANTHROPIC] = ProviderConfig(
                provider_type=ProviderType.ANTHROPIC,
                api_key=self.settings.anthropic_api_key.get_secret_value(),
                enabled=self.settings.anthropic_enabled,
                priority=self.settings.anthropic_priority,
                rate_limit_rpm=self.settings.anthropic_rate_limit_rpm,
                timeout=self.settings.request_timeout,
                max_retries=self.settings.max_retries,
                retry_delay=self.settings.retry_delay,
                budget_limit=self.settings.anthropic_daily_limit,
            )
        
        # OpenAI configuration
        if self.settings.openai_api_key and self.settings.openai_enabled:
            self._provider_configs[ProviderType.OPENAI] = ProviderConfig(
                provider_type=ProviderType.OPENAI,
                api_key=self.settings.openai_api_key.get_secret_value(),
                enabled=self.settings.openai_enabled,
                priority=self.settings.openai_priority,
                rate_limit_rpm=self.settings.openai_rate_limit_rpm,
                timeout=self.settings.request_timeout,
                max_retries=self.settings.max_retries,
                retry_delay=self.settings.retry_delay,
                budget_limit=self.settings.openai_daily_limit,
            )
        
        # Google configuration
        if self.settings.google_api_key and self.settings.google_enabled:
            self._provider_configs[ProviderType.GOOGLE] = ProviderConfig(
                provider_type=ProviderType.GOOGLE,
                api_key=self.settings.google_api_key.get_secret_value(),
                enabled=self.settings.google_enabled,
                priority=self.settings.google_priority,
                rate_limit_rpm=self.settings.google_rate_limit_rpm,
                timeout=self.settings.request_timeout,
                max_retries=self.settings.max_retries,
                retry_delay=self.settings.retry_delay,
                budget_limit=self.settings.google_daily_limit,
            )
    
    def _build_budget_configs(self) -> None:
        """Build budget configurations from settings."""
        # Main budget
        if self.settings.daily_budget_limit or self.settings.monthly_budget_limit:
            main_budget = CostBudget(
                name="Main Budget",
                daily_limit=self.settings.daily_budget_limit,
                monthly_limit=self.settings.monthly_budget_limit,
                alert_thresholds=self.settings.budget_alert_thresholds,
                enabled=True,
            )
            
            # Add provider-specific limits
            provider_limits = {}
            if self.settings.anthropic_daily_limit:
                provider_limits[ProviderType.ANTHROPIC] = self.settings.anthropic_daily_limit
            if self.settings.openai_daily_limit:
                provider_limits[ProviderType.OPENAI] = self.settings.openai_daily_limit
            if self.settings.google_daily_limit:
                provider_limits[ProviderType.GOOGLE] = self.settings.google_daily_limit
            
            if provider_limits:
                main_budget.provider_limits = provider_limits
            
            self._budgets.append(main_budget)
    
    def get_provider_configs(self) -> List[ProviderConfig]:
        """Get list of provider configurations.
        
        Returns:
            List of provider configurations
        """
        return list(self._provider_configs.values())
    
    def get_provider_config(self, provider_type: ProviderType) -> Optional[ProviderConfig]:
        """Get configuration for a specific provider.
        
        Args:
            provider_type: Provider type
            
        Returns:
            Provider configuration or None
        """
        return self._provider_configs.get(provider_type)
    
    def get_budgets(self) -> List[CostBudget]:
        """Get list of budget configurations.
        
        Returns:
            List of budget configurations
        """
        return self._budgets
    
    def update_provider_config(
        self,
        provider_type: ProviderType,
        **kwargs,
    ) -> None:
        """Update provider configuration.
        
        Args:
            provider_type: Provider to update
            **kwargs: Configuration values to update
        """
        if provider_type not in self._provider_configs:
            logger.warning("Provider not configured", provider=provider_type.value)
            return
        
        config = self._provider_configs[provider_type]
        
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        logger.info(
            "Updated provider configuration",
            provider=provider_type.value,
            updates=list(kwargs.keys()),
        )
    
    def save_configuration(self) -> None:
        """Save current configuration to YAML file."""
        config_data = {}
        
        # Save settings
        for field_name, field_value in self.settings.dict().items():
            if field_value is not None:
                # Handle SecretStr fields
                if isinstance(field_value, SecretStr):
                    continue  # Don't save secrets to file
                config_data[field_name] = field_value
        
        # Save provider-specific settings
        for provider_type, config in self._provider_configs.items():
            provider_key = f"{provider_type.value}_config"
            config_dict = config.dict(exclude={"api_key"})  # Exclude API key
            config_data[provider_key] = config_dict
        
        # Save budget settings
        if self._budgets:
            config_data["budgets"] = [
                budget.dict() for budget in self._budgets
            ]
        
        # Write to file
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, "w") as f:
                yaml.dump(config_data, f, default_flow_style=False)
            
            logger.info("Saved configuration to file", path=str(self.config_path))
            
        except Exception as e:
            logger.error(
                "Failed to save configuration",
                path=str(self.config_path),
                error=str(e),
            )
    
    def validate_configuration(self) -> Dict[str, Any]:
        """Validate the configuration and return issues.
        
        Returns:
            Dictionary with validation results
        """
        issues = []
        warnings = []
        
        # Check if at least one provider is configured
        if not self._provider_configs:
            issues.append("No AI providers configured with API keys")
        
        # Check each provider
        for provider_type, config in self._provider_configs.items():
            provider_name = provider_type.value
            
            # Validate API key
            if not config.api_key:
                issues.append(f"{provider_name}: Missing API key")
            
            # Check rate limits
            if config.rate_limit_rpm <= 0:
                warnings.append(f"{provider_name}: Invalid rate limit")
            
            # Check timeouts
            if config.timeout <= 0:
                warnings.append(f"{provider_name}: Invalid timeout")
        
        # Check budgets
        if self._budgets:
            for budget in self._budgets:
                if budget.daily_limit and budget.daily_limit <= 0:
                    warnings.append(f"Budget '{budget.name}': Invalid daily limit")
                if budget.monthly_limit and budget.monthly_limit <= 0:
                    warnings.append(f"Budget '{budget.name}': Invalid monthly limit")
        
        # Check selection strategy
        valid_strategies = ["cost", "priority", "capability", "load_balanced", "failover", "random", "round_robin"]
        if self.settings.default_selection_strategy not in valid_strategies:
            warnings.append(f"Invalid selection strategy: {self.settings.default_selection_strategy}")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "providers_configured": len(self._provider_configs),
            "budgets_configured": len(self._budgets),
        }
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get a summary of the current configuration.
        
        Returns:
            Configuration summary dictionary
        """
        return {
            "providers": {
                provider_type.value: {
                    "enabled": config.enabled,
                    "priority": config.priority,
                    "rate_limit": config.rate_limit_rpm,
                    "has_api_key": bool(config.api_key),
                }
                for provider_type, config in self._provider_configs.items()
            },
            "budgets": [
                {
                    "name": budget.name,
                    "daily_limit": budget.daily_limit,
                    "monthly_limit": budget.monthly_limit,
                    "enabled": budget.enabled,
                }
                for budget in self._budgets
            ],
            "settings": {
                "default_strategy": self.settings.default_selection_strategy,
                "health_check_enabled": self.settings.health_check_enabled,
                "streaming_enabled": self.settings.streaming_enabled,
                "cache_enabled": self.settings.cache_enabled,
                "metrics_enabled": self.settings.metrics_enabled,
            },
        }