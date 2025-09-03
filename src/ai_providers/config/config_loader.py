"""
Configuration Loader for Model Router System

This module provides functionality to load model profiles and routing rules
from YAML configuration files, enabling externalized configuration management.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

from structlog import get_logger
from ..models import ProviderType, CostTier, ProviderCapability

logger = get_logger(__name__)


class ConfigurationError(Exception):
    """Raised when configuration loading fails."""
    pass


class ConfigLoader:
    """Loads and validates external configuration files for the model router."""
    
    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize configuration loader.
        
        Args:
            config_dir: Directory containing configuration files.
                       Defaults to the config directory in this package.
        """
        if config_dir is None:
            config_dir = Path(__file__).parent
        
        self.config_dir = Path(config_dir)
        self._validate_config_directory()
    
    def _validate_config_directory(self) -> None:
        """Validate that the configuration directory exists and is accessible."""
        if not self.config_dir.exists():
            raise ConfigurationError(
                f"Configuration directory does not exist: {self.config_dir}"
            )
        
        if not self.config_dir.is_dir():
            raise ConfigurationError(
                f"Configuration path is not a directory: {self.config_dir}"
            )
        
        logger.info("Configuration directory validated", config_dir=str(self.config_dir))
    
    def load_yaml_file(self, filename: str) -> Dict[str, Any]:
        """
        Load and parse a YAML configuration file.
        
        Args:
            filename: Name of the YAML file to load
            
        Returns:
            Parsed YAML content as dictionary
            
        Raises:
            ConfigurationError: If file cannot be loaded or parsed
        """
        file_path = self.config_dir / filename
        
        try:
            if not file_path.exists():
                raise ConfigurationError(f"Configuration file not found: {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as file:
                content = yaml.safe_load(file)
                
            if content is None:
                raise ConfigurationError(f"Configuration file is empty: {file_path}")
                
            logger.info("Configuration file loaded successfully", 
                       file=filename, entries=len(content))
            return content
            
        except yaml.YAMLError as e:
            raise ConfigurationError(f"YAML parsing error in {file_path}: {e}")
        except IOError as e:
            raise ConfigurationError(f"IO error reading {file_path}: {e}")
    
    def load_model_profiles(self, filename: str = "default_model_profiles.yaml") -> Dict[str, Any]:
        """
        Load model profiles from configuration file.
        
        Args:
            filename: Name of the model profiles configuration file
            
        Returns:
            Dictionary containing model profile configurations
        """
        try:
            config = self.load_yaml_file(filename)
            
            if 'model_profiles' not in config:
                raise ConfigurationError(
                    f"Missing 'model_profiles' section in {filename}"
                )
            
            profiles = config['model_profiles']
            self._validate_model_profiles(profiles)
            
            logger.info("Model profiles loaded", 
                       profile_count=len(profiles),
                       models=list(profiles.keys()))
            
            return profiles
            
        except Exception as e:
            logger.error("Failed to load model profiles", 
                        filename=filename, error=str(e))
            raise ConfigurationError(f"Failed to load model profiles: {e}")
    
    def load_routing_rules(self, filename: str = "default_routing_rules.yaml") -> List[Dict[str, Any]]:
        """
        Load routing rules from configuration file.
        
        Args:
            filename: Name of the routing rules configuration file
            
        Returns:
            List of routing rule configurations
        """
        try:
            config = self.load_yaml_file(filename)
            
            if 'routing_rules' not in config:
                raise ConfigurationError(
                    f"Missing 'routing_rules' section in {filename}"
                )
            
            rules = config['routing_rules']
            self._validate_routing_rules(rules)
            
            logger.info("Routing rules loaded", 
                       rule_count=len(rules),
                       rule_names=[rule.get('name', 'unnamed') for rule in rules])
            
            return rules
            
        except Exception as e:
            logger.error("Failed to load routing rules", 
                        filename=filename, error=str(e))
            raise ConfigurationError(f"Failed to load routing rules: {e}")
    
    def _validate_model_profiles(self, profiles: Dict[str, Any]) -> None:
        """
        Validate model profile configurations.
        
        Args:
            profiles: Dictionary of model profiles to validate
        """
        required_fields = [
            'provider_type', 'category', 'tier', 'context_length',
            'max_output_tokens', 'supports_streaming', 'supports_tools',
            'supports_vision', 'capabilities', 'cost_tier', 'recommended_for'
        ]
        
        required_capabilities = [
            'quality_score', 'reasoning_capability', 'creativity_capability',
            'coding_capability', 'multimodal_capability', 'tool_use_capability'
        ]
        
        for model_id, profile in profiles.items():
            # Check required fields
            missing_fields = [field for field in required_fields if field not in profile]
            if missing_fields:
                raise ConfigurationError(
                    f"Model '{model_id}' missing required fields: {missing_fields}"
                )
            
            # Validate capabilities section
            if 'capabilities' not in profile or not isinstance(profile['capabilities'], dict):
                raise ConfigurationError(
                    f"Model '{model_id}' must have a 'capabilities' dictionary"
                )
            
            capabilities = profile['capabilities']
            missing_capabilities = [
                cap for cap in required_capabilities if cap not in capabilities
            ]
            if missing_capabilities:
                raise ConfigurationError(
                    f"Model '{model_id}' missing capability scores: {missing_capabilities}"
                )
            
            # Validate capability scores are between 0 and 1
            for cap_name, score in capabilities.items():
                if not isinstance(score, (int, float)) or not (0 <= score <= 1):
                    raise ConfigurationError(
                        f"Model '{model_id}' capability '{cap_name}' must be between 0 and 1"
                    )
    
    def _validate_routing_rules(self, rules: List[Dict[str, Any]]) -> None:
        """
        Validate routing rule configurations.
        
        Args:
            rules: List of routing rules to validate
        """
        required_fields = ['name', 'description', 'priority']
        
        for i, rule in enumerate(rules):
            # Check required fields
            missing_fields = [field for field in required_fields if field not in rule]
            if missing_fields:
                raise ConfigurationError(
                    f"Routing rule {i} missing required fields: {missing_fields}"
                )
            
            # Validate priority is a number
            if not isinstance(rule['priority'], (int, float)):
                raise ConfigurationError(
                    f"Routing rule '{rule.get('name', i)}' priority must be numeric"
                )
            
            # Validate request_patterns is a list if present
            if 'request_patterns' in rule and not isinstance(rule['request_patterns'], list):
                raise ConfigurationError(
                    f"Routing rule '{rule['name']}' request_patterns must be a list"
                )
            
            # Validate required_capabilities is a list if present
            if 'required_capabilities' in rule and not isinstance(rule['required_capabilities'], list):
                raise ConfigurationError(
                    f"Routing rule '{rule['name']}' required_capabilities must be a list"
                )


def get_default_config_loader() -> ConfigLoader:
    """
    Get a default configuration loader instance.
    
    Returns:
        ConfigLoader instance for the default configuration directory
    """
    return ConfigLoader()