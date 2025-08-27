#!/usr/bin/env python3
"""
Interactive configuration wizard for TTRPG Assistant MCP Server.
"""

import os
import sys
import json
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
import getpass
import secrets
import string
import argparse
from enum import Enum


class ConfigMode(Enum):
    """Configuration modes."""
    BASIC = "basic"
    ADVANCED = "advanced"
    EXPERT = "expert"


class ConfigurationWizard:
    """Interactive configuration wizard."""
    
    def __init__(self, config_dir: Path = None):
        """Initialize configuration wizard.
        
        Args:
            config_dir: Configuration directory path
        """
        self.config_dir = config_dir or Path('/etc/ttrpg-assistant')
        self.config: Dict[str, Any] = {}
        self.mode = ConfigMode.BASIC
        
    def prompt(self, message: str, default: Optional[str] = None, 
               secret: bool = False, validate=None) -> str:
        """Prompt user for input with optional validation.
        
        Args:
            message: Prompt message
            default: Default value
            secret: Hide input (for passwords)
            validate: Validation function
            
        Returns:
            User input or default value
        """
        if default:
            prompt_text = f"{message} [{default}]: "
        else:
            prompt_text = f"{message}: "
            
        while True:
            if secret:
                value = getpass.getpass(prompt_text) or default
            else:
                value = input(prompt_text) or default
                
            if validate:
                is_valid, error_msg = validate(value)
                if not is_valid:
                    print(f"  ⚠ {error_msg}")
                    continue
                    
            return value if value else default
            
    def prompt_bool(self, message: str, default: bool = False) -> bool:
        """Prompt for boolean value.
        
        Args:
            message: Prompt message
            default: Default value
            
        Returns:
            Boolean value
        """
        default_str = 'y' if default else 'n'
        choices = '[Y/n]' if default else '[y/N]'
        
        response = self.prompt(f"{message} {choices}", default_str).lower()
        return response.startswith('y')
        
    def prompt_choice(self, message: str, choices: list, default: Optional[str] = None) -> str:
        """Prompt for choice from list.
        
        Args:
            message: Prompt message
            choices: List of choices
            default: Default choice
            
        Returns:
            Selected choice
        """
        print(f"\n{message}")
        for i, choice in enumerate(choices, 1):
            marker = " (default)" if choice == default else ""
            print(f"  {i}. {choice}{marker}")
            
        while True:
            response = self.prompt("Select option", str(choices.index(default) + 1) if default else None)
            try:
                index = int(response) - 1
                if 0 <= index < len(choices):
                    return choices[index]
            except (ValueError, IndexError):
                pass
            print("  ⚠ Invalid selection")
            
    def validate_path(self, path_str: str) -> tuple:
        """Validate path input.
        
        Args:
            path_str: Path string
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not path_str:
            return False, "Path cannot be empty"
            
        path = Path(path_str).expanduser()
        if not path.parent.exists():
            return False, f"Parent directory does not exist: {path.parent}"
            
        return True, None
        
    def validate_port(self, port_str: str) -> tuple:
        """Validate port number.
        
        Args:
            port_str: Port string
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            port = int(port_str)
            if 1 <= port <= 65535:
                return True, None
            return False, "Port must be between 1 and 65535"
        except ValueError:
            return False, "Port must be a number"
            
    def generate_secret_key(self, length: int = 32) -> str:
        """Generate secure secret key.
        
        Args:
            length: Key length
            
        Returns:
            Secret key string
        """
        alphabet = string.ascii_letters + string.digits + string.punctuation
        return ''.join(secrets.choice(alphabet) for _ in range(length))
        
    def configure_basic(self):
        """Configure basic settings."""
        print("\n" + "="*60)
        print("Basic Configuration")
        print("="*60)
        
        # Server name
        self.config['MCP_SERVER_NAME'] = self.prompt(
            "Server name",
            default="TTRPG Assistant"
        )
        
        # Data directory
        self.config['DATA_DIR'] = self.prompt(
            "Data directory path",
            default="/var/lib/ttrpg-assistant",
            validate=self.validate_path
        )
        
        # Log level
        self.config['LOG_LEVEL'] = self.prompt_choice(
            "Log level",
            choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
            default='INFO'
        )
        
        # Enable GPU
        self.config['ENABLE_GPU'] = self.prompt_bool(
            "Enable GPU acceleration (if available)?",
            default=False
        )
        
        if self.config['ENABLE_GPU']:
            self.config['GPU_TYPE'] = self.prompt_choice(
                "GPU type",
                choices=['cuda', 'rocm', 'auto'],
                default='auto'
            )
            
    def configure_database(self):
        """Configure database settings."""
        print("\n" + "="*60)
        print("Database Configuration")
        print("="*60)
        
        # ChromaDB settings
        self.config['CHROMA_DB_PATH'] = self.prompt(
            "ChromaDB path",
            default=f"{self.config.get('DATA_DIR', '/var/lib/ttrpg-assistant')}/chromadb",
            validate=self.validate_path
        )
        
        self.config['CHROMA_COLLECTION_PREFIX'] = self.prompt(
            "Collection name prefix",
            default="ttrpg_"
        )
        
        if self.mode in [ConfigMode.ADVANCED, ConfigMode.EXPERT]:
            self.config['CHROMA_DB_IMPL'] = self.prompt_choice(
                "ChromaDB implementation",
                choices=['duckdb+parquet', 'sqlite', 'clickhouse'],
                default='duckdb+parquet'
            )
            
    def configure_search(self):
        """Configure search settings."""
        print("\n" + "="*60)
        print("Search Configuration")
        print("="*60)
        
        self.config['ENABLE_HYBRID_SEARCH'] = self.prompt_bool(
            "Enable hybrid search (semantic + keyword)?",
            default=True
        )
        
        if self.config['ENABLE_HYBRID_SEARCH']:
            while True:
                semantic = float(self.prompt("Semantic search weight (0-1)", default="0.7"))
                keyword = float(self.prompt("Keyword search weight (0-1)", default="0.3"))
                
                if abs(semantic + keyword - 1.0) < 0.01:
                    self.config['SEMANTIC_WEIGHT'] = str(semantic)
                    self.config['KEYWORD_WEIGHT'] = str(keyword)
                    break
                print("  ⚠ Weights must sum to 1.0")
                
        self.config['DEFAULT_SEARCH_RESULTS'] = self.prompt(
            "Default number of search results",
            default="5"
        )
        
        # Embedding model
        if self.mode in [ConfigMode.ADVANCED, ConfigMode.EXPERT]:
            self.config['EMBEDDING_MODEL'] = self.prompt_choice(
                "Embedding model",
                choices=[
                    'all-MiniLM-L6-v2',
                    'all-mpnet-base-v2',
                    'all-distilroberta-v1',
                    'multi-qa-MiniLM-L6-cos-v1'
                ],
                default='all-MiniLM-L6-v2'
            )
            
    def configure_cache(self):
        """Configure caching settings."""
        print("\n" + "="*60)
        print("Cache Configuration")
        print("="*60)
        
        self.config['CACHE_DIR'] = self.prompt(
            "Cache directory",
            default=f"{self.config.get('DATA_DIR', '/var/lib/ttrpg-assistant')}/cache",
            validate=self.validate_path
        )
        
        self.config['CACHE_TTL_SECONDS'] = self.prompt(
            "Cache TTL in seconds",
            default="3600"
        )
        
        if self.mode in [ConfigMode.ADVANCED, ConfigMode.EXPERT]:
            self.config['SEARCH_CACHE_SIZE'] = self.prompt(
                "Search cache size (number of entries)",
                default="1000"
            )
            
            self.config['CACHE_MAX_MEMORY_MB'] = self.prompt(
                "Maximum cache memory (MB)",
                default="100"
            )
            
    def configure_security(self):
        """Configure security settings."""
        print("\n" + "="*60)
        print("Security Configuration")
        print("="*60)
        
        self.config['ENABLE_AUTHENTICATION'] = self.prompt_bool(
            "Enable authentication?",
            default=False
        )
        
        if self.config['ENABLE_AUTHENTICATION']:
            self.config['AUTH_SECRET_KEY'] = self.generate_secret_key()
            print(f"  ✓ Generated secret key (saved to config)")
            
            self.config['AUTH_ALGORITHM'] = self.prompt_choice(
                "Authentication algorithm",
                choices=['HS256', 'HS384', 'HS512', 'RS256'],
                default='HS256'
            )
            
        self.config['ENABLE_RATE_LIMITING'] = self.prompt_bool(
            "Enable rate limiting?",
            default=True
        )
        
        if self.config['ENABLE_RATE_LIMITING']:
            self.config['RATE_LIMIT_REQUESTS'] = self.prompt(
                "Maximum requests per minute",
                default="60"
            )
            
        self.config['ENABLE_AUDIT'] = self.prompt_bool(
            "Enable security audit logging?",
            default=True
        )
        
        if self.config['ENABLE_AUDIT']:
            self.config['AUDIT_RETENTION_DAYS'] = self.prompt(
                "Audit log retention (days)",
                default="90"
            )
            
        self.config['SESSION_TIMEOUT_MINUTES'] = self.prompt(
            "Session timeout (minutes)",
            default="60"
        )
        
    def configure_advanced(self):
        """Configure advanced settings."""
        print("\n" + "="*60)
        print("Advanced Configuration")
        print("="*60)
        
        # PDF Processing
        self.config['MAX_CHUNK_SIZE'] = self.prompt(
            "Maximum text chunk size",
            default="1000"
        )
        
        self.config['CHUNK_OVERLAP'] = self.prompt(
            "Text chunk overlap",
            default="200"
        )
        
        self.config['ENABLE_ADAPTIVE_LEARNING'] = self.prompt_bool(
            "Enable adaptive learning for PDF processing?",
            default=True
        )
        
        # Performance
        self.config['EMBEDDING_BATCH_SIZE'] = self.prompt(
            "Embedding batch size",
            default="32"
        )
        
        self.config['MAX_WORKERS'] = self.prompt(
            "Maximum worker threads",
            default="4"
        )
        
    def configure_network(self):
        """Configure network settings."""
        if self.mode != ConfigMode.EXPERT:
            return
            
        print("\n" + "="*60)
        print("Network Configuration")
        print("="*60)
        
        self.config['SERVER_HOST'] = self.prompt(
            "Server host",
            default="0.0.0.0"
        )
        
        self.config['SERVER_PORT'] = self.prompt(
            "Server port",
            default="8000",
            validate=self.validate_port
        )
        
        self.config['CORS_ENABLED'] = self.prompt_bool(
            "Enable CORS?",
            default=True
        )
        
        if self.config['CORS_ENABLED']:
            origins = self.prompt(
                "Allowed CORS origins (comma-separated)",
                default="*"
            )
            self.config['CORS_ORIGINS'] = origins
            
    def save_env_file(self):
        """Save configuration to .env file."""
        env_path = self.config_dir / '.env'
        
        # Create backup if file exists
        if env_path.exists():
            backup_path = env_path.with_suffix('.env.backup')
            import shutil
            shutil.copy2(env_path, backup_path)
            print(f"\n✓ Created backup: {backup_path}")
            
        # Write new configuration
        with open(env_path, 'w') as f:
            f.write("# TTRPG Assistant MCP Server Configuration\n")
            f.write("# Generated by configuration wizard\n\n")
            
            # Group settings by category
            categories = {
                'General': ['MCP_SERVER_NAME', 'DEBUG', 'LOG_LEVEL'],
                'Paths': ['DATA_DIR', 'CHROMA_DB_PATH', 'CACHE_DIR'],
                'Database': ['CHROMA_COLLECTION_PREFIX', 'CHROMA_DB_IMPL'],
                'Search': ['ENABLE_HYBRID_SEARCH', 'SEMANTIC_WEIGHT', 'KEYWORD_WEIGHT', 
                          'DEFAULT_SEARCH_RESULTS', 'EMBEDDING_MODEL'],
                'Cache': ['CACHE_TTL_SECONDS', 'SEARCH_CACHE_SIZE', 'CACHE_MAX_MEMORY_MB'],
                'Security': ['ENABLE_AUTHENTICATION', 'AUTH_SECRET_KEY', 'AUTH_ALGORITHM',
                            'ENABLE_RATE_LIMITING', 'RATE_LIMIT_REQUESTS', 'ENABLE_AUDIT',
                            'AUDIT_RETENTION_DAYS', 'SESSION_TIMEOUT_MINUTES'],
                'Performance': ['MAX_CHUNK_SIZE', 'CHUNK_OVERLAP', 'ENABLE_ADAPTIVE_LEARNING',
                               'EMBEDDING_BATCH_SIZE', 'MAX_WORKERS'],
                'Network': ['SERVER_HOST', 'SERVER_PORT', 'CORS_ENABLED', 'CORS_ORIGINS'],
                'GPU': ['ENABLE_GPU', 'GPU_TYPE']
            }
            
            for category, keys in categories.items():
                settings = [(k, v) for k, v in self.config.items() if k in keys]
                if settings:
                    f.write(f"# {category} Settings\n")
                    for key, value in settings:
                        # Don't expose secret keys in comments
                        if 'SECRET' not in key and 'PASSWORD' not in key:
                            f.write(f"{key}={value}\n")
                        else:
                            f.write(f"{key}={value}\n")
                    f.write("\n")
                    
        print(f"✓ Configuration saved to: {env_path}")
        
    def save_yaml_config(self):
        """Save configuration to YAML file."""
        yaml_path = self.config_dir / 'config.yaml'
        
        # Structure configuration for YAML
        yaml_config = {
            'server': {
                'name': self.config.get('MCP_SERVER_NAME'),
                'host': self.config.get('SERVER_HOST', '0.0.0.0'),
                'port': int(self.config.get('SERVER_PORT', 8000))
            },
            'database': {
                'chromadb': {
                    'path': self.config.get('CHROMA_DB_PATH'),
                    'collection_prefix': self.config.get('CHROMA_COLLECTION_PREFIX'),
                    'implementation': self.config.get('CHROMA_DB_IMPL', 'duckdb+parquet')
                }
            },
            'search': {
                'hybrid_enabled': self.config.get('ENABLE_HYBRID_SEARCH', 'true') == 'true',
                'semantic_weight': float(self.config.get('SEMANTIC_WEIGHT', 0.7)),
                'keyword_weight': float(self.config.get('KEYWORD_WEIGHT', 0.3)),
                'default_results': int(self.config.get('DEFAULT_SEARCH_RESULTS', 5)),
                'embedding_model': self.config.get('EMBEDDING_MODEL')
            },
            'cache': {
                'directory': self.config.get('CACHE_DIR'),
                'ttl_seconds': int(self.config.get('CACHE_TTL_SECONDS', 3600)),
                'max_memory_mb': int(self.config.get('CACHE_MAX_MEMORY_MB', 100))
            },
            'security': {
                'authentication_enabled': self.config.get('ENABLE_AUTHENTICATION', 'false') == 'true',
                'rate_limiting_enabled': self.config.get('ENABLE_RATE_LIMITING', 'true') == 'true',
                'audit_enabled': self.config.get('ENABLE_AUDIT', 'true') == 'true',
                'session_timeout_minutes': int(self.config.get('SESSION_TIMEOUT_MINUTES', 60))
            }
        }
        
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_config, f, default_flow_style=False, sort_keys=False)
            
        print(f"✓ YAML configuration saved to: {yaml_path}")
        
    def run(self):
        """Run the configuration wizard."""
        print("\n" + "="*60)
        print("TTRPG Assistant MCP Server - Configuration Wizard")
        print("="*60)
        
        # Select configuration mode
        self.mode = ConfigMode(self.prompt_choice(
            "Select configuration mode",
            choices=[m.value for m in ConfigMode],
            default=ConfigMode.BASIC.value
        ))
        
        print(f"\n✓ Running in {self.mode.value} mode")
        
        # Run configuration steps
        self.configure_basic()
        self.configure_database()
        self.configure_search()
        self.configure_cache()
        self.configure_security()
        
        if self.mode in [ConfigMode.ADVANCED, ConfigMode.EXPERT]:
            self.configure_advanced()
            
        if self.mode == ConfigMode.EXPERT:
            self.configure_network()
            
        # Confirm and save
        print("\n" + "="*60)
        print("Configuration Summary")
        print("="*60)
        
        for key, value in sorted(self.config.items()):
            # Hide sensitive values
            if 'SECRET' in key or 'PASSWORD' in key:
                display_value = '***hidden***'
            else:
                display_value = value
            print(f"  {key}: {display_value}")
            
        if self.prompt_bool("\nSave configuration?", default=True):
            # Create config directory if it doesn't exist
            self.config_dir.mkdir(parents=True, exist_ok=True)
            
            self.save_env_file()
            self.save_yaml_config()
            
            print("\n✓ Configuration complete!")
            print(f"  Configuration directory: {self.config_dir}")
            print("\nNext steps:")
            print("  1. Review the configuration files")
            print("  2. Start the server with your configuration")
            print("  3. Check logs for any issues")
        else:
            print("\n✗ Configuration cancelled")
            

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Interactive configuration wizard for TTRPG Assistant"
    )
    parser.add_argument(
        '--config-dir',
        type=Path,
        default=Path('/etc/ttrpg-assistant'),
        help='Configuration directory path'
    )
    parser.add_argument(
        '--load',
        type=Path,
        help='Load existing configuration file'
    )
    parser.add_argument(
        '--export',
        type=Path,
        help='Export configuration to JSON file'
    )
    
    args = parser.parse_args()
    
    wizard = ConfigurationWizard(args.config_dir)
    
    if args.load:
        # Load existing configuration
        with open(args.load) as f:
            if args.load.suffix == '.json':
                wizard.config = json.load(f)
            elif args.load.suffix == '.env':
                for line in f:
                    if '=' in line and not line.startswith('#'):
                        key, value = line.strip().split('=', 1)
                        wizard.config[key] = value
        print(f"✓ Loaded configuration from: {args.load}")
        
    wizard.run()
    
    if args.export:
        # Export configuration to JSON
        with open(args.export, 'w') as f:
            json.dump(wizard.config, f, indent=2)
        print(f"✓ Exported configuration to: {args.export}")


if __name__ == '__main__':
    main()