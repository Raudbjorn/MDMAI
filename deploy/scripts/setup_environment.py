#!/usr/bin/env python3
"""
Environment setup and validation script for TTRPG Assistant MCP Server.
"""

import os
import sys
import platform
import subprocess
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


class EnvironmentSetup:
    """Handles environment setup and validation."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize environment setup.
        
        Args:
            config_path: Path to configuration directory
        """
        self.config_path = config_path or Path("/etc/ttrpg-assistant")
        self.platform = platform.system().lower()
        self.python_version = sys.version_info
        self.errors: List[str] = []
        self.warnings: List[str] = []
        
    def check_python_version(self, min_version: str = "3.9") -> bool:
        """Check if Python version meets requirements.
        
        Args:
            min_version: Minimum Python version required
            
        Returns:
            True if version is sufficient
        """
        min_major, min_minor = map(int, min_version.split('.'))
        
        if self.python_version.major < min_major or \
           (self.python_version.major == min_major and self.python_version.minor < min_minor):
            self.errors.append(
                f"Python {min_version} or higher required, found {self.python_version.major}.{self.python_version.minor}"
            )
            return False
            
        logger.info(f"✓ Python version {self.python_version.major}.{self.python_version.minor} meets requirements")
        return True
        
    def check_system_dependencies(self) -> bool:
        """Check for required system dependencies.
        
        Returns:
            True if all dependencies are met
        """
        dependencies = {
            'all': ['git', 'curl'],
            'linux': ['gcc', 'g++', 'make'],
            'darwin': ['gcc', 'make'],
            'windows': []
        }
        
        missing = []
        for dep in dependencies.get('all', []) + dependencies.get(self.platform, []):
            if not shutil.which(dep):
                missing.append(dep)
                
        if missing:
            self.errors.append(f"Missing system dependencies: {', '.join(missing)}")
            return False
            
        logger.info("✓ All system dependencies found")
        return True
        
    def check_gpu_support(self) -> Dict[str, bool]:
        """Check for GPU support.
        
        Returns:
            Dictionary with GPU support information
        """
        gpu_info = {
            'cuda_available': False,
            'cuda_version': None,
            'rocm_available': False,
            'rocm_version': None,
            'gpu_detected': False
        }
        
        # Check for NVIDIA GPU and CUDA
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                gpu_info['gpu_detected'] = True
                logger.info(f"✓ NVIDIA GPU detected: {result.stdout.strip()}")
                
                # Check CUDA version
                cuda_result = subprocess.run(
                    ['nvcc', '--version'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if cuda_result.returncode == 0:
                    gpu_info['cuda_available'] = True
                    # Parse CUDA version from output
                    for line in cuda_result.stdout.split('\n'):
                        if 'release' in line:
                            parts = line.split('release')[-1].strip().split(',')[0]
                            gpu_info['cuda_version'] = parts
                            logger.info(f"✓ CUDA version {parts} available")
                            break
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
            
        # Check for AMD GPU and ROCm
        if self.platform == 'linux':
            try:
                result = subprocess.run(
                    ['rocm-smi', '--showproductname'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    gpu_info['gpu_detected'] = True
                    gpu_info['rocm_available'] = True
                    logger.info("✓ AMD GPU with ROCm support detected")
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass
                
        if not gpu_info['gpu_detected']:
            self.warnings.append("No GPU detected, will use CPU-only mode")
            
        return gpu_info
        
    def check_disk_space(self, required_gb: int = 10) -> bool:
        """Check available disk space.
        
        Args:
            required_gb: Required disk space in GB
            
        Returns:
            True if sufficient space available
        """
        import shutil
        
        paths_to_check = [
            Path.home(),
            Path("/var/lib") if self.platform != 'windows' else Path("C:\\"),
            Path("/opt") if self.platform != 'windows' else Path("C:\\Program Files")
        ]
        
        for path in paths_to_check:
            if path.exists():
                stat = shutil.disk_usage(str(path))
                free_gb = stat.free / (1024 ** 3)
                
                if free_gb < required_gb:
                    self.warnings.append(
                        f"Low disk space on {path}: {free_gb:.1f} GB free (recommend {required_gb} GB)"
                    )
                else:
                    logger.info(f"✓ Sufficient disk space on {path}: {free_gb:.1f} GB free")
                    
        return True
        
    def check_memory(self, required_gb: int = 4) -> bool:
        """Check available system memory.
        
        Args:
            required_gb: Required memory in GB
            
        Returns:
            True if sufficient memory available
        """
        import psutil
        
        memory = psutil.virtual_memory()
        total_gb = memory.total / (1024 ** 3)
        available_gb = memory.available / (1024 ** 3)
        
        if total_gb < required_gb:
            self.errors.append(
                f"Insufficient system memory: {total_gb:.1f} GB total (require {required_gb} GB)"
            )
            return False
            
        if available_gb < required_gb / 2:
            self.warnings.append(
                f"Low available memory: {available_gb:.1f} GB free"
            )
            
        logger.info(f"✓ System memory: {total_gb:.1f} GB total, {available_gb:.1f} GB available")
        return True
        
    def check_network_connectivity(self) -> bool:
        """Check network connectivity to required services.
        
        Returns:
            True if all services reachable
        """
        import socket
        
        services = [
            ('pypi.org', 443),
            ('github.com', 443),
            ('huggingface.co', 443)
        ]
        
        unreachable = []
        for host, port in services:
            try:
                socket.create_connection((host, port), timeout=5).close()
                logger.info(f"✓ Connected to {host}:{port}")
            except (socket.timeout, socket.error):
                unreachable.append(f"{host}:{port}")
                
        if unreachable:
            self.warnings.append(f"Cannot reach: {', '.join(unreachable)}")
            
        return len(unreachable) < len(services)
        
    def setup_directories(self, base_dir: Path) -> bool:
        """Setup required directory structure.
        
        Args:
            base_dir: Base directory for installation
            
        Returns:
            True if directories created successfully
        """
        directories = [
            base_dir / "data" / "chromadb",
            base_dir / "data" / "cache",
            base_dir / "data" / "backup",
            base_dir / "data" / "export",
            base_dir / "logs",
            base_dir / "config"
        ]
        
        try:
            for directory in directories:
                directory.mkdir(parents=True, exist_ok=True)
                logger.info(f"✓ Created directory: {directory}")
            return True
        except Exception as e:
            self.errors.append(f"Failed to create directories: {e}")
            return False
            
    def generate_env_file(self, output_path: Path, config: Optional[Dict] = None) -> bool:
        """Generate .env configuration file.
        
        Args:
            output_path: Path to write .env file
            config: Optional configuration overrides
            
        Returns:
            True if file created successfully
        """
        default_config = {
            'MCP_SERVER_NAME': 'TTRPG Assistant',
            'DEBUG': 'false',
            'LOG_LEVEL': 'INFO',
            'CHROMA_DB_PATH': './data/chromadb',
            'CACHE_DIR': './data/cache',
            'MAX_CHUNK_SIZE': '1000',
            'CHUNK_OVERLAP': '200',
            'EMBEDDING_MODEL': 'all-MiniLM-L6-v2',
            'DEFAULT_SEARCH_RESULTS': '5',
            'ENABLE_HYBRID_SEARCH': 'true',
            'SEMANTIC_WEIGHT': '0.7',
            'KEYWORD_WEIGHT': '0.3',
            'CACHE_TTL_SECONDS': '3600',
            'ENABLE_RATE_LIMITING': 'true',
            'ENABLE_AUDIT': 'true',
            'SESSION_TIMEOUT_MINUTES': '60'
        }
        
        if config:
            default_config.update(config)
            
        try:
            with open(output_path, 'w') as f:
                f.write("# TTRPG Assistant MCP Server Configuration\n")
                f.write("# Generated by setup_environment.py\n\n")
                
                for key, value in default_config.items():
                    f.write(f"{key}={value}\n")
                    
            logger.info(f"✓ Generated .env file at {output_path}")
            return True
        except Exception as e:
            self.errors.append(f"Failed to generate .env file: {e}")
            return False
            
    def check_python_packages(self) -> Dict[str, bool]:
        """Check for installed Python packages.
        
        Returns:
            Dictionary of package installation status
        """
        required_packages = [
            'mcp',
            'fastmcp',
            'chromadb',
            'torch',
            'transformers',
            'sentence-transformers',
            'spacy',
            'pydantic',
            'fastapi',
            'uvicorn'
        ]
        
        package_status = {}
        
        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
                package_status[package] = True
                logger.info(f"✓ Package '{package}' is installed")
            except ImportError:
                package_status[package] = False
                self.warnings.append(f"Package '{package}' is not installed")
                
        return package_status
        
    def setup_spacy_models(self) -> bool:
        """Download required spaCy models.
        
        Returns:
            True if models downloaded successfully
        """
        try:
            import spacy
            
            # Check if model is already installed
            try:
                spacy.load("en_core_web_sm")
                logger.info("✓ spaCy model 'en_core_web_sm' already installed")
                return True
            except OSError:
                # Download the model
                logger.info("Downloading spaCy model 'en_core_web_sm'...")
                subprocess.run(
                    [sys.executable, '-m', 'spacy', 'download', 'en_core_web_sm'],
                    check=True,
                    capture_output=True
                )
                logger.info("✓ spaCy model downloaded successfully")
                return True
        except Exception as e:
            self.warnings.append(f"Failed to setup spaCy models: {e}")
            return False
            
    def validate_configuration(self, config_path: Path) -> bool:
        """Validate configuration files.
        
        Args:
            config_path: Path to configuration directory
            
        Returns:
            True if configuration is valid
        """
        env_file = config_path / ".env"
        
        if not env_file.exists():
            self.warnings.append(f"Configuration file not found: {env_file}")
            return False
            
        # Check for required configuration values
        required_vars = [
            'CHROMA_DB_PATH',
            'CACHE_DIR',
            'EMBEDDING_MODEL'
        ]
        
        missing_vars = []
        with open(env_file) as f:
            content = f.read()
            for var in required_vars:
                if f"{var}=" not in content:
                    missing_vars.append(var)
                    
        if missing_vars:
            self.warnings.append(f"Missing configuration variables: {', '.join(missing_vars)}")
            return False
            
        logger.info("✓ Configuration validated successfully")
        return True
        
    def run_full_check(self) -> bool:
        """Run all environment checks.
        
        Returns:
            True if all critical checks pass
        """
        logger.info("Starting environment validation...")
        
        checks = [
            ("Python version", self.check_python_version()),
            ("System dependencies", self.check_system_dependencies()),
            ("Memory requirements", self.check_memory()),
            ("Disk space", self.check_disk_space()),
            ("Network connectivity", self.check_network_connectivity())
        ]
        
        # Optional checks
        gpu_info = self.check_gpu_support()
        package_status = self.check_python_packages()
        
        # Print summary
        logger.info("\n" + "="*50)
        logger.info("Environment Check Summary")
        logger.info("="*50)
        
        for check_name, passed in checks:
            status = "✓ PASS" if passed else "✗ FAIL"
            logger.info(f"{check_name}: {status}")
            
        if self.errors:
            logger.error("\nCritical Errors:")
            for error in self.errors:
                logger.error(f"  - {error}")
                
        if self.warnings:
            logger.warning("\nWarnings:")
            for warning in self.warnings:
                logger.warning(f"  - {warning}")
                
        # Return True only if no critical errors
        return len(self.errors) == 0
        

def main():
    """Main entry point for environment setup."""
    parser = argparse.ArgumentParser(
        description="Setup and validate environment for TTRPG Assistant MCP Server"
    )
    parser.add_argument(
        '--config-dir',
        type=Path,
        default=Path('/etc/ttrpg-assistant'),
        help='Configuration directory path'
    )
    parser.add_argument(
        '--install-dir',
        type=Path,
        default=Path('/opt/ttrpg-assistant'),
        help='Installation directory path'
    )
    parser.add_argument(
        '--generate-env',
        action='store_true',
        help='Generate .env configuration file'
    )
    parser.add_argument(
        '--setup-dirs',
        action='store_true',
        help='Create directory structure'
    )
    parser.add_argument(
        '--download-models',
        action='store_true',
        help='Download required models'
    )
    parser.add_argument(
        '--json-output',
        action='store_true',
        help='Output results in JSON format'
    )
    
    args = parser.parse_args()
    
    # Initialize setup
    setup = EnvironmentSetup(args.config_dir)
    
    # Run requested operations
    if args.setup_dirs:
        setup.setup_directories(args.install_dir)
        
    if args.generate_env:
        env_path = args.config_dir / '.env'
        setup.generate_env_file(env_path)
        
    if args.download_models:
        setup.setup_spacy_models()
        
    # Run validation
    success = setup.run_full_check()
    
    if args.json_output:
        result = {
            'success': success,
            'errors': setup.errors,
            'warnings': setup.warnings,
            'platform': setup.platform,
            'python_version': f"{setup.python_version.major}.{setup.python_version.minor}.{setup.python_version.micro}"
        }
        print(json.dumps(result, indent=2))
    else:
        sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()