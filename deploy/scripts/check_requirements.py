#!/usr/bin/env python3
"""
Dependency checker for TTRPG Assistant MCP Server.
Validates all required dependencies and their versions.
"""

import sys
import importlib
import subprocess
from typing import Dict, List, Optional, Tuple
import json
import argparse
import pkg_resources
from packaging import version
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class DependencyChecker:
    """Check and validate project dependencies."""
    
    def __init__(self, requirements_file: Optional[str] = None):
        """Initialize dependency checker.
        
        Args:
            requirements_file: Path to requirements.txt file
        """
        self.requirements_file = requirements_file or 'requirements.txt'
        self.missing_packages: List[str] = []
        self.version_mismatches: List[Dict] = []
        self.optional_missing: List[str] = []
        
    def parse_requirements(self) -> Dict[str, str]:
        """Parse requirements from requirements.txt.
        
        Returns:
            Dictionary of package names to version specifications
        """
        requirements = {}
        
        try:
            with open(self.requirements_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # Parse package and version
                        if '>=' in line:
                            package, version_spec = line.split('>=')
                            requirements[package.strip()] = f'>={version_spec.strip()}'
                        elif '==' in line:
                            package, version_spec = line.split('==')
                            requirements[package.strip()] = f'=={version_spec.strip()}'
                        elif '>' in line:
                            package, version_spec = line.split('>')
                            requirements[package.strip()] = f'>{version_spec.strip()}'
                        else:
                            requirements[line.strip()] = ''
        except FileNotFoundError:
            logger.error(f"Requirements file not found: {self.requirements_file}")
            
        return requirements
        
    def check_package(self, package_name: str, version_spec: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """Check if a package is installed and meets version requirements.
        
        Args:
            package_name: Name of the package
            version_spec: Version specification (e.g., ">=1.0.0")
            
        Returns:
            Tuple of (is_installed, installed_version, meets_requirements)
        """
        try:
            # Try to get package info
            pkg = pkg_resources.get_distribution(package_name)
            installed_version = pkg.version
            
            if version_spec:
                # Check version requirement
                requirement = pkg_resources.Requirement(f"{package_name}{version_spec}")
                meets_requirement = pkg in requirement
            else:
                meets_requirement = True
                
            return True, installed_version, meets_requirement
            
        except pkg_resources.DistributionNotFound:
            return False, None, False
        except Exception as e:
            logger.warning(f"Error checking {package_name}: {e}")
            return False, None, False
            
    def check_torch_gpu(self) -> Dict[str, bool]:
        """Check PyTorch GPU support.
        
        Returns:
            Dictionary with GPU support information
        """
        gpu_info = {
            'torch_installed': False,
            'cuda_available': False,
            'cuda_version': None,
            'cudnn_available': False,
            'gpu_count': 0,
            'gpu_names': []
        }
        
        try:
            import torch
            gpu_info['torch_installed'] = True
            
            if torch.cuda.is_available():
                gpu_info['cuda_available'] = True
                gpu_info['cuda_version'] = torch.version.cuda
                gpu_info['cudnn_available'] = torch.backends.cudnn.is_available()
                gpu_info['gpu_count'] = torch.cuda.device_count()
                
                for i in range(gpu_info['gpu_count']):
                    gpu_info['gpu_names'].append(torch.cuda.get_device_name(i))
                    
        except ImportError:
            pass
            
        return gpu_info
        
    def check_spacy_models(self) -> Dict[str, bool]:
        """Check for installed spaCy models.
        
        Returns:
            Dictionary of model installation status
        """
        models = {
            'en_core_web_sm': False,
            'en_core_web_md': False,
            'en_core_web_lg': False
        }
        
        try:
            import spacy
            
            for model_name in models.keys():
                try:
                    spacy.load(model_name)
                    models[model_name] = True
                except OSError:
                    pass
                    
        except ImportError:
            logger.warning("spaCy not installed, cannot check models")
            
        return models
        
    def check_system_commands(self) -> Dict[str, bool]:
        """Check for required system commands.
        
        Returns:
            Dictionary of command availability
        """
        commands = ['git', 'curl', 'python3', 'pip']
        available = {}
        
        for cmd in commands:
            try:
                result = subprocess.run(
                    [cmd, '--version'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                available[cmd] = result.returncode == 0
            except (subprocess.TimeoutExpired, FileNotFoundError):
                available[cmd] = False
                
        return available
        
    def check_all_dependencies(self) -> Dict:
        """Check all project dependencies.
        
        Returns:
            Dictionary with complete dependency status
        """
        logger.info("Checking project dependencies...")
        
        # Parse requirements
        requirements = self.parse_requirements()
        
        # Check each requirement
        results = {
            'packages': {},
            'missing': [],
            'version_mismatches': [],
            'optional_missing': []
        }
        
        # Core dependencies (must have)
        core_packages = [
            'mcp', 'fastmcp', 'chromadb', 'pydantic', 
            'python-dotenv', 'structlog', 'psutil'
        ]
        
        # Optional dependencies (nice to have)
        optional_packages = [
            'torch', 'transformers', 'sentence-transformers',
            'spacy', 'textblob'
        ]
        
        for package, version_spec in requirements.items():
            is_installed, installed_version, meets_req = self.check_package(package, version_spec)
            
            results['packages'][package] = {
                'required': version_spec,
                'installed': installed_version,
                'meets_requirement': meets_req,
                'is_core': package in core_packages
            }
            
            if not is_installed:
                if package in core_packages:
                    results['missing'].append(package)
                    self.missing_packages.append(package)
                elif package in optional_packages:
                    results['optional_missing'].append(package)
                    self.optional_missing.append(package)
            elif not meets_req:
                results['version_mismatches'].append({
                    'package': package,
                    'required': version_spec,
                    'installed': installed_version
                })
                self.version_mismatches.append({
                    'package': package,
                    'required': version_spec,
                    'installed': installed_version
                })
                
        # Check GPU support
        results['gpu'] = self.check_torch_gpu()
        
        # Check spaCy models
        results['spacy_models'] = self.check_spacy_models()
        
        # Check system commands
        results['system_commands'] = self.check_system_commands()
        
        return results
        
    def generate_install_command(self) -> str:
        """Generate pip install command for missing packages.
        
        Returns:
            pip install command string
        """
        if not self.missing_packages:
            return ""
            
        return f"pip install {' '.join(self.missing_packages)}"
        
    def print_report(self, results: Dict):
        """Print dependency check report.
        
        Args:
            results: Results from check_all_dependencies
        """
        print("\n" + "="*60)
        print("TTRPG Assistant MCP Server - Dependency Report")
        print("="*60 + "\n")
        
        # Core dependencies status
        print("Core Dependencies:")
        print("-" * 40)
        for package, info in results['packages'].items():
            if info.get('is_core'):
                status = "✓" if info['meets_requirement'] else "✗"
                installed = info['installed'] or "Not installed"
                required = info['required'] or "Any version"
                print(f"  {status} {package:25} {installed:15} (Required: {required})")
                
        # Optional dependencies
        print("\nOptional Dependencies:")
        print("-" * 40)
        for package, info in results['packages'].items():
            if not info.get('is_core'):
                status = "✓" if info['installed'] else "○"
                installed = info['installed'] or "Not installed"
                print(f"  {status} {package:25} {installed:15}")
                
        # GPU Support
        print("\nGPU Support:")
        print("-" * 40)
        gpu = results['gpu']
        if gpu['torch_installed']:
            if gpu['cuda_available']:
                print(f"  ✓ CUDA Available: {gpu['cuda_version']}")
                print(f"  ✓ GPU Count: {gpu['gpu_count']}")
                for gpu_name in gpu['gpu_names']:
                    print(f"    - {gpu_name}")
            else:
                print("  ○ No CUDA support (CPU-only mode)")
        else:
            print("  ○ PyTorch not installed")
            
        # spaCy Models
        print("\nspaCy Language Models:")
        print("-" * 40)
        for model, installed in results['spacy_models'].items():
            status = "✓" if installed else "○"
            print(f"  {status} {model}")
            
        # System Commands
        print("\nSystem Commands:")
        print("-" * 40)
        for cmd, available in results['system_commands'].items():
            status = "✓" if available else "✗"
            print(f"  {status} {cmd}")
            
        # Summary
        print("\n" + "="*60)
        if results['missing']:
            print("⚠ Missing core packages:")
            for pkg in results['missing']:
                print(f"  - {pkg}")
            print(f"\nInstall with: pip install {' '.join(results['missing'])}")
            
        if results['version_mismatches']:
            print("\n⚠ Version mismatches:")
            for mismatch in results['version_mismatches']:
                print(f"  - {mismatch['package']}: {mismatch['installed']} installed, {mismatch['required']} required")
                
        if not results['missing'] and not results['version_mismatches']:
            print("✓ All core dependencies satisfied!")
            
        print("="*60 + "\n")
        

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Check dependencies for TTRPG Assistant MCP Server"
    )
    parser.add_argument(
        '--requirements',
        default='requirements.txt',
        help='Path to requirements.txt file'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output results in JSON format'
    )
    parser.add_argument(
        '--install-missing',
        action='store_true',
        help='Generate install command for missing packages'
    )
    parser.add_argument(
        '--check-only',
        choices=['core', 'optional', 'gpu', 'spacy', 'system'],
        help='Check only specific category'
    )
    
    args = parser.parse_args()
    
    # Initialize checker
    checker = DependencyChecker(args.requirements)
    
    # Run checks
    results = checker.check_all_dependencies()
    
    # Output results
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        checker.print_report(results)
        
    # Generate install command if requested
    if args.install_missing and checker.missing_packages:
        print(f"\nInstall command:\n{checker.generate_install_command()}\n")
        
    # Exit with appropriate code
    if checker.missing_packages or checker.version_mismatches:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == '__main__':
    main()