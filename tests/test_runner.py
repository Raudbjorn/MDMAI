#!/usr/bin/env python3
"""
Comprehensive test runner for the TTRPG Assistant project.

This script provides organized test execution with different categories:
- Unit tests: Fast tests for individual components
- Integration tests: Tests for component interactions  
- E2E tests: End-to-end workflow tests
- Load tests: Performance and stress tests
- Security tests: Security-focused tests
- Regression tests: Tests for bug fixes and regressions

Usage:
    python tests/test_runner.py [category] [options]
    
Categories:
    unit        - Run unit tests only (fastest)
    integration - Run integration tests
    e2e         - Run end-to-end tests
    load        - Run performance/load tests
    security    - Run security tests
    regression  - Run regression tests
    all         - Run all tests (default)
    
Options:
    --fast      - Skip slow tests
    --verbose   - Verbose output
    --coverage  - Generate coverage report
    --parallel  - Run tests in parallel
    --marker    - Run tests with specific marker
"""

import sys
import subprocess
import argparse
from pathlib import Path

# Test categories and their paths
TEST_CATEGORIES = {
    'unit': [
        'tests/unit/',
    ],
    'integration': [
        'tests/integration/',
    ],
    'e2e': [
        'tests/e2e/',
    ],
    'load': [
        'tests/load/',
    ],
    'security': [
        'tests/security/',
        'tests/unit/security/',
    ],
    'regression': [
        'tests/regression/',
    ],
    'bridge': [
        'tests/bridge/',
        'tests/unit/bridge/',
    ],
    'all': ['tests/'],
}

def build_pytest_command(category='all', fast=False, verbose=False, coverage=True, parallel=False, marker=None, keyword=None):
    """Build the pytest command based on options."""
    cmd = ['python', '-m', 'pytest']
    
    # Add paths for category
    if category in TEST_CATEGORIES:
        for path in TEST_CATEGORIES[category]:
            if Path(path).exists():
                cmd.append(path)
    else:
        print(f"Unknown category: {category}")
        print(f"Available categories: {', '.join(TEST_CATEGORIES.keys())}")
        return None
    
    # Add options
    if verbose:
        cmd.append('-v')
    else:
        cmd.append('-q')
        
    if fast:
        cmd.extend(['-m', 'not slow'])
        
    if marker:
        cmd.extend(['-m', marker])
        
    if keyword:
        cmd.extend(['-k', keyword])
        
    if parallel:
        cmd.extend(['-n', 'auto'])
        
    if coverage and category in ['all', 'unit']:
        cmd.extend([
            '--cov=src',
            '--cov-report=term-missing',
            '--cov-report=html',
        ])
    else:
        cmd.append('--no-cov')
        
    return cmd

def main():
    parser = argparse.ArgumentParser(
        description='Run TTRPG Assistant tests',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        'category', 
        nargs='?', 
        default='all',
        choices=list(TEST_CATEGORIES.keys()),
        help='Test category to run'
    )
    
    parser.add_argument(
        '--fast', 
        action='store_true',
        help='Skip slow tests'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    
    parser.add_argument(
        '--no-coverage',
        action='store_true',
        help='Disable coverage reporting'
    )
    
    parser.add_argument(
        '--parallel', '-n',
        action='store_true', 
        help='Run tests in parallel'
    )
    
    parser.add_argument(
        '--marker', '-m',
        help='Run tests with specific marker'
    )
    
    parser.add_argument(
        '--keyword', '-k',
        help='Run tests matching keyword expression'
    )
    
    args = parser.parse_args()
    
    # Build command
    cmd = build_pytest_command(
        category=args.category,
        fast=args.fast,
        verbose=args.verbose,
        coverage=not args.no_coverage,
        parallel=args.parallel,
        marker=args.marker,
        keyword=args.keyword
    )
    
    if not cmd:
        return 1
        
    # Print what we're running
    print(f"Running {args.category} tests...")
    if args.verbose:
        print(f"Command: {' '.join(cmd)}")
    
    # Execute
    try:
        result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
        return result.returncode
    except KeyboardInterrupt:
        print("\nTests interrupted by user")
        return 130
    except Exception as e:
        print(f"Error running tests: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())