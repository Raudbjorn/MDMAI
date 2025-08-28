#!/usr/bin/env python3
"""
Test runner for dependency update validation.

This script runs comprehensive tests after dependency updates to ensure
all components work correctly with the new versions.

Usage:
    python tests/run_dependency_tests.py [options]
    
Options:
    --verbose, -v     : Verbose output
    --coverage        : Generate coverage report
    --quick           : Run only quick tests (skip stress tests)
    --report          : Generate HTML report
    --parallel        : Run tests in parallel
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pytest
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text


console = Console()


class DependencyTestRunner:
    """Runner for dependency update tests."""

    def __init__(
        self,
        verbose: bool = False,
        coverage: bool = False,
        quick: bool = False,
        report: bool = False,
        parallel: bool = False
    ):
        """Initialize test runner."""
        self.verbose = verbose
        self.coverage = coverage
        self.quick = quick
        self.report = report
        self.parallel = parallel
        self.test_dir = Path(__file__).parent
        self.project_root = self.test_dir.parent
        self.results: Dict[str, any] = {}

    def check_dependencies(self) -> Tuple[bool, List[str]]:
        """Check if required dependencies are installed with correct versions."""
        console.print("\n[bold blue]Checking Dependencies...[/bold blue]")
        
        required_versions = {
            'pypdf': '6.0.0',
            'transformers': '4.53.0',
            'torch': '2.8.0',
            'aiohttp': '3.12.14',
        }
        
        missing = []
        version_mismatches = []
        
        for package, required_version in required_versions.items():
            try:
                import importlib.metadata
                installed_version = importlib.metadata.version(package)
                
                # Parse versions for comparison
                required_parts = required_version.split('.')
                installed_parts = installed_version.split('.')
                
                # Compare major.minor versions
                if (installed_parts[0] != required_parts[0] or 
                    installed_parts[1] != required_parts[1]):
                    version_mismatches.append(
                        f"{package}: installed={installed_version}, required={required_version}"
                    )
                    console.print(
                        f"  ❌ {package}: [red]v{installed_version}[/red] "
                        f"(required: v{required_version})"
                    )
                else:
                    console.print(
                        f"  ✅ {package}: [green]v{installed_version}[/green]"
                    )
            except Exception:
                missing.append(package)
                console.print(f"  ❌ {package}: [red]Not installed[/red]")
        
        if missing or version_mismatches:
            return False, missing + version_mismatches
        return True, []

    def run_pytest_tests(self) -> bool:
        """Run pytest tests for dependency updates."""
        console.print("\n[bold blue]Running Dependency Update Tests...[/bold blue]")
        
        pytest_args = [
            str(self.test_dir / "test_dependency_updates.py"),
            "-v" if self.verbose else "-q",
            "--tb=short",
            "--color=yes",
        ]
        
        if self.coverage:
            pytest_args.extend([
                "--cov=src",
                "--cov-report=term-missing",
                "--cov-report=html:htmlcov_deps",
            ])
        
        if self.quick:
            pytest_args.extend([
                "-m", "not stress",
                "-k", "not performance"
            ])
        
        if self.parallel:
            pytest_args.extend(["-n", "auto"])
        
        if self.report:
            pytest_args.extend([
                "--html=test_report_deps.html",
                "--self-contained-html"
            ])
        
        # Run tests with progress indicator
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Running tests...", total=None)
            
            result = subprocess.run(
                [sys.executable, "-m", "pytest"] + pytest_args,
                capture_output=not self.verbose,
                text=True
            )
            
            progress.stop()
        
        if result.returncode == 0:
            console.print("  ✅ [green]All tests passed![/green]")
            return True
        else:
            console.print("  ❌ [red]Some tests failed[/red]")
            if not self.verbose and result.stdout:
                console.print("\n[dim]Test Output:[/dim]")
                console.print(result.stdout)
            return False

    def run_integration_tests(self) -> bool:
        """Run integration tests between components."""
        console.print("\n[bold blue]Running Integration Tests...[/bold blue]")
        
        test_files = [
            "test_integration.py",
            "test_pdf_processing.py",
            "test_ai_providers.py",
            "test_bridge.py",
        ]
        
        all_passed = True
        results_table = Table(title="Integration Test Results")
        results_table.add_column("Test File", style="cyan")
        results_table.add_column("Status", style="white")
        results_table.add_column("Time", style="yellow")
        
        for test_file in test_files:
            test_path = self.test_dir / test_file
            if not test_path.exists():
                results_table.add_row(
                    test_file,
                    "[yellow]SKIPPED[/yellow]",
                    "N/A"
                )
                continue
            
            start_time = time.time()
            result = subprocess.run(
                [sys.executable, "-m", "pytest", str(test_path), "-q"],
                capture_output=True,
                text=True
            )
            elapsed_time = time.time() - start_time
            
            if result.returncode == 0:
                results_table.add_row(
                    test_file,
                    "[green]PASSED[/green]",
                    f"{elapsed_time:.2f}s"
                )
            else:
                results_table.add_row(
                    test_file,
                    "[red]FAILED[/red]",
                    f"{elapsed_time:.2f}s"
                )
                all_passed = False
        
        console.print(results_table)
        return all_passed

    def check_imports(self) -> bool:
        """Check if all modules can be imported successfully."""
        console.print("\n[bold blue]Checking Module Imports...[/bold blue]")
        
        modules_to_check = [
            "src.pdf_processing.pdf_parser",
            "src.ai_providers.provider_manager",
            "src.bridge.bridge_server",
            "src.core.database",
            "src.search.search_service",
        ]
        
        import_errors = []
        
        for module in modules_to_check:
            try:
                __import__(module)
                console.print(f"  ✅ {module}")
            except ImportError as e:
                import_errors.append((module, str(e)))
                console.print(f"  ❌ {module}: [red]{e}[/red]")
        
        return len(import_errors) == 0

    def run_performance_benchmarks(self) -> bool:
        """Run performance benchmarks to ensure no regression."""
        if self.quick:
            console.print("\n[dim]Skipping performance benchmarks (--quick mode)[/dim]")
            return True
        
        console.print("\n[bold blue]Running Performance Benchmarks...[/bold blue]")
        
        # Run performance tests
        result = subprocess.run(
            [
                sys.executable, "-m", "pytest",
                str(self.test_dir / "test_dependency_updates.py"),
                "-k", "performance",
                "-q"
            ],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            console.print("  ✅ [green]Performance benchmarks passed[/green]")
            return True
        else:
            console.print("  ❌ [red]Performance regression detected[/red]")
            return False

    def generate_report(self) -> None:
        """Generate test report."""
        console.print("\n[bold blue]Generating Report...[/bold blue]")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'dependencies_checked': True,
            'tests_passed': self.results.get('tests_passed', False),
            'integration_passed': self.results.get('integration_passed', False),
            'imports_successful': self.results.get('imports_successful', False),
            'performance_ok': self.results.get('performance_ok', False),
            'options': {
                'verbose': self.verbose,
                'coverage': self.coverage,
                'quick': self.quick,
                'parallel': self.parallel,
            }
        }
        
        report_path = self.project_root / "dependency_test_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        console.print(f"  📄 Report saved to: {report_path}")
        
        if self.coverage:
            console.print(f"  📊 Coverage report: htmlcov_deps/index.html")
        
        if self.report:
            console.print(f"  📈 HTML report: test_report_deps.html")

    def run(self) -> int:
        """Run all tests and checks."""
        console.print(Panel.fit(
            "[bold]Dependency Update Test Suite[/bold]\n"
            "Testing compatibility with updated dependencies",
            style="blue"
        ))
        
        # Check dependencies
        deps_ok, issues = self.check_dependencies()
        if not deps_ok:
            console.print(
                f"\n[red]⚠️  Dependency issues found:[/red]"
            )
            for issue in issues:
                console.print(f"    • {issue}")
            console.print(
                "\n[yellow]Run 'pip install -r requirements.txt' to fix[/yellow]"
            )
            # Continue anyway for testing purposes
        
        # Check imports
        imports_ok = self.check_imports()
        self.results['imports_successful'] = imports_ok
        
        # Run pytest tests
        tests_passed = self.run_pytest_tests()
        self.results['tests_passed'] = tests_passed
        
        # Run integration tests
        integration_passed = self.run_integration_tests()
        self.results['integration_passed'] = integration_passed
        
        # Run performance benchmarks
        performance_ok = self.run_performance_benchmarks()
        self.results['performance_ok'] = performance_ok
        
        # Generate report if requested
        if self.report:
            self.generate_report()
        
        # Summary
        console.print("\n" + "=" * 50)
        console.print("[bold]Test Summary[/bold]")
        console.print("=" * 50)
        
        summary_table = Table(show_header=False)
        summary_table.add_column("Check", style="cyan")
        summary_table.add_column("Status", style="white")
        
        summary_table.add_row(
            "Dependencies",
            "[green]✅ OK[/green]" if deps_ok else "[yellow]⚠️  Issues[/yellow]"
        )
        summary_table.add_row(
            "Module Imports",
            "[green]✅ OK[/green]" if imports_ok else "[red]❌ Failed[/red]"
        )
        summary_table.add_row(
            "Unit Tests",
            "[green]✅ Passed[/green]" if tests_passed else "[red]❌ Failed[/red]"
        )
        summary_table.add_row(
            "Integration Tests",
            "[green]✅ Passed[/green]" if integration_passed else "[red]❌ Failed[/red]"
        )
        summary_table.add_row(
            "Performance",
            "[green]✅ OK[/green]" if performance_ok else "[red]❌ Regression[/red]"
        )
        
        console.print(summary_table)
        
        # Overall result
        all_passed = all([
            imports_ok,
            tests_passed,
            integration_passed,
            performance_ok
        ])
        
        if all_passed:
            console.print(
                "\n[bold green]✨ All tests passed! Dependencies are compatible.[/bold green]"
            )
            return 0
        else:
            console.print(
                "\n[bold red]⚠️  Some tests failed. Review the output above.[/bold red]"
            )
            return 1


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test runner for dependency updates"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Generate coverage report"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run only quick tests (skip stress tests)"
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate HTML report"
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run tests in parallel"
    )
    
    args = parser.parse_args()
    
    runner = DependencyTestRunner(
        verbose=args.verbose,
        coverage=args.coverage,
        quick=args.quick,
        report=args.report,
        parallel=args.parallel
    )
    
    return runner.run()


if __name__ == "__main__":
    sys.exit(main())