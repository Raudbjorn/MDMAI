#!/usr/bin/env python3
"""Comprehensive test runner for the TTRPG MCP project."""

import argparse
import asyncio
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestRunner:
    """Orchestrates test execution with various configurations."""
    
    def __init__(self, verbose: bool = False, coverage: bool = True):
        self.verbose = verbose
        self.coverage = coverage
        self.results = {}
        self.start_time = None
        self.end_time = None
    
    def run_command(self, cmd: List[str], env: Optional[dict] = None) -> Tuple[int, str, str]:
        """Run a command and return exit code, stdout, and stderr."""
        if self.verbose:
            print(f"Running: {' '.join(cmd)}")
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env or os.environ.copy()
        )
        
        stdout, stderr = process.communicate()
        return process.returncode, stdout, stderr
    
    def run_unit_tests(self) -> bool:
        """Run unit tests."""
        print("\n" + "=" * 60)
        print("RUNNING UNIT TESTS")
        print("=" * 60)
        
        cmd = ["pytest", "-m", "unit", "-v"]
        if self.coverage:
            cmd.extend(["--cov=src", "--cov-report=term-missing"])
        
        exit_code, stdout, stderr = self.run_command(cmd)
        
        self.results["unit"] = {
            "passed": exit_code == 0,
            "output": stdout,
            "errors": stderr
        }
        
        if self.verbose:
            print(stdout)
            if stderr:
                print("Errors:", stderr)
        
        return exit_code == 0
    
    def run_integration_tests(self) -> bool:
        """Run integration tests."""
        print("\n" + "=" * 60)
        print("RUNNING INTEGRATION TESTS")
        print("=" * 60)
        
        cmd = ["pytest", "-m", "integration", "-v"]
        if self.coverage:
            cmd.extend(["--cov=src", "--cov-append"])
        
        exit_code, stdout, stderr = self.run_command(cmd)
        
        self.results["integration"] = {
            "passed": exit_code == 0,
            "output": stdout,
            "errors": stderr
        }
        
        if self.verbose:
            print(stdout)
            if stderr:
                print("Errors:", stderr)
        
        return exit_code == 0
    
    def run_e2e_tests(self) -> bool:
        """Run end-to-end tests."""
        print("\n" + "=" * 60)
        print("RUNNING E2E TESTS")
        print("=" * 60)
        
        cmd = ["pytest", "-m", "e2e", "-v"]
        if self.coverage:
            cmd.extend(["--cov=src", "--cov-append"])
        
        exit_code, stdout, stderr = self.run_command(cmd)
        
        self.results["e2e"] = {
            "passed": exit_code == 0,
            "output": stdout,
            "errors": stderr
        }
        
        if self.verbose:
            print(stdout)
            if stderr:
                print("Errors:", stderr)
        
        return exit_code == 0
    
    def run_security_tests(self) -> bool:
        """Run security tests."""
        print("\n" + "=" * 60)
        print("RUNNING SECURITY TESTS")
        print("=" * 60)
        
        cmd = ["pytest", "-m", "security", "-v"]
        
        exit_code, stdout, stderr = self.run_command(cmd)
        
        self.results["security"] = {
            "passed": exit_code == 0,
            "output": stdout,
            "errors": stderr
        }
        
        if self.verbose:
            print(stdout)
            if stderr:
                print("Errors:", stderr)
        
        return exit_code == 0
    
    def run_load_tests(self, duration: int = 30) -> bool:
        """Run load tests."""
        print("\n" + "=" * 60)
        print("RUNNING LOAD TESTS")
        print("=" * 60)
        
        # Run a subset of load tests (not all by default as they're slow)
        cmd = [
            "pytest",
            "tests/load/test_load_performance.py::TestLoadPerformance::test_sustained_load",
            "-v",
            "--tb=short"
        ]
        
        exit_code, stdout, stderr = self.run_command(cmd)
        
        self.results["load"] = {
            "passed": exit_code == 0,
            "output": stdout,
            "errors": stderr
        }
        
        if self.verbose:
            print(stdout)
            if stderr:
                print("Errors:", stderr)
        
        return exit_code == 0
    
    def run_specific_file(self, file_path: str) -> bool:
        """Run tests in a specific file."""
        print(f"\n" + "=" * 60)
        print(f"RUNNING TESTS IN: {file_path}")
        print("=" * 60)
        
        cmd = ["pytest", file_path, "-v"]
        if self.coverage:
            cmd.extend(["--cov=src", "--cov-report=term-missing"])
        
        exit_code, stdout, stderr = self.run_command(cmd)
        
        self.results["specific"] = {
            "passed": exit_code == 0,
            "output": stdout,
            "errors": stderr,
            "file": file_path
        }
        
        if self.verbose:
            print(stdout)
            if stderr:
                print("Errors:", stderr)
        
        return exit_code == 0
    
    def run_all_tests(self) -> bool:
        """Run all test suites."""
        print("\n" + "=" * 60)
        print("RUNNING ALL TESTS")
        print("=" * 60)
        
        cmd = ["pytest", "-v"]
        if self.coverage:
            cmd.extend([
                "--cov=src",
                "--cov-report=term-missing",
                "--cov-report=html",
                "--cov-report=xml"
            ])
        
        exit_code, stdout, stderr = self.run_command(cmd)
        
        self.results["all"] = {
            "passed": exit_code == 0,
            "output": stdout,
            "errors": stderr
        }
        
        if self.verbose:
            print(stdout)
            if stderr:
                print("Errors:", stderr)
        
        return exit_code == 0
    
    def run_fast_tests(self) -> bool:
        """Run only fast tests (exclude slow, load, stress)."""
        print("\n" + "=" * 60)
        print("RUNNING FAST TESTS")
        print("=" * 60)
        
        cmd = ["pytest", "-m", "not slow and not load and not stress", "-v"]
        if self.coverage:
            cmd.extend(["--cov=src", "--cov-report=term-missing"])
        
        exit_code, stdout, stderr = self.run_command(cmd)
        
        self.results["fast"] = {
            "passed": exit_code == 0,
            "output": stdout,
            "errors": stderr
        }
        
        if self.verbose:
            print(stdout)
            if stderr:
                print("Errors:", stderr)
        
        return exit_code == 0
    
    def generate_coverage_report(self):
        """Generate coverage report."""
        print("\n" + "=" * 60)
        print("GENERATING COVERAGE REPORT")
        print("=" * 60)
        
        cmd = ["coverage", "report", "--show-missing"]
        exit_code, stdout, stderr = self.run_command(cmd)
        
        print(stdout)
        
        # Generate HTML report
        cmd = ["coverage", "html"]
        self.run_command(cmd)
        print("HTML coverage report generated in htmlcov/index.html")
    
    def print_summary(self):
        """Print test execution summary."""
        print("\n" + "=" * 60)
        print("TEST EXECUTION SUMMARY")
        print("=" * 60)
        
        total_duration = self.end_time - self.start_time if self.end_time else 0
        
        for suite, result in self.results.items():
            status = "✓ PASSED" if result["passed"] else "✗ FAILED"
            print(f"{suite.upper():15} {status}")
            
            # Extract test counts from output if available
            output = result.get("output", "")
            if "passed" in output.lower():
                # Try to extract test counts
                import re
                match = re.search(r"(\d+) passed", output)
                if match:
                    print(f"  Tests passed: {match.group(1)}")
        
        print(f"\nTotal duration: {total_duration:.2f} seconds")
        
        # Overall result
        all_passed = all(r["passed"] for r in self.results.values())
        if all_passed:
            print("\n✓ ALL TESTS PASSED")
        else:
            print("\n✗ SOME TESTS FAILED")
            failed_suites = [s for s, r in self.results.items() if not r["passed"]]
            print(f"Failed suites: {', '.join(failed_suites)}")
        
        return all_passed
    
    def run(self, test_suite: str = "all") -> bool:
        """Run the specified test suite."""
        self.start_time = time.time()
        
        suite_map = {
            "all": self.run_all_tests,
            "unit": self.run_unit_tests,
            "integration": self.run_integration_tests,
            "e2e": self.run_e2e_tests,
            "security": self.run_security_tests,
            "load": self.run_load_tests,
            "fast": self.run_fast_tests,
        }
        
        if test_suite in suite_map:
            success = suite_map[test_suite]()
        elif test_suite.endswith(".py"):
            # Run specific file
            success = self.run_specific_file(test_suite)
        else:
            print(f"Unknown test suite: {test_suite}")
            print(f"Available suites: {', '.join(suite_map.keys())}")
            return False
        
        self.end_time = time.time()
        
        if self.coverage and test_suite != "security":
            self.generate_coverage_report()
        
        return self.print_summary()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run TTRPG MCP test suites")
    parser.add_argument(
        "suite",
        nargs="?",
        default="all",
        help="Test suite to run (all, unit, integration, e2e, security, load, fast) or specific file"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--no-coverage",
        action="store_true",
        help="Disable coverage reporting"
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Watch for changes and re-run tests"
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run tests in parallel"
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Profile test execution"
    )
    
    args = parser.parse_args()
    
    # Set up environment
    os.environ["TESTING"] = "true"
    os.environ["PYTHONPATH"] = str(Path(__file__).parent.parent)
    
    runner = TestRunner(
        verbose=args.verbose,
        coverage=not args.no_coverage
    )
    
    if args.watch:
        print("Watching for changes... Press Ctrl+C to stop")
        try:
            while True:
                os.system("clear")  # Clear screen
                success = runner.run(args.suite)
                print("\nWaiting for changes...")
                time.sleep(2)  # Simple polling, could use watchdog for better solution
        except KeyboardInterrupt:
            print("\nStopped watching")
            return 0
    else:
        success = runner.run(args.suite)
        return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())