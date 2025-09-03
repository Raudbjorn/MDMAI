#!/usr/bin/env python3
"""Test Runner for Provider Router Comprehensive Test Suite.

This script orchestrates the execution of all test categories including
unit tests, integration tests, performance tests, chaos engineering tests,
and generates comprehensive test reports.

Usage:
    python test_runner.py [options]
    
Options:
    --all              Run all test categories
    --unit             Run unit tests only
    --integration      Run integration tests only
    --performance      Run performance tests only
    --chaos            Run chaos engineering tests only
    --load             Run load tests only
    --report           Generate detailed test report
    --coverage         Generate coverage report
    --profile          Profile test execution
"""

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pytest


# ============================================================================
# TEST CONFIGURATION
# ============================================================================

class TestConfig:
    """Test suite configuration."""
    
    # Test categories and their configurations
    TEST_CATEGORIES = {
        "unit": {
            "marks": "-m 'not integration and not chaos and not load'",
            "options": "-v --tb=short",
            "files": [
                "test_provider_router_unit.py",
                "test_provider_router_comprehensive.py::TestProviderSelection",
                "test_provider_router_comprehensive.py::TestCircuitBreaker",
                "test_provider_router_comprehensive.py::TestFallbackChain",
                "test_provider_router_comprehensive.py::TestCostOptimization",
                "test_provider_router_comprehensive.py::TestHealthMonitoring"
            ]
        },
        "integration": {
            "marks": "-m integration",
            "options": "-v --tb=short --run-integration",
            "files": [
                "test_provider_router_integration.py",
                "test_provider_router_comprehensive.py::TestMCPProtocolIntegration",
                "test_provider_router_comprehensive.py::TestProviderFailover",
                "test_provider_router_comprehensive.py::TestStateSynchronization",
                "test_provider_router_comprehensive.py::TestCacheConsistency",
                "test_provider_router_comprehensive.py::TestEndToEndWorkflows"
            ]
        },
        "performance": {
            "marks": "-m benchmark",
            "options": "-v --tb=short --benchmark-only --benchmark-autosave",
            "files": [
                "test_provider_router_performance.py",
                "test_provider_router_comprehensive.py::TestRoutingPerformance",
                "test_provider_router_comprehensive.py::TestMemoryPerformance",
                "test_provider_router_comprehensive.py::TestCachePerformance"
            ]
        },
        "chaos": {
            "marks": "-m 'chaos or jepsen'",
            "options": "-v --tb=short --run-chaos --chaos-intensity=0.5",
            "files": [
                "test_provider_router_chaos_engineering.py",
                "test_provider_router_failure_scenarios.py",
                "test_provider_router_comprehensive.py::TestRandomFailures",
                "test_provider_router_comprehensive.py::TestNetworkChaos",
                "test_provider_router_comprehensive.py::TestResourceExhaustion",
                "test_provider_router_comprehensive.py::TestByzantineFailures",
                "test_provider_router_comprehensive.py::TestSplitBrainScenarios"
            ]
        },
        "property": {
            "marks": "",
            "options": "-v --tb=short --hypothesis-show-statistics",
            "files": [
                "test_provider_router_property_based.py"
            ]
        },
        "load": {
            "marks": "-m load",
            "options": "",
            "files": [
                "test_provider_router_load_testing.py"
            ],
            "runner": "locust"
        }
    }
    
    # Performance targets
    PERFORMANCE_TARGETS = {
        "latency_p99_ms": 100,
        "throughput_rps": 1000,
        "error_rate": 0.05,
        "cache_hit_rate": 0.8,
        "availability": 0.99,
        "memory_mb": 512
    }
    
    # Coverage targets
    COVERAGE_TARGETS = {
        "overall": 90,
        "unit": 95,
        "integration": 85
    }


# ============================================================================
# TEST RUNNER
# ============================================================================

class TestRunner:
    """Orchestrates test execution."""
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.results: Dict[str, Dict] = {}
        self.start_time = None
        self.end_time = None
        
    def run_category(self, category: str) -> Tuple[bool, Dict]:
        """Run a specific test category."""
        print(f"\n{'='*60}")
        print(f"Running {category.upper()} Tests")
        print(f"{'='*60}\n")
        
        category_config = self.config.TEST_CATEGORIES[category]
        
        # Special handling for load tests
        if category_config.get("runner") == "locust":
            return self._run_locust_tests(category_config)
            
        # Build pytest command
        cmd = ["pytest"]
        
        # Add files
        for file in category_config["files"]:
            cmd.append(file)
            
        # Add marks
        if category_config["marks"]:
            cmd.extend(category_config["marks"].split())
            
        # Add options
        if category_config["options"]:
            cmd.extend(category_config["options"].split())
            
        # Add coverage if needed
        if category in ["unit", "integration"]:
            cmd.extend([
                f"--cov=src.ai_providers",
                "--cov-report=term-missing",
                f"--cov-report=html:htmlcov/{category}"
            ])
            
        # Add parallel execution for faster tests
        if category in ["unit", "property"]:
            cmd.extend(["-n", "auto"])
            
        # Execute tests
        start = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        duration = time.time() - start
        
        # Parse results
        success = result.returncode == 0
        output = result.stdout + result.stderr
        
        # Extract statistics
        stats = self._parse_pytest_output(output)
        stats["duration"] = duration
        stats["success"] = success
        
        return success, stats
        
    def _run_locust_tests(self, config: Dict) -> Tuple[bool, Dict]:
        """Run Locust load tests."""
        cmd = [
            "locust",
            "-f", config["files"][0],
            "--host", "http://localhost:8000",
            "--users", "100",
            "--spawn-rate", "10",
            "--run-time", "60s",
            "--headless",
            "--only-summary",
            "--print-stats"
        ]
        
        start = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        duration = time.time() - start
        
        # Parse Locust output
        output = result.stdout
        stats = self._parse_locust_output(output)
        stats["duration"] = duration
        stats["success"] = result.returncode == 0
        
        return result.returncode == 0, stats
        
    def _parse_pytest_output(self, output: str) -> Dict:
        """Parse pytest output for statistics."""
        stats = {
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "errors": 0,
            "warnings": 0
        }
        
        # Look for summary line
        for line in output.split('\n'):
            if "passed" in line and "failed" in line:
                parts = line.split()
                for i, part in enumerate(parts):
                    if "passed" in part and i > 0:
                        stats["passed"] = int(parts[i-1])
                    elif "failed" in part and i > 0:
                        stats["failed"] = int(parts[i-1])
                    elif "skipped" in part and i > 0:
                        stats["skipped"] = int(parts[i-1])
                    elif "error" in part and i > 0:
                        stats["errors"] = int(parts[i-1])
                        
        return stats
        
    def _parse_locust_output(self, output: str) -> Dict:
        """Parse Locust output for statistics."""
        stats = {
            "requests": 0,
            "failures": 0,
            "median_latency": 0,
            "p95_latency": 0,
            "p99_latency": 0,
            "rps": 0
        }
        
        # Parse statistics from output
        lines = output.split('\n')
        for line in lines:
            if "Median response time" in line:
                stats["median_latency"] = self._extract_number(line)
            elif "95%ile" in line:
                stats["p95_latency"] = self._extract_number(line)
            elif "99%ile" in line:
                stats["p99_latency"] = self._extract_number(line)
            elif "Total Requests" in line:
                stats["requests"] = self._extract_number(line)
            elif "Failed requests" in line:
                stats["failures"] = self._extract_number(line)
            elif "RPS" in line or "Requests/s" in line:
                stats["rps"] = self._extract_number(line)
                
        return stats
        
    def _extract_number(self, text: str) -> float:
        """Extract number from text."""
        import re
        numbers = re.findall(r'[\d.]+', text)
        return float(numbers[0]) if numbers else 0
        
    def run_all(self, categories: Optional[List[str]] = None) -> bool:
        """Run all specified test categories."""
        if categories is None:
            categories = list(self.config.TEST_CATEGORIES.keys())
            
        self.start_time = datetime.now()
        all_success = True
        
        for category in categories:
            success, stats = self.run_category(category)
            self.results[category] = stats
            all_success = all_success and success
            
            # Print immediate results
            self._print_category_results(category, stats)
            
        self.end_time = datetime.now()
        
        return all_success
        
    def _print_category_results(self, category: str, stats: Dict):
        """Print results for a test category."""
        print(f"\n{category.upper()} Results:")
        print(f"  Success: {'✓' if stats.get('success') else '✗'}")
        print(f"  Duration: {stats.get('duration', 0):.2f}s")
        
        if "passed" in stats:
            print(f"  Passed: {stats['passed']}")
            print(f"  Failed: {stats['failed']}")
            print(f"  Skipped: {stats['skipped']}")
            
        if "rps" in stats:
            print(f"  RPS: {stats['rps']:.2f}")
            print(f"  P99 Latency: {stats['p99_latency']:.2f}ms")
            
    def generate_report(self) -> Dict:
        """Generate comprehensive test report."""
        duration = (self.end_time - self.start_time).total_seconds()
        
        report = {
            "timestamp": self.start_time.isoformat(),
            "duration_seconds": duration,
            "categories": self.results,
            "summary": self._generate_summary(),
            "performance_validation": self._validate_performance(),
            "coverage_analysis": self._analyze_coverage(),
            "recommendations": self._generate_recommendations()
        }
        
        return report
        
    def _generate_summary(self) -> Dict:
        """Generate test summary."""
        total_passed = sum(r.get("passed", 0) for r in self.results.values())
        total_failed = sum(r.get("failed", 0) for r in self.results.values())
        total_tests = total_passed + total_failed
        
        return {
            "total_tests": total_tests,
            "total_passed": total_passed,
            "total_failed": total_failed,
            "success_rate": total_passed / total_tests if total_tests > 0 else 0,
            "categories_run": len(self.results),
            "all_passed": all(r.get("success", False) for r in self.results.values())
        }
        
    def _validate_performance(self) -> Dict:
        """Validate against performance targets."""
        validation = {}
        
        # Check load test results if available
        if "load" in self.results:
            load_stats = self.results["load"]
            
            validation["latency"] = {
                "target": self.config.PERFORMANCE_TARGETS["latency_p99_ms"],
                "actual": load_stats.get("p99_latency", float('inf')),
                "passed": load_stats.get("p99_latency", float('inf')) <= 
                         self.config.PERFORMANCE_TARGETS["latency_p99_ms"]
            }
            
            validation["throughput"] = {
                "target": self.config.PERFORMANCE_TARGETS["throughput_rps"],
                "actual": load_stats.get("rps", 0),
                "passed": load_stats.get("rps", 0) >= 
                         self.config.PERFORMANCE_TARGETS["throughput_rps"]
            }
            
        return validation
        
    def _analyze_coverage(self) -> Dict:
        """Analyze test coverage."""
        coverage = {}
        
        # Check for coverage reports
        coverage_files = list(Path("htmlcov").glob("*/index.html"))
        
        if coverage_files:
            # Parse coverage data (simplified)
            coverage["reports_available"] = len(coverage_files)
            coverage["targets_met"] = False  # Would need actual parsing
            
        return coverage
        
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on results."""
        recommendations = []
        
        # Check for failures
        for category, stats in self.results.items():
            if stats.get("failed", 0) > 0:
                recommendations.append(
                    f"Fix {stats['failed']} failing tests in {category} category"
                )
                
        # Check performance
        if "load" in self.results:
            load_stats = self.results["load"]
            if load_stats.get("p99_latency", 0) > 100:
                recommendations.append(
                    "Optimize routing algorithm to reduce P99 latency below 100ms"
                )
            if load_stats.get("rps", 0) < 1000:
                recommendations.append(
                    "Improve throughput to achieve 1000+ RPS target"
                )
                
        # Check chaos results
        if "chaos" in self.results:
            chaos_stats = self.results["chaos"]
            if chaos_stats.get("failed", 0) > 0:
                recommendations.append(
                    "Improve system resilience to handle chaos scenarios"
                )
                
        return recommendations


# ============================================================================
# REPORT GENERATOR
# ============================================================================

class ReportGenerator:
    """Generates test reports in various formats."""
    
    @staticmethod
    def generate_html_report(report: Dict) -> str:
        """Generate HTML test report."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Provider Router Test Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                h2 {{ color: #666; border-bottom: 2px solid #eee; padding-bottom: 5px; }}
                .success {{ color: green; }}
                .failure {{ color: red; }}
                .warning {{ color: orange; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .metric {{ font-weight: bold; }}
                .recommendation {{ background-color: #fff3cd; padding: 10px; margin: 10px 0; border-left: 4px solid #ffc107; }}
            </style>
        </head>
        <body>
            <h1>Provider Router Test Report</h1>
            <p>Generated: {report['timestamp']}</p>
            <p>Duration: {report['duration_seconds']:.2f} seconds</p>
            
            <h2>Summary</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td class="metric">Total Tests</td>
                    <td>{report['summary']['total_tests']}</td>
                </tr>
                <tr>
                    <td class="metric">Passed</td>
                    <td class="success">{report['summary']['total_passed']}</td>
                </tr>
                <tr>
                    <td class="metric">Failed</td>
                    <td class="failure">{report['summary']['total_failed']}</td>
                </tr>
                <tr>
                    <td class="metric">Success Rate</td>
                    <td>{report['summary']['success_rate']*100:.1f}%</td>
                </tr>
            </table>
            
            <h2>Category Results</h2>
            <table>
                <tr>
                    <th>Category</th>
                    <th>Status</th>
                    <th>Duration</th>
                    <th>Details</th>
                </tr>
        """
        
        for category, stats in report['categories'].items():
            status_class = "success" if stats.get('success') else "failure"
            status_text = "✓ PASSED" if stats.get('success') else "✗ FAILED"
            
            html += f"""
                <tr>
                    <td class="metric">{category.upper()}</td>
                    <td class="{status_class}">{status_text}</td>
                    <td>{stats.get('duration', 0):.2f}s</td>
                    <td>{ReportGenerator._format_stats(stats)}</td>
                </tr>
            """
            
        html += """
            </table>
            
            <h2>Performance Validation</h2>
        """
        
        if report.get('performance_validation'):
            html += "<table><tr><th>Metric</th><th>Target</th><th>Actual</th><th>Status</th></tr>"
            
            for metric, data in report['performance_validation'].items():
                status_class = "success" if data['passed'] else "failure"
                status_text = "✓" if data['passed'] else "✗"
                
                html += f"""
                    <tr>
                        <td class="metric">{metric.upper()}</td>
                        <td>{data['target']}</td>
                        <td>{data['actual']:.2f}</td>
                        <td class="{status_class}">{status_text}</td>
                    </tr>
                """
                
            html += "</table>"
            
        html += "<h2>Recommendations</h2>"
        
        for rec in report.get('recommendations', []):
            html += f'<div class="recommendation">{rec}</div>'
            
        html += """
        </body>
        </html>
        """
        
        return html
        
    @staticmethod
    def _format_stats(stats: Dict) -> str:
        """Format statistics for display."""
        parts = []
        
        if "passed" in stats:
            parts.append(f"Passed: {stats['passed']}")
        if "failed" in stats:
            parts.append(f"Failed: {stats['failed']}")
        if "rps" in stats:
            parts.append(f"RPS: {stats['rps']:.2f}")
        if "p99_latency" in stats:
            parts.append(f"P99: {stats['p99_latency']:.2f}ms")
            
        return ", ".join(parts)
        
    @staticmethod
    def generate_json_report(report: Dict) -> str:
        """Generate JSON test report."""
        return json.dumps(report, indent=2, default=str)
        
    @staticmethod
    def save_report(report: Dict, format: str = "html", filename: Optional[str] = None):
        """Save report to file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"test_report_{timestamp}.{format}"
            
        if format == "html":
            content = ReportGenerator.generate_html_report(report)
        elif format == "json":
            content = ReportGenerator.generate_json_report(report)
        else:
            raise ValueError(f"Unsupported format: {format}")
            
        with open(filename, "w") as f:
            f.write(content)
            
        print(f"\nReport saved to: {filename}")
        return filename


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main test runner entry point."""
    parser = argparse.ArgumentParser(description="Provider Router Test Runner")
    
    # Test selection
    parser.add_argument("--all", action="store_true", help="Run all test categories")
    parser.add_argument("--unit", action="store_true", help="Run unit tests")
    parser.add_argument("--integration", action="store_true", help="Run integration tests")
    parser.add_argument("--performance", action="store_true", help="Run performance tests")
    parser.add_argument("--chaos", action="store_true", help="Run chaos tests")
    parser.add_argument("--load", action="store_true", help="Run load tests")
    parser.add_argument("--property", action="store_true", help="Run property tests")
    
    # Options
    parser.add_argument("--report", action="store_true", help="Generate test report")
    parser.add_argument("--report-format", choices=["html", "json"], 
                       default="html", help="Report format")
    parser.add_argument("--coverage", action="store_true", help="Generate coverage report")
    parser.add_argument("--profile", action="store_true", help="Profile test execution")
    
    args = parser.parse_args()
    
    # Determine which categories to run
    categories = []
    if args.all:
        categories = ["unit", "integration", "performance", "chaos", "load", "property"]
    else:
        if args.unit:
            categories.append("unit")
        if args.integration:
            categories.append("integration")
        if args.performance:
            categories.append("performance")
        if args.chaos:
            categories.append("chaos")
        if args.load:
            categories.append("load")
        if args.property:
            categories.append("property")
            
    # Default to unit tests if nothing specified
    if not categories:
        categories = ["unit"]
        
    # Initialize runner
    config = TestConfig()
    runner = TestRunner(config)
    
    # Run tests
    print("\n" + "="*60)
    print("PROVIDER ROUTER TEST SUITE")
    print("="*60)
    print(f"Categories: {', '.join(categories)}")
    print("="*60)
    
    success = runner.run_all(categories)
    
    # Generate report if requested
    if args.report:
        report = runner.generate_report()
        filename = ReportGenerator.save_report(report, format=args.report_format)
        
        # Print summary
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        print(f"Total Tests: {report['summary']['total_tests']}")
        print(f"Passed: {report['summary']['total_passed']}")
        print(f"Failed: {report['summary']['total_failed']}")
        print(f"Success Rate: {report['summary']['success_rate']*100:.1f}%")
        print(f"Report: {filename}")
        
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()