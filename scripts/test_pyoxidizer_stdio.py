#!/usr/bin/env python3
"""
PyOxidizer Stdio Communication Test Suite for MDMAI MCP Server.

This module provides a comprehensive testing framework for validating PyOxidizer-packaged
MCP server executables. It includes advanced testing patterns, performance monitoring,
and detailed diagnostic capabilities.

Features:
    - Comprehensive MCP protocol testing
    - Performance benchmarking and monitoring
    - Advanced error diagnostics and reporting
    - Concurrent test execution capabilities
    - Configurable test scenarios and timeouts
    - Detailed logging and result reporting
    - Type-safe implementation with full documentation
"""

import argparse
import asyncio
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)


class TestResult(Enum):
    """Test execution results."""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


class TestCategory(Enum):
    """Test categories for organization."""
    STARTUP = "startup"
    PROTOCOL = "protocol"
    FUNCTIONALITY = "functionality"
    PERFORMANCE = "performance"
    INTEGRATION = "integration"


@dataclass
class TestOutcome:
    """Detailed outcome of a test execution."""
    name: str
    category: TestCategory
    result: TestResult
    duration: float
    message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    error_trace: Optional[str] = None
    
    def __post_init__(self) -> None:
        """Initialize details dictionary if None."""
        if self.details is None:
            self.details = {}


@dataclass
class TestConfiguration:
    """Configuration for test execution."""
    executable_path: Optional[Path] = None
    timeout_default: int = 30
    timeout_long: int = 60
    timeout_startup: int = 10
    max_retries: int = 3
    concurrent_tests: bool = False
    performance_benchmarking: bool = False
    verbose_output: bool = False
    
    # Test selection
    skip_slow_tests: bool = False
    only_category: Optional[TestCategory] = None
    
    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.timeout_default <= 0 or self.timeout_long <= 0:
            raise ValueError("Timeouts must be positive")


class MCPMessage:
    """Helper class for creating MCP protocol messages."""
    
    @staticmethod
    def initialize(client_name: str = "PyOxidizerTester", version: str = "1.0.0") -> Dict[str, Any]:
        """Create MCP initialize message."""
        return {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "roots": {"listChanged": False},
                    "sampling": {}
                },
                "clientInfo": {
                    "name": client_name,
                    "version": version
                }
            }
        }
    
    @staticmethod
    def initialized() -> Dict[str, Any]:
        """Create MCP initialized notification."""
        return {
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
            "params": {}
        }
    
    @staticmethod
    def list_tools() -> Dict[str, Any]:
        """Create tools/list request."""
        return {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {}
        }
    
    @staticmethod
    def call_tool(tool_name: str, arguments: Dict[str, Any], message_id: int = 3) -> Dict[str, Any]:
        """Create tools/call request."""
        return {
            "jsonrpc": "2.0",
            "id": message_id,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            }
        }


class ProcessManager:
    """Advanced process management with monitoring and cleanup."""
    
    def __init__(self, config: TestConfiguration) -> None:
        """Initialize process manager.
        
        Args:
            config: Test configuration
        """
        self.config = config
        self.process: Optional[subprocess.Popen] = None
        self.stdout_buffer: List[str] = []
        self.stderr_buffer: List[str] = []
        self._monitoring_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()
    
    @contextmanager
    def managed_process(self, executable_path: Path):
        """Context manager for process lifecycle management."""
        try:
            if self._start_process(executable_path):
                yield self
            else:
                raise RuntimeError("Failed to start process")
        finally:
            self._cleanup_process()
    
    def _start_process(self, executable_path: Path) -> bool:
        """Start the MCP server process with monitoring."""
        logger.info(f"Starting process: {executable_path}")
        
        try:
            self.process = subprocess.Popen(
                [str(executable_path)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,  # Unbuffered for real-time communication
                text=False  # Binary mode for encoding control
            )
            
            # Start monitoring thread
            self._start_monitoring()
            
            # Wait for process to stabilize
            time.sleep(self.config.timeout_startup / 5)
            
            if self.process.poll() is not None:
                stdout, stderr = self.process.communicate()
                logger.error(f"Process exited immediately with code: {self.process.returncode}")
                logger.error(f"STDOUT: {stdout.decode('utf-8', errors='ignore')}")
                logger.error(f"STDERR: {stderr.decode('utf-8', errors='ignore')}")
                return False
            
            logger.info("Process started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start process: {e}")
            return False
    
    def _start_monitoring(self) -> None:
        """Start background monitoring of process output."""
        self._stop_monitoring.clear()
        self._monitoring_thread = threading.Thread(
            target=self._monitor_process_output,
            daemon=True
        )
        self._monitoring_thread.start()
    
    def _monitor_process_output(self) -> None:
        """Monitor process output in background thread."""
        if not self.process:
            return
        
        while not self._stop_monitoring.is_set() and self.process.poll() is None:
            try:
                # Non-blocking read with timeout
                if self.process.stderr:
                    # Read stderr for logging/debugging
                    import select
                    ready, _, _ = select.select([self.process.stderr], [], [], 0.1)
                    if ready:
                        line = self.process.stderr.readline()
                        if line:
                            decoded_line = line.decode('utf-8', errors='ignore').strip()
                            if decoded_line:
                                self.stderr_buffer.append(decoded_line)
                                if self.config.verbose_output:
                                    logger.debug(f"Process stderr: {decoded_line}")
                
                time.sleep(0.1)
                
            except Exception as e:
                logger.warning(f"Error monitoring process output: {e}")
                break
    
    def _cleanup_process(self) -> None:
        """Clean up process and monitoring resources."""
        logger.debug("Cleaning up process")
        
        # Stop monitoring
        self._stop_monitoring.set()
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self._monitoring_thread.join(timeout=1)
        
        # Terminate process
        if self.process and self.process.poll() is None:
            try:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                    logger.debug("Process terminated gracefully")
                except subprocess.TimeoutExpired:
                    logger.warning("Process did not terminate gracefully, forcing kill")
                    self.process.kill()
                    self.process.wait()
            except Exception as e:
                logger.error(f"Error terminating process: {e}")
    
    def send_message(self, message: Dict[str, Any]) -> bool:
        """Send JSON message to process stdin."""
        if not self.process or self.process.poll() is not None:
            return False
        
        try:
            json_message = json.dumps(message)
            message_bytes = f"{json_message}\n".encode('utf-8')
            
            self.process.stdin.write(message_bytes)
            self.process.stdin.flush()
            
            logger.debug(f"Sent message: {json_message}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            return False
    
    def receive_message(self, timeout: float = 5.0) -> Optional[Dict[str, Any]]:
        """Receive JSON message from process stdout with timeout."""
        if not self.process:
            return None
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if self.process.poll() is not None:
                logger.error("Process terminated unexpectedly")
                return None
            
            try:
                # Non-blocking read
                import select
                ready, _, _ = select.select([self.process.stdout], [], [], 0.1)
                
                if ready:
                    line = self.process.stdout.readline()
                    if line:
                        decoded_line = line.decode('utf-8', errors='ignore').strip()
                        if decoded_line:
                            try:
                                response = json.loads(decoded_line)
                                logger.debug(f"Received message: {decoded_line}")
                                return response
                            except json.JSONDecodeError:
                                # Not a JSON line, might be debug output
                                self.stdout_buffer.append(decoded_line)
                                if self.config.verbose_output:
                                    logger.debug(f"Non-JSON output: {decoded_line}")
                
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error receiving message: {e}")
                return None
        
        logger.warning(f"Timeout ({timeout}s) waiting for message")
        return None


class BaseTest(ABC):
    """Abstract base class for MCP server tests."""
    
    def __init__(self, config: TestConfiguration) -> None:
        """Initialize base test.
        
        Args:
            config: Test configuration
        """
        self.config = config
        self.name = self.__class__.__name__
        self.category = TestCategory.FUNCTIONALITY  # Default category
    
    @abstractmethod
    async def execute(self, process_manager: ProcessManager) -> TestOutcome:
        """Execute the test.
        
        Args:
            process_manager: Managed process instance
            
        Returns:
            Test outcome with results
        """
        pass
    
    def _create_outcome(
        self,
        result: TestResult,
        duration: float,
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        error_trace: Optional[str] = None
    ) -> TestOutcome:
        """Create a test outcome."""
        return TestOutcome(
            name=self.name,
            category=self.category,
            result=result,
            duration=duration,
            message=message,
            details=details,
            error_trace=error_trace
        )


class StartupTest(BaseTest):
    """Test basic process startup and stability."""
    
    def __init__(self, config: TestConfiguration) -> None:
        super().__init__(config)
        self.category = TestCategory.STARTUP
    
    async def execute(self, process_manager: ProcessManager) -> TestOutcome:
        """Test process startup."""
        start_time = time.time()
        
        try:
            # Process should already be started by process_manager
            if not process_manager.process or process_manager.process.poll() is not None:
                return self._create_outcome(
                    TestResult.FAILED,
                    time.time() - start_time,
                    "Process not running or exited immediately"
                )
            
            # Wait for stability
            await asyncio.sleep(2)
            
            if process_manager.process.poll() is not None:
                return self._create_outcome(
                    TestResult.FAILED,
                    time.time() - start_time,
                    f"Process exited with code: {process_manager.process.returncode}"
                )
            
            return self._create_outcome(
                TestResult.PASSED,
                time.time() - start_time,
                "Process started and running stably"
            )
            
        except Exception as e:
            return self._create_outcome(
                TestResult.ERROR,
                time.time() - start_time,
                f"Startup test error: {e}",
                error_trace=str(e)
            )


class InitializeHandshakeTest(BaseTest):
    """Test MCP initialize handshake protocol."""
    
    def __init__(self, config: TestConfiguration) -> None:
        super().__init__(config)
        self.category = TestCategory.PROTOCOL
    
    async def execute(self, process_manager: ProcessManager) -> TestOutcome:
        """Test MCP initialize handshake."""
        start_time = time.time()
        details = {}
        
        try:
            # Send initialize message
            initialize_msg = MCPMessage.initialize()
            if not process_manager.send_message(initialize_msg):
                return self._create_outcome(
                    TestResult.FAILED,
                    time.time() - start_time,
                    "Failed to send initialize message"
                )
            
            # Receive initialize response
            response = process_manager.receive_message(self.config.timeout_default)
            if not response:
                return self._create_outcome(
                    TestResult.FAILED,
                    time.time() - start_time,
                    "No response to initialize message"
                )
            
            details['initialize_response'] = response
            
            # Validate response
            if not self._validate_initialize_response(response):
                return self._create_outcome(
                    TestResult.FAILED,
                    time.time() - start_time,
                    "Invalid initialize response format",
                    details=details
                )
            
            # Send initialized notification
            initialized_msg = MCPMessage.initialized()
            if not process_manager.send_message(initialized_msg):
                return self._create_outcome(
                    TestResult.FAILED,
                    time.time() - start_time,
                    "Failed to send initialized notification",
                    details=details
                )
            
            return self._create_outcome(
                TestResult.PASSED,
                time.time() - start_time,
                "Initialize handshake completed successfully",
                details=details
            )
            
        except Exception as e:
            return self._create_outcome(
                TestResult.ERROR,
                time.time() - start_time,
                f"Initialize handshake error: {e}",
                details=details,
                error_trace=str(e)
            )
    
    def _validate_initialize_response(self, response: Dict[str, Any]) -> bool:
        """Validate initialize response format."""
        required_fields = ["jsonrpc", "id", "result"]
        return (
            all(field in response for field in required_fields) and
            response.get("jsonrpc") == "2.0" and
            response.get("id") == 1 and
            isinstance(response.get("result"), dict)
        )


class ListToolsTest(BaseTest):
    """Test tools/list functionality."""
    
    def __init__(self, config: TestConfiguration) -> None:
        super().__init__(config)
        self.category = TestCategory.FUNCTIONALITY
    
    async def execute(self, process_manager: ProcessManager) -> TestOutcome:
        """Test tools listing."""
        start_time = time.time()
        details = {}
        
        try:
            # Perform initialize handshake first
            handshake_test = InitializeHandshakeTest(self.config)
            handshake_result = await handshake_test.execute(process_manager)
            
            if handshake_result.result != TestResult.PASSED:
                return self._create_outcome(
                    TestResult.FAILED,
                    time.time() - start_time,
                    "Initialize handshake failed, cannot test tools"
                )
            
            # Send list tools message
            list_tools_msg = MCPMessage.list_tools()
            if not process_manager.send_message(list_tools_msg):
                return self._create_outcome(
                    TestResult.FAILED,
                    time.time() - start_time,
                    "Failed to send list tools message"
                )
            
            # Receive tools list response
            response = process_manager.receive_message(self.config.timeout_default)
            if not response:
                return self._create_outcome(
                    TestResult.FAILED,
                    time.time() - start_time,
                    "No response to list tools message"
                )
            
            details['tools_response'] = response
            
            # Validate response
            if not self._validate_tools_response(response):
                return self._create_outcome(
                    TestResult.FAILED,
                    time.time() - start_time,
                    "Invalid tools list response format",
                    details=details
                )
            
            tools = response["result"]["tools"]
            details['tool_count'] = len(tools)
            details['tool_names'] = [tool.get("name", "unnamed") for tool in tools]
            
            return self._create_outcome(
                TestResult.PASSED,
                time.time() - start_time,
                f"Found {len(tools)} tools",
                details=details
            )
            
        except Exception as e:
            return self._create_outcome(
                TestResult.ERROR,
                time.time() - start_time,
                f"List tools error: {e}",
                details=details,
                error_trace=str(e)
            )
    
    def _validate_tools_response(self, response: Dict[str, Any]) -> bool:
        """Validate tools list response format."""
        return (
            response.get("jsonrpc") == "2.0" and
            response.get("id") == 2 and
            "result" in response and
            "tools" in response["result"] and
            isinstance(response["result"]["tools"], list)
        )


class PerformanceTest(BaseTest):
    """Test performance characteristics."""
    
    def __init__(self, config: TestConfiguration) -> None:
        super().__init__(config)
        self.category = TestCategory.PERFORMANCE
    
    async def execute(self, process_manager: ProcessManager) -> TestOutcome:
        """Test performance metrics."""
        start_time = time.time()
        details = {}
        
        if not self.config.performance_benchmarking:
            return self._create_outcome(
                TestResult.SKIPPED,
                time.time() - start_time,
                "Performance benchmarking disabled"
            )
        
        try:
            # Measure initialization time
            init_start = time.time()
            handshake_test = InitializeHandshakeTest(self.config)
            handshake_result = await handshake_test.execute(process_manager)
            init_time = time.time() - init_start
            
            details['initialization_time'] = init_time
            
            if handshake_result.result != TestResult.PASSED:
                return self._create_outcome(
                    TestResult.FAILED,
                    time.time() - start_time,
                    "Cannot measure performance without successful initialization"
                )
            
            # Measure tools list time
            tools_start = time.time()
            list_tools_msg = MCPMessage.list_tools()
            process_manager.send_message(list_tools_msg)
            response = process_manager.receive_message(self.config.timeout_default)
            tools_time = time.time() - tools_start
            
            details['tools_list_time'] = tools_time
            
            # Performance thresholds (configurable)
            init_threshold = 5.0  # seconds
            tools_threshold = 2.0  # seconds
            
            performance_issues = []
            if init_time > init_threshold:
                performance_issues.append(f"Slow initialization: {init_time:.2f}s > {init_threshold}s")
            if tools_time > tools_threshold:
                performance_issues.append(f"Slow tools listing: {tools_time:.2f}s > {tools_threshold}s")
            
            details['performance_issues'] = performance_issues
            
            if performance_issues:
                return self._create_outcome(
                    TestResult.FAILED,
                    time.time() - start_time,
                    f"Performance issues detected: {'; '.join(performance_issues)}",
                    details=details
                )
            
            return self._create_outcome(
                TestResult.PASSED,
                time.time() - start_time,
                f"Performance acceptable (init: {init_time:.2f}s, tools: {tools_time:.2f}s)",
                details=details
            )
            
        except Exception as e:
            return self._create_outcome(
                TestResult.ERROR,
                time.time() - start_time,
                f"Performance test error: {e}",
                details=details,
                error_trace=str(e)
            )


class ExecutableFinder:
    """Finds PyOxidizer-built executables."""
    
    @staticmethod
    def find_executable(project_root: Path, platform_hint: Optional[str] = None) -> Optional[Path]:
        """Find the PyOxidizer executable.
        
        Args:
            project_root: Project root directory
            platform_hint: Platform hint for executable location
            
        Returns:
            Path to executable if found, None otherwise
        """
        search_locations = ExecutableFinder._get_search_locations(project_root, platform_hint)
        
        for location in search_locations:
            if location.exists() and location.is_file():
                logger.info(f"Found executable: {location}")
                return location
        
        logger.error("Executable not found in search locations:")
        for location in search_locations:
            logger.error(f"  - {location}")
        
        return None
    
    @staticmethod
    def _get_search_locations(project_root: Path, platform_hint: Optional[str] = None) -> List[Path]:
        """Get list of potential executable locations."""
        locations = []
        
        # Platform-specific executable names
        if platform_hint == "windows":
            exe_name = "mdmai-mcp-server.exe"
            platform_patterns = ["windows", "pc-windows-msvc"]
        else:
            exe_name = "mdmai-mcp-server"
            platform_patterns = ["linux", "unknown-linux-gnu", "apple-darwin"]
        
        # Distribution directories
        dist_dir = project_root / "dist" / "pyoxidizer"
        for pattern in platform_patterns:
            locations.extend(dist_dir.glob(f"*{pattern}*/{exe_name}"))
        
        # Build directories
        build_dir = project_root / "build" / "targets"
        for pattern in platform_patterns:
            locations.extend(build_dir.glob(f"*{pattern}*/*/install/{exe_name}"))
            locations.extend(build_dir.glob(f"*{pattern}*/*/{exe_name}"))
        
        return locations


class TestSuite:
    """Comprehensive test suite for PyOxidizer MCP server."""
    
    def __init__(self, config: TestConfiguration) -> None:
        """Initialize test suite.
        
        Args:
            config: Test configuration
        """
        self.config = config
        self.tests: List[BaseTest] = self._create_test_instances()
    
    def _create_test_instances(self) -> List[BaseTest]:
        """Create test instances based on configuration."""
        all_tests = [
            StartupTest(self.config),
            InitializeHandshakeTest(self.config),
            ListToolsTest(self.config),
            PerformanceTest(self.config),
        ]
        
        # Filter tests based on configuration
        filtered_tests = []
        for test in all_tests:
            # Skip tests by category
            if self.config.only_category and test.category != self.config.only_category:
                continue
            
            # Skip slow tests if requested
            if self.config.skip_slow_tests and test.category == TestCategory.PERFORMANCE:
                continue
            
            filtered_tests.append(test)
        
        return filtered_tests
    
    async def run_tests(self, executable_path: Path) -> List[TestOutcome]:
        """Run all tests in the suite.
        
        Args:
            executable_path: Path to executable to test
            
        Returns:
            List of test outcomes
        """
        logger.info(f"Running {len(self.tests)} tests")
        logger.info(f"Testing executable: {executable_path}")
        
        if not self.config.concurrent_tests:
            return await self._run_sequential_tests(executable_path)
        else:
            return await self._run_concurrent_tests(executable_path)
    
    async def _run_sequential_tests(self, executable_path: Path) -> List[TestOutcome]:
        """Run tests sequentially."""
        outcomes = []
        
        process_manager = ProcessManager(self.config)
        
        try:
            with process_manager.managed_process(executable_path):
                for test in self.tests:
                    logger.info(f"Running test: {test.name}")
                    
                    try:
                        outcome = await test.execute(process_manager)
                        outcomes.append(outcome)
                        
                        # Log result
                        status_symbol = {
                            TestResult.PASSED: "✓",
                            TestResult.FAILED: "✗",
                            TestResult.SKIPPED: "⊝",
                            TestResult.ERROR: "⚠"
                        }.get(outcome.result, "?")
                        
                        logger.info(f"{status_symbol} {test.name}: {outcome.message}")
                        
                    except Exception as e:
                        logger.error(f"Unexpected error in test {test.name}: {e}")
                        outcomes.append(TestOutcome(
                            name=test.name,
                            category=test.category,
                            result=TestResult.ERROR,
                            duration=0.0,
                            message=f"Unexpected test error: {e}",
                            error_trace=str(e)
                        ))
        
        except Exception as e:
            logger.error(f"Failed to manage process: {e}")
            # Create error outcomes for all tests
            for test in self.tests:
                outcomes.append(TestOutcome(
                    name=test.name,
                    category=test.category,
                    result=TestResult.ERROR,
                    duration=0.0,
                    message=f"Process management error: {e}",
                    error_trace=str(e)
                ))
        
        return outcomes
    
    async def _run_concurrent_tests(self, executable_path: Path) -> List[TestOutcome]:
        """Run tests concurrently (future enhancement)."""
        # For now, fall back to sequential execution
        # Concurrent testing would require multiple process instances
        logger.warning("Concurrent testing not yet implemented, falling back to sequential")
        return await self._run_sequential_tests(executable_path)
    
    def generate_report(self, outcomes: List[TestOutcome]) -> str:
        """Generate a comprehensive test report.
        
        Args:
            outcomes: List of test outcomes
            
        Returns:
            Formatted test report
        """
        report_lines = [
            "=" * 80,
            "PyOxidizer MCP Server Test Report",
            "=" * 80,
        ]
        
        # Summary statistics
        total_tests = len(outcomes)
        passed = sum(1 for o in outcomes if o.result == TestResult.PASSED)
        failed = sum(1 for o in outcomes if o.result == TestResult.FAILED)
        errors = sum(1 for o in outcomes if o.result == TestResult.ERROR)
        skipped = sum(1 for o in outcomes if o.result == TestResult.SKIPPED)
        
        report_lines.extend([
            "",
            "SUMMARY:",
            f"  Total Tests: {total_tests}",
            f"  Passed:      {passed}",
            f"  Failed:      {failed}",
            f"  Errors:      {errors}",
            f"  Skipped:     {skipped}",
            "",
            f"Success Rate: {(passed/total_tests*100):.1f}%" if total_tests > 0 else "N/A",
        ])
        
        # Test details
        report_lines.extend([
            "",
            "DETAILED RESULTS:",
            ""
        ])
        
        for outcome in outcomes:
            status_symbol = {
                TestResult.PASSED: "✓",
                TestResult.FAILED: "✗",
                TestResult.SKIPPED: "⊝",
                TestResult.ERROR: "⚠"
            }.get(outcome.result, "?")
            
            report_lines.append(f"{status_symbol} {outcome.name} ({outcome.category.value})")
            report_lines.append(f"  Duration: {outcome.duration:.3f}s")
            report_lines.append(f"  Message:  {outcome.message or 'N/A'}")
            
            if outcome.details:
                report_lines.append("  Details:")
                for key, value in outcome.details.items():
                    report_lines.append(f"    {key}: {value}")
            
            if outcome.error_trace:
                report_lines.append(f"  Error: {outcome.error_trace}")
            
            report_lines.append("")
        
        return "\n".join(report_lines)


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(
        description="Test PyOxidizer-packaged MCP server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s
  %(prog)s --simple
  %(prog)s --executable /path/to/mdmai-mcp-server
  %(prog)s --performance --verbose
  %(prog)s --category startup
        """
    )
    
    parser.add_argument(
        "--executable", "-e",
        type=Path,
        help="Path to the PyOxidizer executable to test"
    )
    
    parser.add_argument(
        "--simple", "-s",
        action="store_true",
        help="Run only basic startup test"
    )
    
    parser.add_argument(
        "--performance", "-p",
        action="store_true",
        help="Enable performance benchmarking"
    )
    
    parser.add_argument(
        "--category", "-c",
        choices=[cat.value for cat in TestCategory],
        help="Run only tests from specific category"
    )
    
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Default timeout for tests in seconds"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--skip-slow",
        action="store_true",
        help="Skip slow/long-running tests"
    )
    
    return parser


async def main() -> int:
    """Main entry point for the test script.
    
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Find project root
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
        
        # Find executable
        executable_path = args.executable
        if not executable_path:
            executable_path = ExecutableFinder.find_executable(project_root)
            if not executable_path:
                logger.error("No executable found. Build with: python scripts/build_pyoxidizer.py")
                return 1
        
        if not executable_path.exists():
            logger.error(f"Executable not found: {executable_path}")
            return 1
        
        # Create test configuration
        config = TestConfiguration(
            executable_path=executable_path,
            timeout_default=args.timeout,
            performance_benchmarking=args.performance,
            verbose_output=args.verbose,
            skip_slow_tests=args.skip_slow,
            only_category=TestCategory(args.category) if args.category else None
        )
        
        # Handle simple mode
        if args.simple:
            config.only_category = TestCategory.STARTUP
        
        # Run tests
        logger.info("=" * 60)
        logger.info("PyOxidizer MCP Server Test Suite")
        logger.info("=" * 60)
        
        test_suite = TestSuite(config)
        outcomes = await test_suite.run_tests(executable_path)
        
        # Generate and display report
        report = test_suite.generate_report(outcomes)
        print(report)
        
        # Determine exit code
        total_tests = len(outcomes)
        passed_tests = sum(1 for o in outcomes if o.result == TestResult.PASSED)
        
        if total_tests == 0:
            logger.warning("No tests were executed")
            return 1
        
        if passed_tests == total_tests:
            logger.info("🎉 All tests passed!")
            return 0
        else:
            logger.error(f"❌ {total_tests - passed_tests} test(s) failed or had errors")
            return 1
            
    except KeyboardInterrupt:
        logger.info("Testing interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        if args.verbose:
            logger.error("Full traceback:", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))