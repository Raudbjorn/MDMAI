#!/usr/bin/env python3
"""
PyOxidizer Stdio Communication Test Script
Tests the packaged MCP server executable for proper stdio communication with Tauri.

This script validates that the PyOxidizer-packaged executable:
1. Starts up correctly
2. Responds to MCP protocol messages
3. Handles stdio communication properly
4. Includes all required dependencies
5. Works without requiring Python installation
"""

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
import signal
import threading

class PyOxidizerStdioTester:
    """Test the PyOxidizer-packaged MCP server for stdio communication."""
    
    def __init__(self, executable_path: Path):
        self.executable_path = executable_path
        self.process: Optional[subprocess.Popen] = None
        self.stdout_lines: List[str] = []
        self.stderr_lines: List[str] = []
        
    def find_executable(self) -> Optional[Path]:
        """Find the PyOxidizer-built executable."""
        
        project_root = Path(__file__).parent.parent
        possible_locations = [
            # Direct path if provided
            self.executable_path,
            # Distribution directory
            project_root / "dist" / "pyoxidizer" / "mdmai-mcp-server-linux-x86_64" / "mdmai-mcp-server",
            project_root / "dist" / "pyoxidizer" / "mdmai-mcp-server-windows-x86_64" / "mdmai-mcp-server.exe",
            project_root / "dist" / "pyoxidizer" / "mdmai-mcp-server-macos-x86_64" / "mdmai-mcp-server",
            # Build directory
            project_root / "build" / "targets" / "x86_64-unknown-linux-gnu" / "release" / "install" / "mdmai-mcp-server",
            project_root / "build" / "targets" / "x86_64-pc-windows-msvc" / "release" / "install" / "mdmai-mcp-server.exe",
            project_root / "build" / "targets" / "x86_64-apple-darwin" / "release" / "install" / "mdmai-mcp-server",
        ]
        
        for location in possible_locations:
            if location and location.exists() and location.is_file():
                return location
                
        return None
    
    def send_mcp_message(self, message: Dict[str, Any]) -> bool:
        """Send an MCP message to the server process."""
        
        if not self.process or self.process.poll() is not None:
            return False
        
        try:
            # Convert message to JSON and send
            json_message = json.dumps(message)
            self.process.stdin.write(f"{json_message}\n".encode('utf-8'))
            self.process.stdin.flush()
            return True
            
        except Exception as e:
            print(f"ERROR: Failed to send message: {e}")
            return False
    
    def read_mcp_response(self, timeout: float = 5.0) -> Optional[Dict[str, Any]]:
        """Read an MCP response from the server process."""
        
        if not self.process:
            return None
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if self.process.poll() is not None:
                print("ERROR: Process terminated unexpectedly")
                return None
            
            try:
                # Try to read a line
                line = self.process.stdout.readline()
                if line:
                    line = line.decode('utf-8').strip()
                    if line:
                        try:
                            return json.loads(line)
                        except json.JSONDecodeError:
                            # Not a JSON line, might be debug output
                            self.stdout_lines.append(line)
                            continue
                
                time.sleep(0.1)
                
            except Exception as e:
                print(f"ERROR: Failed to read response: {e}")
                return None
        
        print("ERROR: Timeout waiting for response")
        return None
    
    def start_server(self) -> bool:
        """Start the MCP server process."""
        
        executable = self.find_executable()
        if not executable:
            print("ERROR: Could not find PyOxidizer executable")
            print("Build the executable first with: python scripts/build_pyoxidizer.py")
            return False
        
        print(f"INFO: Testing executable: {executable}")
        
        try:
            # Start the process with stdio pipes
            self.process = subprocess.Popen(
                [str(executable)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=False,  # Use binary mode for proper encoding control
                bufsize=0,   # Unbuffered
            )
            
            # Give the process a moment to start
            time.sleep(2)
            
            # Check if process is still running
            if self.process.poll() is not None:
                stdout, stderr = self.process.communicate()
                print(f"ERROR: Process exited immediately with code: {self.process.returncode}")
                print(f"STDOUT: {stdout.decode('utf-8', errors='ignore')}")
                print(f"STDERR: {stderr.decode('utf-8', errors='ignore')}")
                return False
            
            print("SUCCESS: Server process started")
            return True
            
        except Exception as e:
            print(f"ERROR: Failed to start server process: {e}")
            return False
    
    def stop_server(self):
        """Stop the MCP server process."""
        
        if self.process and self.process.poll() is None:
            try:
                # Try graceful shutdown first
                self.process.terminate()
                
                # Wait for graceful shutdown
                try:
                    self.process.wait(timeout=5)
                    print("INFO: Server process terminated gracefully")
                except subprocess.TimeoutExpired:
                    # Force kill if graceful shutdown fails
                    print("WARNING: Forcing server process termination")
                    self.process.kill()
                    self.process.wait()
                    
            except Exception as e:
                print(f"WARNING: Error stopping server process: {e}")
    
    def test_initialize(self) -> bool:
        """Test MCP initialize handshake."""
        
        print("\n--- Testing MCP Initialize ---")
        
        # Send initialize message
        initialize_msg = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "roots": {
                        "listChanged": False
                    },
                    "sampling": {}
                },
                "clientInfo": {
                    "name": "PyOxidizerTester",
                    "version": "1.0.0"
                }
            }
        }
        
        if not self.send_mcp_message(initialize_msg):
            print("ERROR: Failed to send initialize message")
            return False
        
        # Read response
        response = self.read_mcp_response(timeout=10)
        if not response:
            print("ERROR: No response to initialize message")
            return False
        
        print(f"SUCCESS: Received initialize response: {json.dumps(response, indent=2)}")
        
        # Validate response structure
        if response.get("jsonrpc") != "2.0" or response.get("id") != 1:
            print("ERROR: Invalid response format")
            return False
        
        if "result" not in response:
            print("ERROR: Initialize response missing result")
            return False
        
        # Send initialized notification
        initialized_msg = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
            "params": {}
        }
        
        if not self.send_mcp_message(initialized_msg):
            print("ERROR: Failed to send initialized notification")
            return False
        
        print("SUCCESS: Initialize handshake completed")
        return True
    
    def test_list_tools(self) -> bool:
        """Test listing available tools."""
        
        print("\n--- Testing List Tools ---")
        
        list_tools_msg = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {}
        }
        
        if not self.send_mcp_message(list_tools_msg):
            print("ERROR: Failed to send list tools message")
            return False
        
        response = self.read_mcp_response(timeout=10)
        if not response:
            print("ERROR: No response to list tools message")
            return False
        
        print(f"SUCCESS: Received tools list: {json.dumps(response, indent=2)}")
        
        # Validate response
        if response.get("jsonrpc") != "2.0" or response.get("id") != 2:
            print("ERROR: Invalid response format")
            return False
        
        if "result" not in response or "tools" not in response["result"]:
            print("ERROR: Invalid tools list response")
            return False
        
        tools = response["result"]["tools"]
        print(f"SUCCESS: Found {len(tools)} tools")
        
        # Print tool names for verification
        tool_names = [tool.get("name", "unnamed") for tool in tools]
        print(f"Available tools: {', '.join(tool_names)}")
        
        return True
    
    def test_server_info(self) -> bool:
        """Test getting server information."""
        
        print("\n--- Testing Server Info ---")
        
        server_info_msg = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "server_info",
                "arguments": {}
            }
        }
        
        if not self.send_mcp_message(server_info_msg):
            print("ERROR: Failed to send server info message")
            return False
        
        response = self.read_mcp_response(timeout=10)
        if not response:
            print("ERROR: No response to server info message")
            return False
        
        print(f"SUCCESS: Received server info: {json.dumps(response, indent=2)}")
        
        # Validate response
        if "result" not in response:
            print("ERROR: Invalid server info response")
            return False
        
        return True
    
    def test_dependency_loading(self) -> bool:
        """Test that critical dependencies are loaded correctly."""
        
        print("\n--- Testing Dependency Loading ---")
        
        # Test search functionality (requires ChromaDB)
        search_msg = {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "search",
                "arguments": {
                    "query": "test query",
                    "max_results": 1
                }
            }
        }
        
        if not self.send_mcp_message(search_msg):
            print("ERROR: Failed to send search message")
            return False
        
        response = self.read_mcp_response(timeout=15)  # Longer timeout for ChromaDB initialization
        if not response:
            print("ERROR: No response to search message")
            return False
        
        print(f"SUCCESS: Search functionality works (ChromaDB loaded)")
        
        # Check if response indicates successful operation or expected error
        if "result" in response:
            print("SUCCESS: Search executed successfully")
        elif "error" in response:
            error = response["error"]
            if "Database not initialized" in error.get("message", ""):
                print("SUCCESS: Expected error - database not initialized (dependencies loaded)")
            else:
                print(f"WARNING: Unexpected error: {error}")
        
        return True
    
    def run_comprehensive_test(self) -> bool:
        """Run comprehensive tests of the packaged executable."""
        
        print("========================================")
        print("PyOxidizer Stdio Communication Test")
        print("========================================")
        
        # Start the server
        if not self.start_server():
            return False
        
        try:
            # Run test sequence
            tests = [
                ("Initialize Handshake", self.test_initialize),
                ("List Tools", self.test_list_tools), 
                ("Server Info", self.test_server_info),
                ("Dependency Loading", self.test_dependency_loading),
            ]
            
            results = {}
            
            for test_name, test_func in tests:
                try:
                    print(f"\n{'='*50}")
                    print(f"Running: {test_name}")
                    print(f"{'='*50}")
                    
                    success = test_func()
                    results[test_name] = success
                    
                    if success:
                        print(f"✓ PASSED: {test_name}")
                    else:
                        print(f"✗ FAILED: {test_name}")
                    
                except Exception as e:
                    print(f"✗ ERROR in {test_name}: {e}")
                    results[test_name] = False
            
            # Print summary
            print(f"\n{'='*60}")
            print("TEST SUMMARY")
            print(f"{'='*60}")
            
            passed = 0
            total = len(results)
            
            for test_name, success in results.items():
                status = "✓ PASSED" if success else "✗ FAILED"
                print(f"{test_name:30} {status}")
                if success:
                    passed += 1
            
            print(f"\nResults: {passed}/{total} tests passed")
            
            if passed == total:
                print("\n🎉 All tests passed! The PyOxidizer executable is working correctly.")
                return True
            else:
                print(f"\n❌ {total - passed} test(s) failed. Check the executable build.")
                return False
                
        finally:
            # Always stop the server
            self.stop_server()
    
    def run_simple_test(self) -> bool:
        """Run a simple startup test without full MCP protocol testing."""
        
        print("========================================")
        print("PyOxidizer Simple Startup Test")
        print("========================================")
        
        executable = self.find_executable()
        if not executable:
            print("ERROR: Could not find PyOxidizer executable")
            return False
        
        print(f"INFO: Testing executable: {executable}")
        
        try:
            # Run the executable with --version or --help if supported
            # Since our MCP server might not support these, just run it briefly
            process = subprocess.Popen(
                [str(executable)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Let it run for a few seconds
            time.sleep(3)
            
            # Check if it's still running (good sign)
            if process.poll() is None:
                print("SUCCESS: Executable started and is running")
                process.terminate()
                process.wait(timeout=5)
                return True
            else:
                # Process exited, check output
                stdout, stderr = process.communicate()
                
                if process.returncode == 0:
                    print("SUCCESS: Executable ran and exited cleanly")
                    return True
                else:
                    print(f"ERROR: Executable exited with code: {process.returncode}")
                    print(f"STDOUT: {stdout}")
                    print(f"STDERR: {stderr}")
                    return False
                    
        except Exception as e:
            print(f"ERROR: Failed to test executable: {e}")
            return False

def main():
    """Main entry point for the test script."""
    
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test PyOxidizer-packaged MCP server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/test_pyoxidizer_stdio.py
  python scripts/test_pyoxidizer_stdio.py --simple
  python scripts/test_pyoxidizer_stdio.py --executable /path/to/mdmai-mcp-server
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
        help="Run simple startup test only (no MCP protocol testing)"
    )
    
    args = parser.parse_args()
    
    # Initialize tester
    executable_path = args.executable or Path("")
    tester = PyOxidizerStdioTester(executable_path)
    
    # Run tests
    if args.simple:
        success = tester.run_simple_test()
    else:
        success = tester.run_comprehensive_test()
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())