#!/usr/bin/env python
"""Test FastMCP 2.11.3 compatibility"""

import sys
try:
    from fastmcp import FastMCP
    
    # Create a simple MCP server
    mcp = FastMCP("test-server")
    
    @mcp.tool()
    async def test_tool(message: str) -> str:
        """A simple test tool"""
        return f"Test response: {message}"
    
    print("✅ FastMCP 2.11.3 imported and initialized successfully")
    print(f"✅ Server name: {mcp.name}")
    print("✅ Tool decorator works")
    sys.exit(0)
    
except ImportError as e:
    print(f"❌ Failed to import FastMCP: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error initializing FastMCP: {e}")
    sys.exit(1)
