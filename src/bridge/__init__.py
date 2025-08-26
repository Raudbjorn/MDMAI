"""MCP Bridge Service for Web UI Integration.

This module provides a bridge between web clients and stdio MCP servers,
enabling HTTP/WebSocket/SSE communication while maintaining MCP protocol compliance.
"""

from .bridge_server import BridgeServer
from .mcp_process_manager import MCPProcessManager
from .session_manager import BridgeSessionManager
from .protocol_translator import MCPProtocolTranslator

__all__ = [
    "BridgeServer",
    "MCPProcessManager", 
    "BridgeSessionManager",
    "MCPProtocolTranslator",
]

__version__ = "0.1.0"