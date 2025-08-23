"""
Port-free IPC Bridge for TTRPG Assistant MCP Server

This module provides a bridge architecture that avoids TCP/UDP ports entirely,
using Protocol Buffers over stdio for control and Apache Arrow for data transfer.
"""

from .bridge_server import MCPBridge, MCPSession, ArrowDataManager
from .mcp_adapter import MCPProtobufAdapter

__all__ = [
    'MCPBridge',
    'MCPSession', 
    'ArrowDataManager',
    'MCPProtobufAdapter'
]

__version__ = '0.1.0'