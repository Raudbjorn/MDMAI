# TTRPG MCP Demo - Quick Start Guide

## Overview
This is a minimal viable demo that shows Claude (or any MCP client) using MCP tools through a web browser interface.

## What It Does
- Provides a simple MCP server with 3 demo tools:
  - `roll_dice` - Roll dice using RPG notation (e.g., "3d6+2")
  - `search_rules` - Search demo game rules
  - `get_character_stats` - Get demo character information
- Bridge server that translates WebSocket to MCP protocol
- Simple web interface to test the tools

## Quick Start

### Method 1: Using the run script
```bash
./run_demo.sh
```

### Method 2: Manual start
```bash
# Install dependencies (if not already installed)
pip install fastapi uvicorn fastmcp

# Start the bridge server
python src/bridge_server.py
```

## Access the Demo
Open your browser and go to: http://localhost:8000

## Using with Claude Desktop

1. Start the demo server using one of the methods above
2. Configure Claude Desktop to use the MCP server by adding to your Claude Desktop config:

```json
{
  "mcpServers": {
    "ttrpg-demo": {
      "command": "python",
      "args": ["src/mcp_server.py"],
      "cwd": "/absolute/path/to/MDMAI"
    }
  }
}
```

Replace `/absolute/path/to/MDMAI` with your actual project path. You can find it by running `pwd` in the MDMAI directory.

## Testing the Tools

### Through the Web Interface:
1. Click "Connect" to establish WebSocket connection
2. Click "List Available Tools" to see what's available
3. Try rolling dice with expressions like "3d6+2" or "1d20"
4. Search for rules with terms like "advantage" or "critical"

### Through Claude:
Once configured, you can ask Claude to:
- "Roll 3d6+2 for me"
- "Search for rules about advantage"
- "Get the character stats"

## Architecture

```
[Web Browser] <--WebSocket--> [Bridge Server] <--stdio--> [MCP Server]
                                (port 8000)                 (subprocess)
```

1. **MCP Server** (`src/mcp_server.py`): Implements MCP protocol and tools
2. **Bridge Server** (`src/bridge_server.py`): WebSocket-to-MCP translation
3. **Web Interface**: Simple HTML/JS client for testing

## Troubleshooting

### Port Already in Use
If port 8000 is already in use, edit `src/bridge_server.py` and change:
```python
uvicorn.run(app, host="0.0.0.0", port=8000)  # Change 8000 to another port
```

### MCP Server Not Starting
Check the debug log in the web interface or console output for errors.

### Tools Not Working
1. Make sure you're connected (status shows "Connected")
2. Check the debug log for error messages
3. Verify the MCP server is running (check console output)

## Next Steps

This is a minimal demo. For production use, you would want to add:
- Authentication and security
- Persistent storage (ChromaDB for vector search)
- More sophisticated tools
- Better error handling
- Production-grade WebSocket management
- Real game data instead of hardcoded demo data

## Files

- `src/mcp_server.py` - MCP server with demo tools
- `src/bridge_server.py` - WebSocket bridge server
- `run_demo.sh` - Startup script
- `DEMO_README.md` - This file

## Why This Matters

This demo proves the core concept: **Claude can use MCP tools through a web interface**. This is the foundation for building a full-featured TTRPG assistant that can:
- Process PDFs and search rules
- Manage campaigns and sessions
- Track initiative and combat
- Generate NPCs and encounters
- All accessible through both Claude Desktop and web browsers