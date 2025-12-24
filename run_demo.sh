#!/bin/bash
# Simple script to run the MCP demo
# Uses the unified build system for dependency management

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Starting TTRPG MCP Demo..."
echo "==============================="
echo ""

# Check if build.sh exists and use it for setup
if [ -f "./build.sh" ]; then
    # Check if dependencies are already installed
    if [ ! -d ".venv" ] && [ ! -d "venv" ]; then
        echo "Running initial setup via build.sh..."
        ./build.sh setup
    fi
fi

# Activate virtual environment (check both possible names)
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "No virtual environment found. Creating one..."
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
fi

# Ensure demo-specific dependencies are installed
pip install -q fastapi uvicorn fastmcp 2>/dev/null || pip install fastapi uvicorn fastmcp

# Start the bridge server (which starts the MCP server)
echo ""
echo "Starting bridge server..."
echo "Access the demo at: http://localhost:8000"
echo "==============================="
echo ""

python src/bridge_server.py
