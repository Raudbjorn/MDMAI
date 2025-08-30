#!/bin/bash
# Simple script to run the MCP demo

echo "Starting TTRPG MCP Demo..."
echo "==============================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies if needed
echo "Checking dependencies..."
pip install -q fastapi uvicorn fastmcp 2>/dev/null || {
    echo "Installing required packages..."
    pip install fastapi uvicorn fastmcp
}

# Start the bridge server (which starts the MCP server)
echo ""
echo "Starting bridge server..."
echo "Access the demo at: http://localhost:8000"
echo "==============================="
echo ""

python src/bridge_server.py