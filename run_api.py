#!/usr/bin/env python
"""Script to run the MDMAI API server."""

import sys
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

if __name__ == "__main__":
    import uvicorn
    
    print("Starting MDMAI API Server...")
    print("Docs: http://localhost:8000/docs")
    print("Health: http://localhost:8000/health")
    print("Ollama endpoints: /api/ollama/models, /api/ollama/status")
    
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )