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
    print("=" * 60)
    print("API Documentation will be available at:")
    print("  - Swagger UI: http://localhost:8000/docs")
    print("  - ReDoc: http://localhost:8000/redoc")
    print("  - Health check: http://localhost:8000/health")
    print("\nOllama endpoints:")
    print("  - GET  /api/ollama/models  - List installed models")
    print("  - GET  /api/ollama/status  - Check Ollama status")
    print("  - POST /api/ollama/select  - Select embedding model")
    print("  - GET  /api/ollama/current - Get current model info")
    print("=" * 60)
    
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )