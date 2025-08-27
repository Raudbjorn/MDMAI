"""Main entry point for the MCP Bridge Server."""

import asyncio
import os
import signal
import sys
from pathlib import Path

import uvicorn
from structlog import get_logger

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.logging_config import setup_logging
from src.bridge.bridge_server import create_bridge_app
from src.bridge.config import get_bridge_config, settings

# Setup logging
setup_logging(level=settings.log_level, log_file=settings.log_file)
logger = get_logger(__name__)


def main():
    """Main entry point."""
    try:
        # Get configuration
        config = get_bridge_config()
        
        logger.info(
            "Starting MCP Bridge Server",
            host=settings.host,
            port=settings.port,
            max_processes=config.max_processes,
            websocket_enabled=config.enable_websocket,
            sse_enabled=config.enable_sse,
            auth_required=config.require_auth,
        )
        
        # Create application
        app = create_bridge_app(config)
        
        # Configure uvicorn
        uvicorn_config = uvicorn.Config(
            app,
            host=settings.host,
            port=settings.port,
            workers=settings.workers,
            log_level=settings.log_level.lower(),
            access_log=settings.log_requests,
            use_colors=True,
            reload=os.getenv("BRIDGE_RELOAD", "false").lower() == "true",
        )
        
        # Create and run server
        server = uvicorn.Server(uvicorn_config)
        
        # Setup signal handlers
        loop = asyncio.new_event_loop()
        
        def handle_signal(sig, frame):
            logger.info(f"Received signal {sig}, shutting down...")
            loop.stop()
        
        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)
        
        # Run server
        asyncio.set_event_loop(loop)
        loop.run_until_complete(server.serve())
        
    except KeyboardInterrupt:
        logger.info("Bridge server interrupted by user")
    except Exception as e:
        logger.error("Bridge server failed to start", error=str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()