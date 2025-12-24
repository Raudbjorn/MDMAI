#!/usr/bin/env python
"""Production-ready MDMAI API server launcher with comprehensive configuration and monitoring."""

import argparse
import os
import signal
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add src to Python path
src_path = Path(__file__).parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Import after path setup
import uvicorn
from config.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ServerConfig:
    """Comprehensive server configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    workers: int = 1
    log_level: str = "info"
    access_log: bool = True
    app_module: str = "src.api.main:app"
    ssl_keyfile: Optional[str] = None
    ssl_certfile: Optional[str] = None
    max_workers: Optional[int] = None
    timeout_keep_alive: int = 5
    timeout_graceful_shutdown: int = 30
    limit_concurrency: Optional[int] = None
    limit_max_requests: Optional[int] = None
    backlog: int = 2048
    proxy_headers: bool = False
    forwarded_allow_ips: str = "127.0.0.1"
    
    # Environment-based configuration
    environment: str = field(default_factory=lambda: os.getenv("ENVIRONMENT", "development"))
    
    @classmethod
    def from_env(cls) -> "ServerConfig":
        """Create configuration from environment variables."""
        return cls(
            host=os.getenv("API_HOST", cls.host),
            port=int(os.getenv("API_PORT", cls.port)),
            reload=os.getenv("API_RELOAD", "false").lower() == "true",
            workers=int(os.getenv("API_WORKERS", cls.workers)),
            log_level=os.getenv("LOG_LEVEL", cls.log_level).lower(),
            access_log=os.getenv("ACCESS_LOG", "true").lower() == "true",
            ssl_keyfile=os.getenv("SSL_KEYFILE"),
            ssl_certfile=os.getenv("SSL_CERTFILE"),
            timeout_keep_alive=int(os.getenv("TIMEOUT_KEEP_ALIVE", cls.timeout_keep_alive)),
            timeout_graceful_shutdown=int(os.getenv("TIMEOUT_GRACEFUL_SHUTDOWN", cls.timeout_graceful_shutdown)),
            limit_concurrency=int(os.getenv("LIMIT_CONCURRENCY")) if os.getenv("LIMIT_CONCURRENCY") else None,
            limit_max_requests=int(os.getenv("LIMIT_MAX_REQUESTS")) if os.getenv("LIMIT_MAX_REQUESTS") else None,
            backlog=int(os.getenv("BACKLOG", cls.backlog)),
            proxy_headers=os.getenv("PROXY_HEADERS", "false").lower() == "true",
            forwarded_allow_ips=os.getenv("FORWARDED_ALLOW_IPS", cls.forwarded_allow_ips),
        )
    
    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.environment.lower() == "production"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.environment.lower() == "development"
    
    def to_uvicorn_kwargs(self) -> Dict[str, Any]:
        """Convert to uvicorn configuration dictionary."""
        config = {
            "app": self.app_module,
            "host": self.host,
            "port": self.port,
            "log_level": self.log_level,
            "access_log": self.access_log,
            "timeout_keep_alive": self.timeout_keep_alive,
            "timeout_graceful_shutdown": self.timeout_graceful_shutdown,
            "backlog": self.backlog,
            "proxy_headers": self.proxy_headers,
            "forwarded_allow_ips": self.forwarded_allow_ips,
        }
        
        # Add optional configurations
        if self.ssl_keyfile and self.ssl_certfile:
            config.update({
                "ssl_keyfile": self.ssl_keyfile,
                "ssl_certfile": self.ssl_certfile,
            })
        
        if self.limit_concurrency:
            config["limit_concurrency"] = self.limit_concurrency
        
        if self.limit_max_requests:
            config["limit_max_requests"] = self.limit_max_requests
        
        # Development vs production settings
        if self.is_development:
            config.update({
                "reload": self.reload,
                "reload_dirs": ["src", "config"] if self.reload else None,
            })
        else:
            config.update({
                "workers": self.workers,
                "reload": False,  # Never reload in production
            })
        
        return {k: v for k, v in config.items() if v is not None}


class APIServer:
    """Production-ready API server manager."""
    
    def __init__(self, config: ServerConfig):
        self.config = config
        self._server: Optional[uvicorn.Server] = None
        self._setup_signal_handlers()
    
    def _setup_signal_handlers(self) -> None:
        """Setup graceful shutdown signal handlers."""
        def signal_handler(signum, frame):
            logger.info("Received shutdown signal", signal=signum)
            if self._server:
                self._server.should_exit = True
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def run(self) -> None:
        """Run the API server with comprehensive logging and error handling."""
        try:
            self._log_startup_info()
            self._validate_configuration()
            
            # Create uvicorn configuration
            uvicorn_config = uvicorn.Config(**self.config.to_uvicorn_kwargs())
            self._server = uvicorn.Server(uvicorn_config)
            
            # Run the server
            logger.info("Starting MDMAI API server")
            self._server.run()
            
        except KeyboardInterrupt:
            logger.info("Server interrupted by user")
        except Exception as e:
            logger.error("Server startup failed", error=str(e), exc_info=True)
            sys.exit(1)
        finally:
            logger.info("Server shutdown complete")
    
    def _log_startup_info(self) -> None:
        """Log comprehensive startup information."""
        protocol = "https" if self.config.ssl_keyfile else "http"
        base_url = f"{protocol}://{self.config.host}:{self.config.port}"
        
        logger.info("MDMAI API Server Configuration", **{
            "environment": self.config.environment,
            "host": self.config.host,
            "port": self.config.port,
            "workers": self.config.workers if not self.config.is_development else "N/A (reload mode)",
            "reload": self.config.reload,
            "log_level": self.config.log_level,
            "ssl_enabled": bool(self.config.ssl_keyfile),
        })
        
        print("\n" + "="*70)
        print("🚀 MDMAI API Server Starting")
        print("="*70)
        print(f"📍 Base URL:      {base_url}")
        print(f"📚 API Docs:      {base_url}/docs")
        print(f"🔄 ReDoc:         {base_url}/redoc")
        print(f"❤️  Health:       {base_url}/health")
        print(f"🤖 Ollama API:    {base_url}/api/ollama")
        print(f"📄 PDF API:       {base_url}/api/pdf")
        print(f"🌍 Environment:   {self.config.environment}")
        print(f"🔧 Log Level:     {self.config.log_level}")
        if self.config.ssl_keyfile:
            print(f"🔒 SSL:           Enabled")
        print("="*70 + "\n")
    
    def _validate_configuration(self) -> None:
        """Validate server configuration."""
        # Validate port range
        if not (1 <= self.config.port <= 65535):
            raise ValueError(f"Port must be between 1 and 65535, got {self.config.port}")
        
        # Validate SSL configuration
        if bool(self.config.ssl_keyfile) != bool(self.config.ssl_certfile):
            raise ValueError("Both ssl_keyfile and ssl_certfile must be provided for SSL")
        
        if self.config.ssl_keyfile and not Path(self.config.ssl_keyfile).exists():
            raise ValueError(f"SSL keyfile not found: {self.config.ssl_keyfile}")
        
        if self.config.ssl_certfile and not Path(self.config.ssl_certfile).exists():
            raise ValueError(f"SSL certfile not found: {self.config.ssl_certfile}")
        
        # Validate workers in production
        if self.config.is_production and self.config.workers < 2:
            logger.warning("Running production server with only 1 worker - consider increasing for better performance")
        
        # Validate log level
        valid_log_levels = ["critical", "error", "warning", "info", "debug", "trace"]
        if self.config.log_level not in valid_log_levels:
            raise ValueError(f"Invalid log level: {self.config.log_level}. Must be one of {valid_log_levels}")


def create_arg_parser() -> argparse.ArgumentParser:
    """Create comprehensive argument parser."""
    parser = argparse.ArgumentParser(
        description="MDMAI API Server - AI-powered document processing with Ollama integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment Variables:
  API_HOST                 Server host (default: 0.0.0.0)
  API_PORT                 Server port (default: 8000)
  API_RELOAD               Enable auto-reload (default: false)
  API_WORKERS              Number of worker processes (default: 1)
  LOG_LEVEL                Logging level (default: info)
  ENVIRONMENT              Environment (development/production)
  SSL_KEYFILE              SSL private key file path
  SSL_CERTFILE             SSL certificate file path
  TIMEOUT_KEEP_ALIVE       Keep-alive timeout in seconds (default: 5)
  TIMEOUT_GRACEFUL_SHUTDOWN Graceful shutdown timeout (default: 30)
  LIMIT_CONCURRENCY        Maximum concurrent connections
  LIMIT_MAX_REQUESTS       Maximum requests per worker
  PROXY_HEADERS            Trust proxy headers (default: false)
  FORWARDED_ALLOW_IPS      Allowed IPs for forwarded headers

Examples:
  python run_api.py                    # Basic development server
  python run_api.py --reload           # Development with auto-reload
  python run_api.py --port 8080 --workers 4  # Production-like setup
  python run_api.py --ssl-cert cert.pem --ssl-key key.pem  # HTTPS server
"""
    )
    
    parser.add_argument(
        "--host",
        default=None,
        help="Host to bind the server to (default: from config/env)"
    )
    
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=None,
        help="Port to bind the server to (default: from config/env)"
    )
    
    parser.add_argument(
        "--reload", "-r",
        action="store_true",
        help="Enable auto-reload for development"
    )
    
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=None,
        help="Number of worker processes (default: from config/env)"
    )
    
    parser.add_argument(
        "--log-level", "-l",
        choices=["critical", "error", "warning", "info", "debug", "trace"],
        default=None,
        help="Logging level (default: from config/env)"
    )
    
    parser.add_argument(
        "--ssl-cert",
        help="SSL certificate file path"
    )
    
    parser.add_argument(
        "--ssl-key",
        help="SSL private key file path"
    )
    
    parser.add_argument(
        "--env",
        choices=["development", "production"],
        help="Environment mode (default: from config/env)"
    )
    
    parser.add_argument(
        "--no-access-log",
        action="store_true",
        help="Disable access logging"
    )
    
    parser.add_argument(
        "--proxy-headers",
        action="store_true",
        help="Trust X-Forwarded-* headers"
    )
    
    parser.add_argument(
        "--version", "-v",
        action="version",
        version="MDMAI API v0.1.0"
    )
    
    return parser


def main() -> None:
    """Main entry point with comprehensive error handling and configuration."""
    try:
        # Parse command line arguments
        parser = create_arg_parser()
        args = parser.parse_args()
        
        # Create base configuration from environment
        config = ServerConfig.from_env()
        
        # Override with command line arguments
        if args.host is not None:
            config.host = args.host
        if args.port is not None:
            config.port = args.port
        if args.reload:
            config.reload = True
        if args.workers is not None:
            config.workers = args.workers
        if args.log_level is not None:
            config.log_level = args.log_level
        if args.ssl_cert and args.ssl_key:
            config.ssl_certfile = args.ssl_cert
            config.ssl_keyfile = args.ssl_key
        if args.env is not None:
            config.environment = args.env
        if args.no_access_log:
            config.access_log = False
        if args.proxy_headers:
            config.proxy_headers = True
        
        # Validate src directory exists
        if not src_path.exists():
            logger.error("Source directory not found", path=str(src_path))
            print(f"Error: Source directory '{src_path}' not found.")
            print("Please run this script from the project root directory.")
            sys.exit(1)
        
        # Pre-flight checks
        try:
            # Test import of the main app
            from src.api.main import app
            logger.info("Application module loaded successfully")
        except ImportError as e:
            logger.error("Failed to import application module", error=str(e))
            print(f"Error: Failed to import application: {e}")
            print("Please ensure all dependencies are installed and the src directory is properly configured.")
            sys.exit(1)
        
        # Create and run server
        server = APIServer(config)
        server.run()
        
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.error("Server failed to start", error=str(e), exc_info=True)
        print(f"\n💥 Server failed to start: {e}")
        print("\nCheck the logs for more details.")
        sys.exit(1)


if __name__ == "__main__":
    main()