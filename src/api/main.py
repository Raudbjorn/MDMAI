"""Main FastAPI application for MDMAI API."""

import os
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Annotated, Any, Dict, List, Optional

import httpx
from fastapi import Depends, FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config.logging_config import get_logger
from src.api.ollama_routes import create_ollama_router
from src.api.pdf_routes import router as pdf_router

logger = get_logger(__name__)


class HealthStatus(str, Enum):
    """Health check status values."""
    HEALTHY = "healthy"
    STARTING = "starting"
    UNHEALTHY = "unhealthy"


class ServiceStatus(str, Enum):
    """Service status values."""
    ONLINE = "online"
    OFFLINE = "offline"
    DEGRADED = "degraded"


@dataclass
class AppConfig:
    """Application configuration."""
    ollama_base_url: str = "http://localhost:11434"
    cors_origins: List[str] = field(default_factory=lambda: [
        "http://localhost:3000", "http://localhost:5173",
        "http://127.0.0.1:3000", "http://127.0.0.1:5173",
        "tauri://localhost", "https://tauri.localhost"
    ])
    request_timeout: float = 5.0
    cache_ttl: int = 60
    
    @classmethod
    def from_env(cls) -> "AppConfig":
        """Create config from environment variables."""
        env_origins = os.getenv("CORS_ALLOWED_ORIGINS")
        cors_origins = (
            [origin.strip() for origin in env_origins.split(",")]
            if env_origins
            else cls.__dataclass_fields__["cors_origins"].default_factory()
        )
        
        return cls(
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", cls.__dataclass_fields__["ollama_base_url"].default),
            cors_origins=cors_origins,
            request_timeout=float(os.getenv("REQUEST_TIMEOUT", cls.__dataclass_fields__["request_timeout"].default)),
            cache_ttl=int(os.getenv("CACHE_TTL", cls.__dataclass_fields__["cache_ttl"].default)),
        )


@dataclass
class ServiceHealth:
    """Service health information."""
    status: HealthStatus
    ollama_status: ServiceStatus
    uptime: Optional[float] = None
    version: str = "0.1.0"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON response."""
        result = {
            "status": self.status,
            "version": self.version,
            "services": {
                "ollama": self.ollama_status
            }
        }
        if self.uptime is not None:
            result["uptime_seconds"] = self.uptime
        return result


class MDMAIApp:
    """Main MDMAI API application with dependency injection."""
    
    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or AppConfig.from_env()
        self._startup_complete = False
        self._startup_time: Optional[float] = None
    
    async def startup(self) -> None:
        """Initialize application and check service dependencies."""
        import time
        
        self._startup_time = time.time()
        logger.info("Starting MDMAI API", config=self.config)
        
        ollama_status = await self._check_ollama_health()
        logger.info(
            "Ollama service status",
            status=ollama_status,
            url=self.config.ollama_base_url
        )
        
        self._startup_complete = True
        logger.info("MDMAI API startup complete")
    
    async def _check_ollama_health(self) -> ServiceStatus:
        """Check Ollama service health."""
        try:
            async with httpx.AsyncClient(timeout=self.config.request_timeout) as client:
                response = await client.get(f"{self.config.ollama_base_url}/api/tags")
                return ServiceStatus.ONLINE if response.status_code == 200 else ServiceStatus.DEGRADED
        except Exception as e:
            logger.warning("Ollama service check failed", error=str(e))
            return ServiceStatus.OFFLINE
    
    async def shutdown(self) -> None:
        """Graceful shutdown with cleanup."""
        logger.info("Shutting down MDMAI API")
        self._startup_complete = False
        self._startup_time = None
    
    def create_app(self) -> FastAPI:
        """Create and configure the FastAPI application with dependency injection."""
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            await self.startup()
            yield
            await self.shutdown()
        
        app = FastAPI(
            title="MDMAI API",
            description="API for MD&D AI System with Ollama Integration",
            version="0.1.0",
            lifespan=lifespan,
            docs_url="/docs",
            redoc_url="/redoc",
            openapi_tags=[
                {"name": "system", "description": "System health and info"},
                {"name": "ollama", "description": "Ollama model management"},
                {"name": "pdf", "description": "PDF processing operations"},
            ]
        )
        
        # Configure CORS with improved security
        self._configure_cors(app)
        
        # Register system endpoints with dependency injection
        self._register_system_endpoints(app)
        
        # Include feature routers
        app.include_router(
            create_ollama_router(self.config.ollama_base_url, cache_ttl=self.config.cache_ttl)
        )
        app.include_router(pdf_router)
        
        # Register exception handlers
        self._register_exception_handlers(app)
        
        logger.info("API configured successfully", config=self.config)
        return app
    
    def _configure_cors(self, app: FastAPI) -> None:
        """Configure CORS middleware with security considerations."""
        allow_credentials = all(origin != "*" for origin in self.config.cors_origins)
        
        app.add_middleware(
            CORSMiddleware,
            allow_origins=self.config.cors_origins,
            allow_credentials=allow_credentials,
            allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            allow_headers=[
                "Accept", "Accept-Language", "Content-Language",
                "Content-Type", "Authorization", "X-Requested-With"
            ],
            expose_headers=["X-Request-ID"],
        )
        logger.info("CORS configured", origins=self.config.cors_origins, credentials=allow_credentials)
    
    def _register_system_endpoints(self, app: FastAPI) -> None:
        """Register system health and information endpoints."""
        
        @app.get("/health", response_model=None, tags=["system"])
        async def health_check(app_instance: Annotated[MDMAIApp, Depends(lambda: self)]) -> dict:
            """Comprehensive health check with service dependencies."""
            health = await app_instance.get_health_status()
            return health.to_dict()
        
        @app.get("/", tags=["system"])
        async def root() -> dict:
            """API root with navigation links."""
            return {
                "service": "MDMAI API",
                "version": "0.1.0",
                "docs": "/docs",
                "health": "/health",
                "endpoints": {
                    "ollama": "/api/ollama",
                    "pdf": "/api/pdf"
                }
            }
    
    def _register_exception_handlers(self, app: FastAPI) -> None:
        """Register global exception handlers with structured error responses."""
        
        @app.exception_handler(ValueError)
        async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
            logger.warning(
                "Validation error",
                path=request.url.path,
                method=request.method,
                error=str(exc)
            )
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "error": "validation_error",
                    "message": str(exc),
                    "path": request.url.path
                }
            )
        
        @app.exception_handler(httpx.TimeoutException)
        async def timeout_handler(request: Request, exc: httpx.TimeoutException) -> JSONResponse:
            logger.error(
                "Request timeout",
                path=request.url.path,
                method=request.method,
                error=str(exc)
            )
            return JSONResponse(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                content={
                    "error": "timeout_error",
                    "message": "Service request timed out",
                    "path": request.url.path
                }
            )
        
        @app.exception_handler(Exception)
        async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
            logger.error(
                "Unhandled exception",
                path=request.url.path,
                method=request.method,
                error=str(exc),
                exc_info=True
            )
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "error": "internal_error",
                    "message": "An internal error occurred",
                    "path": request.url.path
                }
            )
    
    async def get_health_status(self) -> ServiceHealth:
        """Get comprehensive application health status."""
        import time
        
        if not self._startup_complete:
            return ServiceHealth(
                status=HealthStatus.STARTING,
                ollama_status=ServiceStatus.OFFLINE
            )
        
        ollama_status = await self._check_ollama_health()
        status = HealthStatus.HEALTHY if ollama_status == ServiceStatus.ONLINE else HealthStatus.DEGRADED
        
        uptime = time.time() - self._startup_time if self._startup_time else None
        
        return ServiceHealth(
            status=status,
            ollama_status=ollama_status,
            uptime=uptime
        )


def create_app(config: Optional[AppConfig] = None) -> FastAPI:
    """Factory function to create MDMAI API application with configuration."""
    app_config = config or AppConfig.from_env()
    return MDMAIApp(app_config).create_app()


# Default app instance for uvicorn with environment-based configuration
app = create_app()