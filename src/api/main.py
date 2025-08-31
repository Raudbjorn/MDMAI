"""Main FastAPI application for MDMAI API."""

import asyncio
import os
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config.logging_config import get_logger
from src.api.ollama_routes import create_ollama_router

logger = get_logger(__name__)


class MDMAIApp:
    """Main MDMAI API application."""
    
    def __init__(
        self,
        title: str = "MDMAI API",
        description: str = "API for MD&D AI System",
        version: str = "0.1.0",
        ollama_base_url: str = "http://localhost:11434"
    ):
        """
        Initialize MDMAI API application.
        
        Args:
            title: API title
            description: API description
            version: API version
            ollama_base_url: Base URL for Ollama service
        """
        self.title = title
        self.description = description
        self.version = version
        self.ollama_base_url = ollama_base_url
        self.app: Optional[FastAPI] = None
        self._startup_complete = False
    
    async def startup(self) -> None:
        """Perform startup tasks."""
        logger.info(f"Starting {self.title} v{self.version}")
        
        # Add any startup tasks here
        # For example, check if Ollama is running
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.ollama_base_url}/api/tags", timeout=5)
                if response.status_code == 200:
                    logger.info("Ollama service is running")
                else:
                    logger.warning("Ollama service not responding properly")
        except Exception as e:
            logger.warning(f"Ollama service not available: {e}")
        
        self._startup_complete = True
        logger.info("Startup complete")
    
    async def shutdown(self) -> None:
        """Perform shutdown tasks."""
        logger.info("Shutting down API")
        
        # Add any cleanup tasks here
        
        self._startup_complete = False
        logger.info("Shutdown complete")
    
    def create_app(self) -> FastAPI:
        """
        Create and configure the FastAPI application.
        
        Returns:
            Configured FastAPI application
        """
        if self.app is not None:
            return self.app
        
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            """Manage application lifecycle."""
            # Startup
            await self.startup()
            yield
            # Shutdown
            await self.shutdown()
        
        # Create FastAPI app
        self.app = FastAPI(
            title=self.title,
            description=self.description,
            version=self.version,
            lifespan=lifespan
        )
        
        # Configure CORS - use environment variable or defaults for development
        allow_credentials = all(origin.strip() != "*" for origin in allowed_origins)
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=allowed_origins,
            allow_credentials=allow_credentials,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Add health check endpoint
        @self.app.get("/health", tags=["system"])
        async def health_check():
            """Health check endpoint."""
            return {
                "status": "healthy" if self._startup_complete else "starting",
                "service": self.title,
                "version": self.version
            }
        
        # Add root endpoint
        @self.app.get("/", tags=["system"])
        async def root():
            """Root endpoint with API information."""
            return {
                "service": self.title,
                "version": self.version,
                "description": self.description,
                "docs": "/docs",
                "redoc": "/redoc",
                "health": "/health"
            }
        
        # Include routers
        self._include_routers()
        
        # Add exception handlers
        self._add_exception_handlers()
        
        return self.app
    
    def _get_cors_origins(self) -> List[str]:
        """
        Get CORS allowed origins from environment or use development defaults.
        
        Returns:
            List of allowed origins
        """
        # Check for environment variable first
        env_origins = os.getenv("CORS_ALLOWED_ORIGINS")
        if env_origins:
            return [origin.strip() for origin in env_origins.split(",")]
        
        # Development defaults - restrict to common local development ports
        return [
            "http://localhost:3000",      # SvelteKit dev server
            "http://localhost:5173",      # Vite dev server
            "http://127.0.0.1:3000",      # Alternative localhost
            "http://127.0.0.1:5173",      # Alternative localhost
            "tauri://localhost",          # Tauri desktop app
            "https://tauri.localhost"     # Tauri desktop app (secure)
        ]
    
    def _include_routers(self) -> None:
        """Include all API routers."""
        if self.app is None:
            return
        
        # Include Ollama router
        ollama_router = create_ollama_router(
            base_url=self.ollama_base_url,
            cache_ttl=60
        )
        self.app.include_router(ollama_router)
        
        # Include PDF router
        from src.api.pdf_routes import router as pdf_router
        self.app.include_router(pdf_router)
        
        logger.info("All routers included")
    
    def _add_exception_handlers(self) -> None:
        """Add global exception handlers."""
        if self.app is None:
            return
        
        @self.app.exception_handler(ValueError)
        async def value_error_handler(request, exc):
            """Handle ValueError exceptions."""
            logger.error(f"ValueError: {exc}")
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "error": "validation_error",
                    "message": str(exc)
                }
            )
        
        @self.app.exception_handler(Exception)
        async def general_exception_handler(request, exc):
            """Handle general exceptions."""
            logger.error(f"Unhandled exception: {exc}", exc_info=True)
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "error": "internal_error",
                    "message": "An internal error occurred"
                }
            )


def create_app(
    title: str = "MDMAI API",
    description: str = "API for MD&D AI System",
    version: str = "0.1.0",
    ollama_base_url: str = "http://localhost:11434"
) -> FastAPI:
    """
    Create MDMAI API application.
    
    Args:
        title: API title
        description: API description
        version: API version
        ollama_base_url: Base URL for Ollama service
        
    Returns:
        FastAPI application instance
    """
    api = MDMAIApp(
        title=title,
        description=description,
        version=version,
        ollama_base_url=ollama_base_url
    )
    return api.create_app()


# Create default app instance for uvicorn
app = create_app()


if __name__ == "__main__":
    """Run the API server."""
    import uvicorn
    
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )