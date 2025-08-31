"""Main FastAPI application for MDMAI API."""

import os
from contextlib import asynccontextmanager
from typing import List

import httpx
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config.logging_config import get_logger
from src.api.ollama_routes import create_ollama_router
from src.api.pdf_routes import router as pdf_router

logger = get_logger(__name__)


class MDMAIApp:
    """Main MDMAI API application."""
    
    def __init__(self, ollama_base_url: str = "http://localhost:11434"):
        self.ollama_base_url = ollama_base_url
        self._startup_complete = False
    
    async def startup(self) -> None:
        """Check Ollama service availability on startup."""
        logger.info("Starting MDMAI API")
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.ollama_base_url}/api/tags", timeout=5)
                logger.info("Ollama service is running" if response.status_code == 200 
                          else "Ollama service not responding properly")
        except Exception as e:
            logger.warning(f"Ollama service not available: {e}")
        self._startup_complete = True
    
    async def shutdown(self) -> None:
        """Clean shutdown."""
        logger.info("Shutting down MDMAI API")
        self._startup_complete = False
    
    def create_app(self) -> FastAPI:
        """Create and configure the FastAPI application."""
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            await self.startup()
            yield
            await self.shutdown()
        
        app = FastAPI(
            title="MDMAI API",
            description="API for MD&D AI System",
            version="0.1.0",
            lifespan=lifespan
        )
        
        # Configure CORS
        allowed_origins = self._get_cors_origins()
        app.add_middleware(
            CORSMiddleware,
            allow_origins=allowed_origins,
            allow_credentials=all(origin != "*" for origin in allowed_origins),
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # System endpoints
        @app.get("/health", tags=["system"])
        async def health_check():
            return {"status": "healthy" if self._startup_complete else "starting"}
        
        @app.get("/", tags=["system"])
        async def root():
            return {"service": "MDMAI API", "docs": "/docs", "health": "/health"}
        
        # Include routers
        app.include_router(create_ollama_router(self.ollama_base_url, cache_ttl=60))
        app.include_router(pdf_router)
        
        # Exception handlers
        @app.exception_handler(ValueError)
        async def value_error_handler(request: Request, exc: ValueError):
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"error": "validation_error", "message": str(exc)}
            )
        
        @app.exception_handler(Exception)
        async def general_exception_handler(request: Request, exc: Exception):
            logger.error(f"Unhandled exception: {exc}", exc_info=True)
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"error": "internal_error", "message": "An internal error occurred"}
            )
        
        logger.info("API configured successfully")
        return app
    
    def _get_cors_origins(self) -> List[str]:
        """Get CORS allowed origins from environment or defaults."""
        env_origins = os.getenv("CORS_ALLOWED_ORIGINS")
        if env_origins:
            return [origin.strip() for origin in env_origins.split(",")]
        
        return [
            "http://localhost:3000", "http://localhost:5173",
            "http://127.0.0.1:3000", "http://127.0.0.1:5173",
            "tauri://localhost", "https://tauri.localhost"
        ]


def create_app(ollama_base_url: str = "http://localhost:11434") -> FastAPI:
    """Create MDMAI API application."""
    return MDMAIApp(ollama_base_url).create_app()


# Default app instance for uvicorn
app = create_app()