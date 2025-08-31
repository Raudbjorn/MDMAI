"""FastAPI router for Ollama model management."""

import asyncio
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

import requests
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from config.logging_config import get_logger

logger = get_logger(__name__)


class ModelType(str, Enum):
    """Types of Ollama models."""
    
    EMBEDDING = "embedding"
    TEXT_GENERATION = "text_generation"
    MULTIMODAL = "multimodal"
    UNKNOWN = "unknown"


class OllamaModel(BaseModel):
    """Model information from Ollama."""
    name: str
    size: int
    digest: str
    modified_at: datetime
    model_type: ModelType = ModelType.UNKNOWN
    dimension: Optional[int] = None
    description: Optional[str] = None


class OllamaStatus(BaseModel):
    """Ollama service status."""
    is_running: bool
    version: Optional[str] = None
    models_count: int = 0
    api_url: str


class ModelSelectionRequest(BaseModel):
    """Request to select a model for embeddings."""
    model_name: str
    pull_if_missing: bool = False


class ModelSelectionResponse(BaseModel):
    """Response for model selection."""
    success: bool
    model_name: str
    dimension: Optional[int] = None
    message: str


class OllamaModelCache:
    """Simple cache for model list with TTL."""
    
    def __init__(self, ttl_seconds: int = 60):
        self.ttl = timedelta(seconds=ttl_seconds)
        self.cache: Optional[List[OllamaModel]] = None
        self.last_fetch: Optional[datetime] = None
    
    def is_valid(self) -> bool:
        """Check if cache is still valid."""
        if self.cache is None or self.last_fetch is None:
            return False
        return datetime.now() - self.last_fetch < self.ttl
    
    def get(self) -> Optional[List[OllamaModel]]:
        """Get cached models if valid."""
        return self.cache if self.is_valid() else None
    
    def set(self, models: List[OllamaModel]) -> None:
        """Update cache with new models."""
        self.cache = models
        self.last_fetch = datetime.now()
    
    def invalidate(self) -> None:
        """Invalidate the cache."""
        self.cache = None
        self.last_fetch = None


class OllamaRoutes:
    """Ollama API routes handler."""
    
    def __init__(self, base_url: str = "http://localhost:11434", cache_ttl: int = 60):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.model_cache = OllamaModelCache(ttl_seconds=cache_ttl)
        self.current_model: Optional[str] = None
        self.router = self._create_router()
    
    def _classify_model(self, model_name: str) -> ModelType:
        """Classify model type based on name."""
        name = model_name.lower()
        
        embedding_keywords = ["embed", "bge", "e5", "gte", "nomic", "mxbai", "minilm"]
        if any(kw in name for kw in embedding_keywords):
            return ModelType.EMBEDDING
            
        multimodal_keywords = ["llava", "bakllava", "vision", "clip"]
        if any(kw in name for kw in multimodal_keywords):
            return ModelType.MULTIMODAL
            
        text_keywords = ["llama", "mistral", "phi", "qwen", "gemma", "codellama"]
        if any(kw in name for kw in text_keywords):
            return ModelType.TEXT_GENERATION
            
        return ModelType.UNKNOWN
    
    async def _fetch_models(self) -> List[OllamaModel]:
        """Fetch models from Ollama API."""
        cached_models = self.model_cache.get()
        if cached_models:
            return cached_models
            
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None, lambda: requests.get(f"{self.api_url}/tags", timeout=10)
            )
            response.raise_for_status()
            
            data = response.json()
            models = []
            
            for model_data in data.get("models", []):
                model_name = model_data.get("name", "")
                model_type = self._classify_model(model_name)
                
                # Get dimension for embedding models (simplified)
                dimension = None
                if model_type == ModelType.EMBEDDING:
                    if "nomic-embed" in model_name:
                        dimension = 768
                    elif "all-minilm" in model_name:
                        dimension = 384
                
                models.append(OllamaModel(
                    name=model_name,
                    size=model_data.get("size", 0),
                    digest=model_data.get("digest", ""),
                    modified_at=datetime.fromisoformat(
                        model_data.get("modified_at", datetime.now().isoformat())
                    ),
                    model_type=model_type,
                    dimension=dimension
                ))
            
            self.model_cache.set(models)
            logger.info(f"Fetched {len(models)} models")
            return models
            
        except Exception as e:
            logger.error(f"Error fetching models: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Failed to fetch models: {str(e)}"
            )
    
    async def _check_status(self) -> OllamaStatus:
        """Check Ollama service status."""
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None, lambda: requests.get(f"{self.api_url}/tags", timeout=5)
            )
            
            if response.status_code == 200:
                data = response.json()
                return OllamaStatus(
                    is_running=True,
                    models_count=len(data.get("models", [])),
                    api_url=self.base_url
                )
            
        except Exception:
            pass
            
        return OllamaStatus(is_running=False, api_url=self.base_url)
    
    async def _select_model(self, model_name: str, pull_if_missing: bool = False) -> ModelSelectionResponse:
        """Select a model for embeddings."""
        try:
            # Check if model exists
            models = await self._fetch_models()
            model_names = [m.name for m in models]
            
            if model_name not in model_names:
                if pull_if_missing:
                    # Simplified pull logic
                    try:
                        await asyncio.get_event_loop().run_in_executor(
                            None, lambda: requests.post(f"{self.api_url}/pull", 
                                                       json={"name": model_name}, timeout=300)
                        )
                        self.model_cache.invalidate()
                    except Exception as e:
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"Failed to pull model: {str(e)}"
                        )
                else:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Model {model_name} not found"
                    )
            
            # Get model info
            model_info = next((m for m in models if m.name == model_name), None)
            dimension = model_info.dimension if model_info else None
            
            self.current_model = model_name
            
            return ModelSelectionResponse(
                success=True,
                model_name=model_name,
                dimension=dimension,
                message=f"Successfully selected {model_name}"
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error selecting model: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to select model: {str(e)}"
            )
    
    def _create_router(self) -> APIRouter:
        """Create FastAPI router with endpoints."""
        router = APIRouter(prefix="/api/ollama", tags=["ollama"])
        
        @router.get("/models", response_model=List[OllamaModel])
        async def list_models():
            """List all installed Ollama models."""
            return await self._fetch_models()
        
        @router.get("/status", response_model=OllamaStatus)
        async def check_status():
            """Check if Ollama service is running."""
            return await self._check_status()
        
        @router.post("/select", response_model=ModelSelectionResponse)
        async def select_model(request: ModelSelectionRequest):
            """Select a model for embeddings."""
            return await self._select_model(request.model_name, request.pull_if_missing)
        
        @router.get("/current")
        async def get_current_model():
            """Get information about the currently selected model."""
            if self.current_model:
                return {"model_name": self.current_model}
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No model currently selected"
            )
        
        @router.post("/models/pull")
        async def pull_model(model_name: str):
            """Pull a new Ollama model."""
            try:
                await asyncio.get_event_loop().run_in_executor(
                    None, lambda: requests.post(f"{self.api_url}/pull", 
                                               json={"name": model_name}, timeout=300)
                )
                self.model_cache.invalidate()
                return {"success": True, "message": f"Successfully pulled {model_name}"}
            except Exception as e:
                logger.error(f"Error pulling model: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to pull model: {str(e)}"
                )
        
        @router.delete("/cache")
        async def clear_cache():
            """Clear the model cache."""
            self.model_cache.invalidate()
            return {"message": "Cache cleared"}
        
        return router


def create_ollama_router(
    base_url: str = "http://localhost:11434",
    cache_ttl: int = 60
) -> APIRouter:
    """
    Create Ollama router instance.
    
    Args:
        base_url: Base URL for Ollama API
        cache_ttl: Cache TTL in seconds
        
    Returns:
        FastAPI router with Ollama endpoints
    """
    routes = OllamaRoutes(base_url=base_url, cache_ttl=cache_ttl)
    return routes.router