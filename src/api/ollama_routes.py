"""FastAPI router for Ollama model management with Result pattern."""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any, Dict, Generic, List, Optional, TypeVar, Union

import httpx
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from config.logging_config import get_logger

logger = get_logger(__name__)

# Result/Either pattern implementation
T = TypeVar('T')
E = TypeVar('E')


@dataclass(frozen=True)
class Result(Generic[T, E]):
    """Result type for error handling without exceptions."""
    value: Optional[T] = None
    error: Optional[E] = None
    
    @classmethod
    def ok(cls, value: T) -> 'Result[T, E]':
        """Create successful result."""
        return cls(value=value)
    
    @classmethod
    def err(cls, error: E) -> 'Result[T, E]':
        """Create error result."""
        return cls(error=error)
    
    @property
    def is_ok(self) -> bool:
        """Check if result is successful."""
        return self.error is None and self.value is not None
    
    @property
    def is_err(self) -> bool:
        """Check if result contains error."""
        return self.error is not None
    
    def unwrap(self) -> T:
        """Get value or raise exception if error."""
        if self.is_err:
            raise ValueError(f"Unwrap called on error: {self.error}")
        return self.value
    
    def unwrap_or(self, default: T) -> T:
        """Get value or default if error."""
        return self.value if self.is_ok else default
    
    def map(self, func) -> 'Result':
        """Transform success value."""
        if self.is_ok:
            return Result.ok(func(self.value))
        return Result.err(self.error)
    
    def and_then(self, func) -> 'Result':
        """Chain operations that return Results."""
        if self.is_ok:
            return func(self.value)
        return Result.err(self.error)


class ModelType(StrEnum):
    """Types of Ollama models with string enum."""
    EMBEDDING = "embedding"
    TEXT_GENERATION = "text_generation"
    MULTIMODAL = "multimodal"
    UNKNOWN = "unknown"


class ServiceStatus(StrEnum):
    """Service status enumeration."""
    ONLINE = "online"
    OFFLINE = "offline"
    DEGRADED = "degraded"


class OllamaModel(BaseModel):
    """Model information from Ollama with validation."""
    name: str = Field(..., min_length=1, description="Model name")
    size: int = Field(..., ge=0, description="Model size in bytes")
    digest: str = Field(..., min_length=1, description="Model digest/hash")
    modified_at: datetime = Field(..., description="Last modification time")
    model_type: ModelType = Field(default=ModelType.UNKNOWN, description="Classified model type")
    dimension: Optional[int] = Field(default=None, ge=1, description="Embedding dimension if applicable")
    description: Optional[str] = Field(default=None, description="Model description")
    
    @field_validator('name')
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate model name format."""
        if not v or v.isspace():
            raise ValueError("Model name cannot be empty or whitespace")
        return v.strip().lower()
    
    @property
    def size_mb(self) -> float:
        """Get size in megabytes."""
        return round(self.size / (1024 * 1024), 2)
    
    @property
    def is_embedding_model(self) -> bool:
        """Check if this is an embedding model."""
        return self.model_type == ModelType.EMBEDDING


class OllamaStatus(BaseModel):
    """Ollama service status with enhanced information."""
    status: ServiceStatus = Field(..., description="Service status")
    api_url: str = Field(..., description="API base URL")
    version: Optional[str] = Field(default=None, description="Ollama version")
    models_count: int = Field(default=0, ge=0, description="Number of available models")
    response_time_ms: Optional[int] = Field(default=None, ge=0, description="API response time")
    
    @property
    def is_healthy(self) -> bool:
        """Check if service is healthy."""
        return self.status == ServiceStatus.ONLINE


class ModelSelectionRequest(BaseModel):
    """Request to select a model for embeddings with validation."""
    model_name: str = Field(..., min_length=1, description="Name of model to select")
    pull_if_missing: bool = Field(default=False, description="Whether to pull model if not available")
    
    @field_validator('model_name')
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        """Validate and normalize model name."""
        if not v or v.isspace():
            raise ValueError("Model name cannot be empty")
        return v.strip().lower()


class ModelSelectionResponse(BaseModel):
    """Response for model selection with detailed information."""
    success: bool = Field(..., description="Whether selection was successful")
    model_name: str = Field(..., description="Selected model name")
    dimension: Optional[int] = Field(default=None, ge=1, description="Model embedding dimension")
    message: str = Field(..., description="Status message")
    model_type: Optional[ModelType] = Field(default=None, description="Model type classification")
    size_mb: Optional[float] = Field(default=None, ge=0, description="Model size in MB")


@dataclass
class OllamaModelCache:
    """Type-safe cache for model list with TTL and metrics."""
    ttl_seconds: int = 60
    cache: Optional[List[OllamaModel]] = field(default=None, init=False)
    last_fetch: Optional[datetime] = field(default=None, init=False)
    hit_count: int = field(default=0, init=False)
    miss_count: int = field(default=0, init=False)
    
    @property
    def ttl(self) -> timedelta:
        """Get TTL as timedelta."""
        return timedelta(seconds=self.ttl_seconds)
    
    def is_valid(self) -> bool:
        """Check if cache is still valid."""
        if self.cache is None or self.last_fetch is None:
            return False
        return datetime.now() - self.last_fetch < self.ttl
    
    def get(self) -> Optional[List[OllamaModel]]:
        """Get cached models if valid with hit/miss tracking."""
        if self.is_valid():
            self.hit_count += 1
            return self.cache
        else:
            self.miss_count += 1
            return None
    
    def set(self, models: List[OllamaModel]) -> None:
        """Update cache with new models."""
        self.cache = models
        self.last_fetch = datetime.now()
    
    def invalidate(self) -> None:
        """Invalidate the cache."""
        self.cache = None
        self.last_fetch = None
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 0.0
    
    def get_stats(self) -> dict:
        """Get cache statistics."""
        return {
            "hits": self.hit_count,
            "misses": self.miss_count,
            "hit_rate": self.hit_rate,
            "is_valid": self.is_valid(),
            "last_fetch": self.last_fetch.isoformat() if self.last_fetch else None
        }


@dataclass
class ModelDimensions:
    """Model dimension mappings for embedding models."""
    dimensions: Dict[str, int] = field(default_factory=lambda: {
        "nomic-embed": 768,
        "nomic-embed-text": 768,
        "all-minilm": 384,
        "mxbai-embed-large": 1024,
        "bge-small": 384,
        "bge-base": 768,
        "bge-large": 1024,
        "e5-small": 384,
        "e5-base": 768,
        "e5-large": 1024,
    })
    
    def get_dimension(self, model_name: str) -> Optional[int]:
        """Get embedding dimension for a model with fuzzy matching."""
        model_lower = model_name.lower()
        
        # Direct match
        if model_lower in self.dimensions:
            return self.dimensions[model_lower]
        
        # Partial match for model families
        for key, dim in self.dimensions.items():
            if key in model_lower or model_lower in key:
                return dim
        
        # Pattern-based fallbacks
        if "large" in model_lower:
            return 1024
        elif "base" in model_lower or "medium" in model_lower:
            return 768
        elif "small" in model_lower or "mini" in model_lower:
            return 384
        
        return None


class OllamaRoutes:
    """Ollama API routes handler with Result pattern and async HTTP."""
    
    def __init__(self, base_url: str = "http://localhost:11434", cache_ttl: int = 60, timeout: float = 30.0):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.timeout = timeout
        self.model_cache = OllamaModelCache(cache_ttl)
        self.model_dimensions = ModelDimensions()
        self.current_model: Optional[OllamaModel] = None
        self.router = self._create_router()
    
    def _classify_model(self, model_name: str) -> ModelType:
        """Classify model type based on name patterns with improved logic."""
        name = model_name.lower()
        
        # Embedding model patterns (most specific first)
        embedding_patterns = [
            "embed", "bge", "e5", "gte", "nomic", "mxbai", "minilm",
            "sentence-transformer", "text-embedding", "instructor"
        ]
        if any(pattern in name for pattern in embedding_patterns):
            return ModelType.EMBEDDING
        
        # Multimodal model patterns
        multimodal_patterns = [
            "llava", "bakllava", "vision", "clip", "blip", "dalle",
            "imagen", "multimodal", "vlm"
        ]
        if any(pattern in name for pattern in multimodal_patterns):
            return ModelType.MULTIMODAL
        
        # Text generation model patterns
        text_patterns = [
            "llama", "mistral", "phi", "qwen", "gemma", "codellama",
            "falcon", "vicuna", "alpaca", "chatglm", "baichuan"
        ]
        if any(pattern in name for pattern in text_patterns):
            return ModelType.TEXT_GENERATION
        
        return ModelType.UNKNOWN
    
    async def _fetch_models(self) -> Result[List[OllamaModel], str]:
        """Fetch models from Ollama API using Result pattern."""
        cached_models = self.model_cache.get()
        if cached_models:
            logger.debug("Returning cached models", count=len(cached_models))
            return Result.ok(cached_models)
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{self.api_url}/tags")
                response.raise_for_status()
                
                data = response.json()
                models = []
                
                for model_data in data.get("models", []):
                    try:
                        model_name = model_data.get("name", "")
                        if not model_name:
                            continue
                        
                        model_type = self._classify_model(model_name)
                        dimension = None
                        
                        if model_type == ModelType.EMBEDDING:
                            base_name = model_name.split(":")[0]
                            dimension = self.model_dimensions.get_dimension(base_name)
                        
                        # Parse modified_at with error handling
                        modified_at_str = model_data.get("modified_at")
                        try:
                            modified_at = datetime.fromisoformat(
                                modified_at_str.replace("Z", "+00:00") if modified_at_str else datetime.now().isoformat()
                            )
                        except (ValueError, AttributeError):
                            modified_at = datetime.now()
                        
                        models.append(OllamaModel(
                            name=model_name,
                            size=model_data.get("size", 0),
                            digest=model_data.get("digest", ""),
                            modified_at=modified_at,
                            model_type=model_type,
                            dimension=dimension
                        ))
                    except Exception as model_err:
                        logger.warning(
                            "Failed to parse model data",
                            model_data=model_data,
                            error=str(model_err)
                        )
                        continue
                
                self.model_cache.set(models)
                logger.info(
                    "Successfully fetched models",
                    count=len(models),
                    cache_stats=self.model_cache.get_stats()
                )
                return Result.ok(models)
                
        except httpx.TimeoutException:
            error_msg = f"Timeout connecting to Ollama API at {self.api_url}"
            logger.error(error_msg)
            return Result.err(error_msg)
        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP error {e.response.status_code} from Ollama API"
            logger.error(error_msg, response=e.response.text)
            return Result.err(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error fetching models: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return Result.err(error_msg)
    
    async def _check_status(self) -> Result[OllamaStatus, str]:
        """Check Ollama service status with detailed timing."""
        start_time = datetime.now()
        
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.api_url}/tags")
                response_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
                
                if response.status_code == 200:
                    data = response.json()
                    status = OllamaStatus(
                        status=ServiceStatus.ONLINE,
                        api_url=self.base_url,
                        models_count=len(data.get("models", [])),
                        response_time_ms=response_time_ms
                    )
                    return Result.ok(status)
                else:
                    status = OllamaStatus(
                        status=ServiceStatus.DEGRADED,
                        api_url=self.base_url,
                        response_time_ms=response_time_ms
                    )
                    return Result.ok(status)
        
        except httpx.TimeoutException:
            error_msg = "Ollama API timeout"
            logger.warning(error_msg, url=self.api_url)
            status = OllamaStatus(status=ServiceStatus.OFFLINE, api_url=self.base_url)
            return Result.ok(status)  # Still return status, not an error
        except Exception as e:
            error_msg = f"Error checking Ollama status: {str(e)}"
            logger.error(error_msg, exc_info=True)
            status = OllamaStatus(status=ServiceStatus.OFFLINE, api_url=self.base_url)
            return Result.ok(status)  # Still return status, not an error
    
    async def _select_model(self, model_name: str, pull_if_missing: bool = False) -> Result[ModelSelectionResponse, str]:
        """Select a model for embeddings using Result pattern."""
        # Fetch available models
        models_result = await self._fetch_models()
        if models_result.is_err:
            return Result.err(f"Failed to fetch models: {models_result.error}")
        
        models = models_result.unwrap()
        model_names = [m.name for m in models]
        
        # Check if model exists
        if model_name not in model_names:
            if pull_if_missing:
                pull_result = await self._pull_model(model_name)
                if pull_result.is_err:
                    return Result.err(f"Failed to pull model: {pull_result.error}")
                
                # Refresh models after pull
                self.model_cache.invalidate()
                models_result = await self._fetch_models()
                if models_result.is_err:
                    return Result.err("Failed to refresh models after pull")
                models = models_result.unwrap()
            else:
                return Result.err(f"Model '{model_name}' not found. Available models: {', '.join(model_names[:5])}{'...' if len(model_names) > 5 else ''}")
        
        # Get model info
        model_info = next((m for m in models if m.name == model_name), None)
        
        if not model_info:
            # Create basic model info if still not found
            model_info = OllamaModel(
                name=model_name,
                size=0,
                digest="unknown",
                modified_at=datetime.now(),
                model_type=self._classify_model(model_name),
                dimension=self.model_dimensions.get_dimension(model_name)
            )
        
        # Store current model
        self.current_model = model_info
        
        response = ModelSelectionResponse(
            success=True,
            model_name=model_name,
            dimension=model_info.dimension,
            message=f"Successfully selected {model_name}",
            model_type=model_info.model_type,
            size_mb=model_info.size_mb if model_info.size > 0 else None
        )
        
        logger.info(
            "Model selected",
            model_name=model_name,
            model_type=model_info.model_type,
            dimension=model_info.dimension
        )
        
        return Result.ok(response)
    
    async def _pull_model(self, model_name: str) -> Result[dict, str]:
        """Pull a model from Ollama registry."""
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:  # 5 minute timeout for pulls
                response = await client.post(
                    f"{self.api_url}/pull",
                    json={"name": model_name}
                )
                response.raise_for_status()
                
                logger.info("Successfully pulled model", model_name=model_name)
                return Result.ok({"model_name": model_name, "status": "pulled"})
                
        except httpx.TimeoutException:
            return Result.err(f"Timeout pulling model '{model_name}' - operation may continue in background")
        except httpx.HTTPStatusError as e:
            return Result.err(f"HTTP error pulling model: {e.response.status_code} - {e.response.text}")
        except Exception as e:
            return Result.err(f"Unexpected error pulling model: {str(e)}")
    
    # This method is now handled by ModelDimensions class
    # Keeping for backward compatibility
    def _get_embedding_dimension(self, model_name: str) -> Optional[int]:
        """Get embedding dimension - delegates to ModelDimensions."""
        return self.model_dimensions.get_dimension(model_name)
    
    def _create_router(self) -> APIRouter:
        """Create FastAPI router with endpoints."""
        router = APIRouter(prefix="/api/ollama", tags=["ollama"])
        
        @router.get("/models", response_model=List[OllamaModel])
        async def list_models():
            """List all installed Ollama models with error handling."""
            result = await self._fetch_models()
            if result.is_err:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=result.error
                )
            return result.unwrap()
        
        @router.get("/status", response_model=OllamaStatus)
        async def check_status():
            """Check Ollama service status with detailed information."""
            result = await self._check_status()
            if result.is_err:
                # This shouldn't happen as _check_status always returns Ok with status info
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=result.error
                )
            return result.unwrap()
        
        @router.post("/select", response_model=ModelSelectionResponse)
        async def select_model(request: ModelSelectionRequest):
            """Select a model for embeddings with comprehensive validation."""
            result = await self._select_model(request.model_name, request.pull_if_missing)
            if result.is_err:
                # Determine appropriate HTTP status based on error type
                if "not found" in result.error.lower():
                    status_code = status.HTTP_404_NOT_FOUND
                elif "timeout" in result.error.lower() or "pull" in result.error.lower():
                    status_code = status.HTTP_408_REQUEST_TIMEOUT
                else:
                    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
                
                raise HTTPException(status_code=status_code, detail=result.error)
            return result.unwrap()
        
        @router.get("/current", response_model=Optional[OllamaModel])
        async def get_current_model():
            """Get information about the currently selected model."""
            if self.current_model:
                return self.current_model
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No model currently selected"
            )
        
        @router.post("/models/pull")
        async def pull_model(model_name: str = Field(..., description="Name of model to pull")):
            """Pull a new Ollama model with proper error handling."""
            result = await self._pull_model(model_name)
            if result.is_err:
                # Determine appropriate status code
                if "timeout" in result.error.lower():
                    status_code = status.HTTP_408_REQUEST_TIMEOUT
                else:
                    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
                
                raise HTTPException(status_code=status_code, detail=result.error)
            
            self.model_cache.invalidate()
            return {
                "success": True,
                "message": f"Successfully pulled {model_name}",
                "model_name": model_name
            }
        
        @router.delete("/cache")
        async def clear_cache():
            """Clear the model cache and return statistics."""
            cache_stats = self.model_cache.get_stats()
            self.model_cache.invalidate()
            return {
                "message": "Cache cleared successfully",
                "previous_stats": cache_stats
            }
        
        @router.get("/cache/stats")
        async def get_cache_stats():
            """Get cache performance statistics."""
            return self.model_cache.get_stats()
        
        return router


def create_ollama_router(
    base_url: str = "http://localhost:11434",
    cache_ttl: int = 60,
    timeout: float = 30.0
) -> APIRouter:
    """
    Factory function to create Ollama router with configuration.
    
    Args:
        base_url: Base URL for Ollama API
        cache_ttl: Cache TTL in seconds
        timeout: Request timeout in seconds
        
    Returns:
        FastAPI router with Ollama endpoints and Result pattern error handling
    """
    routes = OllamaRoutes(base_url=base_url, cache_ttl=cache_ttl, timeout=timeout)
    return routes.router