"""FastAPI router for Ollama model management."""

import asyncio
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

import requests
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from returns.result import Failure, Result, Success

from config.logging_config import get_logger
from src.core.result_pattern import AppError, ErrorKind
from src.pdf_processing.ollama_provider import OllamaEmbeddingProvider

logger = get_logger(__name__)


class ModelType(str, Enum):
    """Types of Ollama models."""
    
    EMBEDDING = "embedding"
    TEXT_GENERATION = "text_generation"
    MULTIMODAL = "multimodal"
    UNKNOWN = "unknown"


class OllamaModel(BaseModel):
    """Model information from Ollama."""
    
    name: str = Field(..., description="Model name")
    size: int = Field(..., description="Model size in bytes")
    digest: str = Field(..., description="Model digest/hash")
    modified_at: datetime = Field(..., description="Last modification time")
    model_type: ModelType = Field(default=ModelType.UNKNOWN, description="Type of model")
    dimension: Optional[int] = Field(None, description="Embedding dimension if applicable")
    description: Optional[str] = Field(None, description="Model description")


class OllamaStatus(BaseModel):
    """Ollama service status."""
    
    is_running: bool = Field(..., description="Whether Ollama service is running")
    version: Optional[str] = Field(None, description="Ollama version")
    models_count: int = Field(0, description="Number of installed models")
    api_url: str = Field(..., description="Ollama API URL")


class ModelSelectionRequest(BaseModel):
    """Request to select a model for embeddings."""
    
    model_name: str = Field(..., description="Name of the model to select")
    pull_if_missing: bool = Field(False, description="Pull model if not installed")


class ModelSelectionResponse(BaseModel):
    """Response for model selection."""
    
    success: bool = Field(..., description="Whether selection was successful")
    model_name: str = Field(..., description="Selected model name")
    dimension: Optional[int] = Field(None, description="Embedding dimension")
    message: str = Field(..., description="Status message")


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
    
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        cache_ttl: int = 60
    ):
        """
        Initialize Ollama routes.
        
        Args:
            base_url: Base URL for Ollama API
            cache_ttl: Cache TTL in seconds
        """
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.model_cache = OllamaModelCache(ttl_seconds=cache_ttl)
        self.current_provider: Optional[OllamaEmbeddingProvider] = None
        self.router = self._create_router()
    
    def _classify_model(self, model_name: str, model_info: Dict[str, Any]) -> ModelType:
        """
        Classify model type based on name and metadata.
        
        Args:
            model_name: Name of the model
            model_info: Model information from Ollama
            
        Returns:
            Model type classification
        """
        name_lower = model_name.lower()
        
        # Check for embedding models
        embedding_keywords = [
            "embed", "embedding", "bge", "e5", "gte",
            "nomic", "mxbai", "minilm", "sentence"
        ]
        if any(keyword in name_lower for keyword in embedding_keywords):
            return ModelType.EMBEDDING
        
        # Check for multimodal models
        multimodal_keywords = ["llava", "bakllava", "vision", "clip"]
        if any(keyword in name_lower for keyword in multimodal_keywords):
            return ModelType.MULTIMODAL
        
        # Check for text generation models
        text_keywords = [
            "llama", "mistral", "mixtral", "phi", "qwen",
            "gemma", "codellama", "deepseek", "vicuna",
            "wizard", "neural", "openchat", "orca"
        ]
        if any(keyword in name_lower for keyword in text_keywords):
            return ModelType.TEXT_GENERATION
        
        return ModelType.UNKNOWN
    
    async def _fetch_models(self) -> Result[List[OllamaModel], AppError]:
        """
        Fetch models from Ollama API.
        
        Returns:
            Result with list of models or error
        """
        try:
            # Check cache first
            cached_models = self.model_cache.get()
            if cached_models is not None:
                logger.debug("Returning cached model list")
                return Success(cached_models)
            
            # Fetch from API
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: requests.get(f"{self.api_url}/tags", timeout=10)
            )
            
            if response.status_code != 200:
                return Failure(AppError(
                    kind=ErrorKind.NETWORK,
                    message=f"Failed to fetch models: HTTP {response.status_code}",
                    source="ollama_api"
                ))
            
            data = response.json()
            models = []
            
            for model_data in data.get("models", []):
                # Extract model information
                model_name = model_data.get("name", "")
                model_type = self._classify_model(model_name, model_data)
                
                # Get dimension for known embedding models
                dimension = None
                description = None
                if model_type == ModelType.EMBEDDING:
                    # Check if it's a known embedding model
                    base_name = model_name.split(":")[0]
                    if hasattr(OllamaEmbeddingProvider, "EMBEDDING_MODELS") and base_name in OllamaEmbeddingProvider.EMBEDDING_MODELS:
                        model_info = OllamaEmbeddingProvider.EMBEDDING_MODELS[base_name]
                        dimension = model_info.get("dimension")
                        description = model_info.get("description")
                
                model = OllamaModel(
                    name=model_name,
                    size=model_data.get("size", 0),
                    digest=model_data.get("digest", ""),
                    modified_at=datetime.fromisoformat(
                        model_data.get("modified_at", datetime.now().isoformat())
                    ),
                    model_type=model_type,
                    dimension=dimension,
                    description=description
                )
                models.append(model)
            
            # Update cache
            self.model_cache.set(models)
            logger.info(f"Fetched {len(models)} models from Ollama")
            
            return Success(models)
            
        except requests.RequestException as e:
            logger.error(f"Network error fetching models: {e}")
            return Failure(AppError(
                kind=ErrorKind.NETWORK,
                message=f"Network error: {str(e)}",
                source="ollama_api",
                recoverable=True
            ))
        except Exception as e:
            logger.error(f"Unexpected error fetching models: {e}")
            return Failure(AppError(
                kind=ErrorKind.SYSTEM,
                message=f"Unexpected error: {str(e)}",
                source="ollama_api",
                recoverable=False
            ))
    
    async def _check_status(self) -> Result[OllamaStatus, AppError]:
        """
        Check Ollama service status.
        
        Returns:
            Result with status or error
        """
        try:
            # Try to connect to Ollama API
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: requests.get(f"{self.api_url}/tags", timeout=5)
            )
            
            if response.status_code == 200:
                data = response.json()
                models_count = len(data.get("models", []))
                
                # Try to get version
                version = None
                try:
                    version_response = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: requests.get(f"{self.base_url}/api/version", timeout=3)
                    )
                    if version_response.status_code == 200:
                        version = version_response.json().get("version")
                except requests.RequestException:
                    pass  # Version endpoint might not exist
                
                return Success(OllamaStatus(
                    is_running=True,
                    version=version,
                    models_count=models_count,
                    api_url=self.base_url
                ))
            else:
                return Success(OllamaStatus(
                    is_running=False,
                    api_url=self.base_url
                ))
                
        except requests.RequestException:
            return Success(OllamaStatus(
                is_running=False,
                api_url=self.base_url
            ))
        except Exception as e:
            logger.error(f"Error checking Ollama status: {e}")
            return Failure(AppError(
                kind=ErrorKind.SYSTEM,
                message=f"Failed to check status: {str(e)}",
                source="ollama_status"
            ))
    
    async def _select_model(
        self,
        model_name: str,
        pull_if_missing: bool = False
    ) -> Result[ModelSelectionResponse, AppError]:
        """
        Select a model for embeddings.
        
        Args:
            model_name: Name of the model to select
            pull_if_missing: Whether to pull model if not installed
            
        Returns:
            Result with selection response or error
        """
        try:
            # Create provider instance
            provider = OllamaEmbeddingProvider(
                model_name=model_name,
                base_url=self.base_url
            )
            
            # Check if model is available
            if not provider.is_model_available(model_name):
                if pull_if_missing:
                    logger.info(f"Model {model_name} not found, attempting to pull...")
                    success = await asyncio.get_event_loop().run_in_executor(
                        None,
                        provider.pull_model,
                        model_name,
                        False  # Don't show progress in API
                    )
                    if not success:
                        return Failure(AppError(
                            kind=ErrorKind.NOT_FOUND,
                            message=f"Failed to pull model {model_name}",
                            source="ollama_provider"
                        ))
                else:
                    return Failure(AppError(
                        kind=ErrorKind.NOT_FOUND,
                        message=f"Model {model_name} not installed",
                        details={"available_models": provider.list_available_models()},
                        source="ollama_provider"
                    ))
            
            # Test the model by generating a sample embedding
            try:
                test_embedding = await asyncio.get_event_loop().run_in_executor(
                    None,
                    provider.generate_embedding,
                    "test"
                )
                dimension = len(test_embedding) if test_embedding else None
            except Exception as e:
                logger.warning(f"Could not test model {model_name}: {e}")
                dimension = provider.get_embedding_dimension()
            
            # Store the provider for future use
            self.current_provider = provider
            
            # Invalidate cache since we might have pulled a new model
            if pull_if_missing:
                self.model_cache.invalidate()
            
            return Success(ModelSelectionResponse(
                success=True,
                model_name=model_name,
                dimension=dimension,
                message=f"Successfully selected model {model_name}"
            ))
            
        except Exception as e:
            logger.error(f"Error selecting model: {e}")
            return Failure(AppError(
                kind=ErrorKind.SYSTEM,
                message=f"Failed to select model: {str(e)}",
                source="model_selection"
            ))
    
    def _create_router(self) -> APIRouter:
        """Create FastAPI router with endpoints."""
        router = APIRouter(prefix="/api/ollama", tags=["ollama"])
        
        @router.get(
            "/models",
            response_model=List[OllamaModel],
            summary="List installed Ollama models",
            description="Get a list of all installed Ollama models with caching for performance"
        )
        async def list_models():
            """List all installed Ollama models."""
            result = await self._fetch_models()
            
            if isinstance(result, Success):
                return result.unwrap()
            else:
                error = result.failure()
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE
                    if error.kind == ErrorKind.NETWORK
                    else status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=error.to_dict()
                )
        
        @router.get(
            "/status",
            response_model=OllamaStatus,
            summary="Check Ollama service status",
            description="Check if the Ollama service is running and get basic information"
        )
        async def check_status():
            """Check if Ollama service is running."""
            result = await self._check_status()
            
            if isinstance(result, Success):
                return result.unwrap()
            else:
                error = result.failure()
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=error.to_dict()
                )
        
        @router.post(
            "/select",
            response_model=ModelSelectionResponse,
            summary="Select a model for embeddings",
            description="Select an Ollama model for generating embeddings"
        )
        async def select_model(request: ModelSelectionRequest):
            """Select a model for embeddings."""
            result = await self._select_model(
                model_name=request.model_name,
                pull_if_missing=request.pull_if_missing
            )
            
            if isinstance(result, Success):
                return result.unwrap()
            else:
                error = result.failure()
                status_code = status.HTTP_404_NOT_FOUND \
                    if error.kind == ErrorKind.NOT_FOUND \
                    else status.HTTP_500_INTERNAL_SERVER_ERROR
                raise HTTPException(
                    status_code=status_code,
                    detail=error.to_dict()
                )
        
        @router.get(
            "/current",
            summary="Get current selected model",
            description="Get information about the currently selected model"
        )
        async def get_current_model():
            """Get information about the currently selected model."""
            if self.current_provider:
                return self.current_provider.get_model_info()
            else:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail={
                        "error": "no_model_selected",
                        "message": "No model currently selected",
                        "hint": "Use POST /api/ollama/select to select a model"
                    }
                )
        
        @router.post(
            "/models/pull",
            summary="Pull a new model",
            description="Download and install a new Ollama model"
        )
        async def pull_model(
            model_name: str,
            show_progress: bool = False
        ):
            """Pull a new Ollama model."""
            try:
                provider = OllamaEmbeddingProvider(base_url=self.base_url)
                
                # Run pull in executor to avoid blocking
                success = await asyncio.get_event_loop().run_in_executor(
                    None,
                    provider.pull_model,
                    model_name,
                    show_progress
                )
                
                if success:
                    # Invalidate cache after pulling new model
                    self.model_cache.invalidate()
                    return {
                        "success": True,
                        "message": f"Successfully pulled model {model_name}"
                    }
                else:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail={
                            "error": "pull_failed",
                            "message": f"Failed to pull model {model_name}"
                        }
                    )
                    
            except Exception as e:
                logger.error(f"Error pulling model: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail={
                        "error": "system_error",
                        "message": f"Failed to pull model: {str(e)}"
                    }
                )
        
        @router.delete(
            "/cache",
            summary="Clear model cache",
            description="Clear the cached model list to force a refresh"
        )
        async def clear_cache():
            """Clear the model cache."""
            self.model_cache.invalidate()
            return {"message": "Cache cleared successfully"}
        
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