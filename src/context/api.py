"""Context management API endpoints."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, HTTPException, Path, Query, status
from pydantic import BaseModel, Field
from structlog import get_logger

from ..utils.serialization import serialize_to_json, deserialize_from_json
from .context_manager import ContextManager
from .models import Context, ContextMetadata, ContextState

logger = get_logger(__name__)


class ContextCreateRequest(BaseModel):
    """Request model for creating a new context."""
    
    name: str = Field(..., description="Context name")
    description: Optional[str] = Field(None, description="Context description")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Context metadata")
    initial_data: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Initial context data")


class ContextUpdateRequest(BaseModel):
    """Request model for updating context data."""
    
    data: Dict[str, Any] = Field(..., description="Context data to update")
    merge: bool = Field(True, description="Whether to merge with existing data or replace")
    version_comment: Optional[str] = Field(None, description="Version comment for this update")


class ContextResponse(BaseModel):
    """Response model for context operations."""
    
    context_id: str
    name: str
    description: Optional[str]
    state: ContextState
    version: int
    data: Dict[str, Any]
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime


class ContextListResponse(BaseModel):
    """Response model for listing contexts."""
    
    contexts: List[ContextResponse]
    total_count: int
    page: int
    page_size: int


def create_context_router(context_manager: ContextManager) -> APIRouter:
    """Create context management API router.
    
    Args:
        context_manager: Context manager instance
        
    Returns:
        FastAPI router with context endpoints
    """
    router = APIRouter(prefix="/api/context", tags=["context"])
    
    @router.get("/", response_model=ContextListResponse)
    async def list_contexts(
        page: int = Query(1, ge=1, description="Page number"),
        page_size: int = Query(20, ge=1, le=100, description="Page size"),
        state: Optional[ContextState] = Query(None, description="Filter by state"),
    ):
        """List all contexts with pagination."""
        try:
            all_contexts = await context_manager.list_contexts(state=state)
            
            # Apply pagination
            start_idx = (page - 1) * page_size
            end_idx = start_idx + page_size
            paginated_contexts = all_contexts[start_idx:end_idx]
            
            # Convert to response format
            response_contexts = []
            for context in paginated_contexts:
                response_contexts.append(ContextResponse(
                    context_id=context.context_id,
                    name=context.name,
                    description=context.description,
                    state=context.state,
                    version=context.version,
                    data=context.data,
                    metadata=context.metadata.dict() if hasattr(context.metadata, 'dict') else context.metadata,
                    created_at=context.created_at,
                    updated_at=context.updated_at,
                ))
            
            return ContextListResponse(
                contexts=response_contexts,
                total_count=len(all_contexts),
                page=page,
                page_size=page_size,
            )
            
        except Exception as e:
            logger.error("Failed to list contexts", error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e),
            )
    
    @router.post("/", response_model=ContextResponse, status_code=status.HTTP_201_CREATED)
    async def create_context(
        request: ContextCreateRequest = Body(..., description="Context creation request")
    ):
        """Create a new context."""
        try:
            # Create context metadata
            metadata = ContextMetadata(
                created_by="api",
                tags=[],
                custom_fields=request.metadata or {},
            )
            
            # Create context
            context = await context_manager.create_context(
                name=request.name,
                description=request.description,
                metadata=metadata,
                initial_data=request.initial_data,
            )
            
            return ContextResponse(
                context_id=context.context_id,
                name=context.name,
                description=context.description,
                state=context.state,
                version=context.version,
                data=context.data,
                metadata=context.metadata.dict() if hasattr(context.metadata, 'dict') else context.metadata,
                created_at=context.created_at,
                updated_at=context.updated_at,
            )
            
        except Exception as e:
            logger.error("Failed to create context", error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e),
            )
    
    @router.get("/{context_id}", response_model=ContextResponse)
    async def get_context(
        context_id: str = Path(..., description="Context ID"),
        version: Optional[int] = Query(None, ge=1, description="Specific version to retrieve"),
    ):
        """Get a specific context by ID."""
        try:
            if version:
                context = await context_manager.get_context_version(context_id, version)
            else:
                context = await context_manager.get_context(context_id)
            
            if not context:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Context {context_id} not found",
                )
            
            return ContextResponse(
                context_id=context.context_id,
                name=context.name,
                description=context.description,
                state=context.state,
                version=context.version,
                data=context.data,
                metadata=context.metadata.dict() if hasattr(context.metadata, 'dict') else context.metadata,
                created_at=context.created_at,
                updated_at=context.updated_at,
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Failed to get context", context_id=context_id, error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e),
            )
    
    @router.put("/{context_id}", response_model=ContextResponse)
    async def update_context(
        context_id: str = Path(..., description="Context ID"),
        request: ContextUpdateRequest = Body(..., description="Context update request"),
    ):
        """Update context data using request body.
        
        This endpoint properly uses the request body for context data,
        not the path parameter.
        """
        try:
            context = await context_manager.get_context(context_id)
            
            if not context:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Context {context_id} not found",
                )
            
            # Update context data
            if request.merge:
                # Merge with existing data
                updated_data = {**context.data, **request.data}
            else:
                # Replace data
                updated_data = request.data
            
            # Update context
            updated_context = await context_manager.update_context(
                context_id=context_id,
                data=updated_data,
                version_comment=request.version_comment,
            )
            
            return ContextResponse(
                context_id=updated_context.context_id,
                name=updated_context.name,
                description=updated_context.description,
                state=updated_context.state,
                version=updated_context.version,
                data=updated_context.data,
                metadata=updated_context.metadata.dict() if hasattr(updated_context.metadata, 'dict') else updated_context.metadata,
                created_at=updated_context.created_at,
                updated_at=updated_context.updated_at,
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Failed to update context", context_id=context_id, error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e),
            )
    
    @router.post("/{context_id}/data", response_model=ContextResponse)
    async def update_context_data(
        context_id: str = Path(..., description="Context ID"),
        data: Dict[str, Any] = Body(..., description="Data to update"),
        merge: bool = Query(True, description="Whether to merge or replace"),
    ):
        """Update context data with POST method.
        
        This is the proper implementation that uses the request body
        for context data, not path parameters.
        """
        try:
            context = await context_manager.get_context(context_id)
            
            if not context:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Context {context_id} not found",
                )
            
            # Update context data
            if merge:
                updated_data = {**context.data, **data}
            else:
                updated_data = data
            
            # Update context
            updated_context = await context_manager.update_context(
                context_id=context_id,
                data=updated_data,
            )
            
            return ContextResponse(
                context_id=updated_context.context_id,
                name=updated_context.name,
                description=updated_context.description,
                state=updated_context.state,
                version=updated_context.version,
                data=updated_context.data,
                metadata=updated_context.metadata.dict() if hasattr(updated_context.metadata, 'dict') else updated_context.metadata,
                created_at=updated_context.created_at,
                updated_at=updated_context.updated_at,
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Failed to update context data", context_id=context_id, error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e),
            )
    
    @router.delete("/{context_id}", status_code=status.HTTP_204_NO_CONTENT)
    async def delete_context(
        context_id: str = Path(..., description="Context ID"),
    ):
        """Delete a context."""
        try:
            success = await context_manager.delete_context(context_id)
            
            if not success:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Context {context_id} not found",
                )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Failed to delete context", context_id=context_id, error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e),
            )
    
    @router.post("/{context_id}/archive", response_model=ContextResponse)
    async def archive_context(
        context_id: str = Path(..., description="Context ID"),
    ):
        """Archive a context."""
        try:
            context = await context_manager.archive_context(context_id)
            
            if not context:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Context {context_id} not found",
                )
            
            return ContextResponse(
                context_id=context.context_id,
                name=context.name,
                description=context.description,
                state=context.state,
                version=context.version,
                data=context.data,
                metadata=context.metadata.dict() if hasattr(context.metadata, 'dict') else context.metadata,
                created_at=context.created_at,
                updated_at=context.updated_at,
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Failed to archive context", context_id=context_id, error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e),
            )
    
    @router.post("/{context_id}/restore", response_model=ContextResponse)
    async def restore_context(
        context_id: str = Path(..., description="Context ID"),
        version: Optional[int] = Query(None, ge=1, description="Version to restore to"),
    ):
        """Restore an archived context or restore to a specific version."""
        try:
            if version:
                context = await context_manager.restore_version(context_id, version)
            else:
                context = await context_manager.restore_context(context_id)
            
            if not context:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Context {context_id} not found",
                )
            
            return ContextResponse(
                context_id=context.context_id,
                name=context.name,
                description=context.description,
                state=context.state,
                version=context.version,
                data=context.data,
                metadata=context.metadata.dict() if hasattr(context.metadata, 'dict') else context.metadata,
                created_at=context.created_at,
                updated_at=context.updated_at,
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Failed to restore context", context_id=context_id, error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e),
            )
    
    @router.get("/{context_id}/versions")
    async def list_context_versions(
        context_id: str = Path(..., description="Context ID"),
    ):
        """List all versions of a context."""
        try:
            versions = await context_manager.list_versions(context_id)
            
            if not versions:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Context {context_id} not found",
                )
            
            return {
                "context_id": context_id,
                "versions": versions,
                "total_versions": len(versions),
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Failed to list context versions", context_id=context_id, error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e),
            )
    
    @router.post("/{context_id}/export")
    async def export_context(
        context_id: str = Path(..., description="Context ID"),
        format: str = Query("json", description="Export format (json, yaml)"),
    ):
        """Export context data."""
        try:
            context = await context_manager.get_context(context_id)
            
            if not context:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Context {context_id} not found",
                )
            
            export_data = {
                "context_id": context.context_id,
                "name": context.name,
                "description": context.description,
                "version": context.version,
                "data": context.data,
                "metadata": context.metadata.dict() if hasattr(context.metadata, 'dict') else context.metadata,
                "exported_at": datetime.now().isoformat(),
            }
            
            if format == "yaml":
                try:
                    import yaml
                    return yaml.dump(export_data, default_flow_style=False)
                except ImportError:
                    # Fallback to JSON if yaml not available
                    format = "json"
            
            # Default to JSON
            return serialize_to_json(export_data, pretty=True)
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Failed to export context", context_id=context_id, error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e),
            )
    
    return router