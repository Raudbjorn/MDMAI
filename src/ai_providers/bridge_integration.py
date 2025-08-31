"""Integration layer between AI providers and MCP Bridge infrastructure."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import HTTPException, status
from pydantic import BaseModel
from structlog import get_logger

from ..bridge.models import MCPRequest, MCPResponse
from .models import AIRequest, ProviderSelection, ProviderType, MCPTool
from .provider_manager import AIProviderManager

logger = get_logger(__name__)


class AIProviderRequest(BaseModel):
    """Request to AI provider through MCP Bridge."""
    
    session_id: Optional[str] = None
    provider_type: Optional[ProviderType] = None
    model: str
    messages: List[Dict[str, Any]]
    tools: Optional[List[MCPTool]] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    stream: bool = False
    selection_criteria: Optional[ProviderSelection] = None
    strategy: str = "cost"
    budget_limit: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class AIProviderResponse(BaseModel):
    """Response from AI provider."""
    
    request_id: str
    provider_type: str
    model: str
    content: str
    tool_calls: Optional[List[Dict[str, Any]]] = None
    usage: Optional[Dict[str, Any]] = None
    cost: Optional[float] = None
    latency_ms: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class AIProviderBridgeIntegration:
    """Integration layer for AI providers with MCP Bridge."""
    
    def __init__(self, provider_manager: AIProviderManager):
        self.provider_manager = provider_manager
        self._request_mapping: Dict[str, str] = {}  # MCP request ID -> AI request ID
    
    async def handle_ai_request(
        self,
        mcp_request: MCPRequest,
        ai_request_data: AIProviderRequest,
    ) -> MCPResponse:
        """Handle AI provider request from MCP Bridge.
        
        Args:
            mcp_request: Original MCP request
            ai_request_data: AI provider request data
            
        Returns:
            MCPResponse with AI provider result
        """
        try:
            # Convert to internal AI request format
            ai_request = self._convert_to_ai_request(ai_request_data)
            
            # Track request mapping
            self._request_mapping[mcp_request.id] = ai_request.request_id
            
            logger.info(
                "Processing AI provider request",
                mcp_request_id=mcp_request.id,
                ai_request_id=ai_request.request_id,
                model=ai_request.model,
                strategy=ai_request_data.strategy,
            )
            
            # Process through provider manager
            response = await self.provider_manager.process_request(
                ai_request,
                tools=ai_request_data.tools,
                selection=ai_request_data.selection_criteria,
                strategy=ai_request_data.strategy,
            )
            
            # Convert response
            ai_response = AIProviderResponse(
                request_id=response.request_id,
                provider_type=response.provider_type.value,
                model=response.model,
                content=response.content,
                tool_calls=response.tool_calls,
                usage=response.usage,
                cost=response.cost,
                latency_ms=response.latency_ms,
                metadata=response.metadata,
            )
            
            return MCPResponse(
                id=mcp_request.id,
                result={
                    "ai_response": ai_response.dict(),
                    "success": True,
                    "timestamp": datetime.now().isoformat(),
                },
            )
            
        except Exception as e:
            logger.error(
                "AI provider request failed",
                mcp_request_id=mcp_request.id,
                error=str(e),
            )
            
            return MCPResponse(
                id=mcp_request.id,
                error={
                    "code": -32603,  # Internal error
                    "message": f"AI provider request failed: {str(e)}",
                    "data": {
                        "error_type": type(e).__name__,
                        "timestamp": datetime.now().isoformat(),
                    },
                },
            )
    
    async def handle_streaming_request(
        self,
        mcp_request: MCPRequest,
        ai_request_data: AIProviderRequest,
    ):
        """Handle streaming AI provider request.
        
        Args:
            mcp_request: Original MCP request
            ai_request_data: AI provider request data
            
        Yields:
            Streaming response chunks
        """
        try:
            # Convert to internal AI request format
            ai_request = self._convert_to_ai_request(ai_request_data)
            ai_request.stream = True
            
            # Track request mapping
            self._request_mapping[mcp_request.id] = ai_request.request_id
            
            logger.info(
                "Starting AI provider streaming request",
                mcp_request_id=mcp_request.id,
                ai_request_id=ai_request.request_id,
                model=ai_request.model,
            )
            
            # Stream through provider manager
            async for chunk in self.provider_manager.stream_request(
                ai_request,
                tools=ai_request_data.tools,
                selection=ai_request_data.selection_criteria,
                strategy=ai_request_data.strategy,
            ):
                yield {
                    "mcp_request_id": mcp_request.id,
                    "ai_request_id": ai_request.request_id,
                    "chunk": {
                        "chunk_id": chunk.chunk_id,
                        "content": chunk.content,
                        "tool_calls": chunk.tool_calls,
                        "finish_reason": chunk.finish_reason,
                        "is_complete": chunk.is_complete,
                        "metadata": chunk.metadata,
                    },
                    "timestamp": datetime.now().isoformat(),
                }
            
        except Exception as e:
            logger.error(
                "AI provider streaming request failed",
                mcp_request_id=mcp_request.id,
                error=str(e),
            )
            
            yield {
                "mcp_request_id": mcp_request.id,
                "error": {
                    "code": -32603,
                    "message": f"Streaming failed: {str(e)}",
                    "data": {"error_type": type(e).__name__},
                },
                "timestamp": datetime.now().isoformat(),
            }
    
    async def get_provider_status(self) -> Dict[str, Any]:
        """Get status of all AI providers.
        
        Returns:
            Dictionary with provider status information
        """
        stats = self.provider_manager.get_manager_stats()
        
        return {
            "initialized": stats["initialized"],
            "providers": stats["registry_stats"]["provider_stats"],
            "total_requests": stats["usage_stats"]["total_requests"],
            "total_cost": stats["usage_stats"]["total_cost"],
            "active_streams": stats["streaming_stats"]["active_streams"],
            "budget_alerts": stats["budget_alerts"],
            "timestamp": datetime.now().isoformat(),
        }
    
    async def get_cost_recommendations(
        self, model: str, messages: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Get cost recommendations for a request.
        
        Args:
            model: Preferred model (if available)
            messages: Request messages
            
        Returns:
            Cost recommendations across providers
        """
        ai_request = AIRequest(
            model=model,
            messages=messages,
        )
        
        recommendations = await self.provider_manager.get_cost_recommendations(ai_request)
        
        return {
            "recommendations": {
                provider.value: rec for provider, rec in recommendations.items()
            },
            "timestamp": datetime.now().isoformat(),
        }
    
    async def configure_provider_settings(
        self, settings: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Configure AI provider settings.
        
        Args:
            settings: Settings to configure
            
        Returns:
            Configuration result
        """
        try:
            # Handle provider priority changes
            if "provider_priorities" in settings:
                for provider_str, priority in settings["provider_priorities"].items():
                    try:
                        provider_type = ProviderType(provider_str)
                        self.provider_manager.configure_provider_priority(provider_type, priority)
                    except ValueError:
                        logger.warning("Invalid provider type", provider=provider_str)
            
            # Handle budget configuration
            if "budgets" in settings:
                for budget_data in settings["budgets"]:
                    # This would need to be implemented based on budget structure
                    pass
            
            # Handle retry configuration
            if "retry_config" in settings:
                retry_config = settings["retry_config"]
                self.provider_manager.error_handler.configure_retries(
                    max_retries=retry_config.get("max_retries"),
                    base_delay=retry_config.get("base_delay"),
                    max_delay=retry_config.get("max_delay"),
                )
            
            return {
                "success": True,
                "message": "Settings configured successfully",
                "timestamp": datetime.now().isoformat(),
            }
            
        except Exception as e:
            logger.error("Failed to configure provider settings", error=str(e))
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }
    
    def _convert_to_ai_request(self, ai_request_data: AIProviderRequest) -> AIRequest:
        """Convert bridge request to internal AI request format.
        
        Args:
            ai_request_data: Bridge request data
            
        Returns:
            Internal AI request
        """
        return AIRequest(
            session_id=ai_request_data.session_id,
            model=ai_request_data.model,
            messages=ai_request_data.messages,
            tools=ai_request_data.tools,
            max_tokens=ai_request_data.max_tokens,
            temperature=ai_request_data.temperature,
            stream=ai_request_data.stream,
            budget_limit=ai_request_data.budget_limit,
            metadata=ai_request_data.metadata,
        )


# Integration functions for extending the existing bridge server

def extend_bridge_routes(app, ai_provider_manager: AIProviderManager):
    """Extend the MCP Bridge server with AI provider routes.
    
    Args:
        app: FastAPI application
        ai_provider_manager: AI provider manager instance
    """
    bridge_integration = AIProviderBridgeIntegration(ai_provider_manager)
    
    @app.post("/ai/generate")
    async def ai_generate(request_data: AIProviderRequest):
        """Generate AI response through provider integration."""
        try:
            # Create mock MCP request for compatibility
            mcp_request = MCPRequest(
                method="ai/generate",
                params=request_data.dict(),
            )
            
            response = await bridge_integration.handle_ai_request(mcp_request, request_data)
            
            if response.error:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=response.error,
                )
            
            return response.result
            
        except Exception as e:
            logger.error("AI generation request failed", error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e),
            )
    
    @app.get("/ai/status")
    async def ai_status():
        """Get AI provider status."""
        return await bridge_integration.get_provider_status()
    
    @app.post("/ai/recommendations")
    async def ai_cost_recommendations(
        model: str,
        messages: List[Dict[str, Any]],
    ):
        """Get cost recommendations for AI request."""
        return await bridge_integration.get_cost_recommendations(model, messages)
    
    @app.post("/ai/configure")
    async def ai_configure(settings: Dict[str, Any]):
        """Configure AI provider settings."""
        return await bridge_integration.configure_provider_settings(settings)
    
    # Streaming endpoints would be added to the WebSocket and SSE handlers
    # in the main bridge server
    
    logger.info("Extended MCP Bridge with AI provider routes")