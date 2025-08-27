"""Anthropic Claude AI provider implementation."""

import json
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional

import httpx
from structlog import get_logger

from .abstract_provider import AbstractProvider
from .models import (
    AIRequest,
    AIResponse,
    CostTier,
    ModelSpec,
    ProviderCapability,
    ProviderType,
    StreamingChunk,
)

logger = get_logger(__name__)


class AnthropicProvider(AbstractProvider):
    """Anthropic Claude AI provider implementation.
    
    Supports Claude models with text generation, tool calling, and vision capabilities.
    Implements streaming responses and comprehensive error handling.
    """
    
    # Anthropic model configurations
    MODELS = {
        "claude-3-5-sonnet-20241022": ModelSpec(
            model_id="claude-3-5-sonnet-20241022",
            provider_type=ProviderType.ANTHROPIC,
            display_name="Claude 3.5 Sonnet",
            capabilities=[
                ProviderCapability.TEXT_GENERATION,
                ProviderCapability.TOOL_CALLING,
                ProviderCapability.VISION,
                ProviderCapability.STREAMING,
            ],
            context_length=200000,
            max_output_tokens=8192,
            cost_per_input_token=3.00,  # Per 1K tokens
            cost_per_output_token=15.00,  # Per 1K tokens  
            cost_tier=CostTier.HIGH,
            supports_streaming=True,
            supports_tools=True,
            supports_vision=True,
        ),
        "claude-3-haiku-20240307": ModelSpec(
            model_id="claude-3-haiku-20240307",
            provider_type=ProviderType.ANTHROPIC,
            display_name="Claude 3 Haiku",
            capabilities=[
                ProviderCapability.TEXT_GENERATION,
                ProviderCapability.TOOL_CALLING,
                ProviderCapability.VISION,
                ProviderCapability.STREAMING,
            ],
            context_length=200000,
            max_output_tokens=4096,
            cost_per_input_token=0.25,  # Per 1K tokens
            cost_per_output_token=1.25,  # Per 1K tokens
            cost_tier=CostTier.LOW,
            supports_streaming=True,
            supports_tools=True,
            supports_vision=True,
        ),
    }
    
    def __init__(self, config):
        """Initialize the Anthropic provider."""
        super().__init__(config)
        self._base_url = config.base_url or "https://api.anthropic.com"
        self._client = None
    
    async def _initialize_client(self) -> None:
        """Initialize the Anthropic HTTP client."""
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=self.config.timeout,
            headers={
                "x-api-key": self.config.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
        )
    
    async def _cleanup_client(self) -> None:
        """Clean up the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
    
    async def _load_models(self) -> None:
        """Load available Anthropic models."""
        self._models = self.MODELS.copy()
        
        # Optionally validate models with API
        try:
            # For Anthropic, we'll use the predefined models
            # In a real implementation, you might want to verify availability
            logger.info(
                "Loaded Anthropic models",
                models=list(self._models.keys()),
            )
        except Exception as e:
            logger.warning(
                "Could not validate Anthropic models",
                error=str(e),
            )
    
    async def _generate_response_impl(self, request: AIRequest) -> AIResponse:
        """Generate response using Anthropic API."""
        start_time = datetime.now()
        
        # Convert request to Anthropic format
        anthropic_request = self._convert_request(request)
        
        try:
            response = await self._client.post(
                "/v1/messages",
                json=anthropic_request,
            )
            response.raise_for_status()
            
            result = response.json()
            
            # Convert response back to standard format
            ai_response = self._convert_response(request, result, start_time)
            
            logger.info(
                "Generated Anthropic response",
                model=request.model,
                tokens=ai_response.usage.get("total_tokens", 0) if ai_response.usage else 0,
                cost=ai_response.cost,
                latency_ms=ai_response.latency_ms,
            )
            
            return ai_response
            
        except httpx.HTTPStatusError as e:
            logger.error(
                "Anthropic API error",
                status_code=e.response.status_code,
                error=e.response.text,
            )
            raise
        except Exception as e:
            logger.error("Anthropic request failed", error=str(e))
            raise
    
    async def _stream_response_impl(
        self, request: AIRequest
    ) -> AsyncGenerator[StreamingChunk, None]:
        """Stream response using Anthropic API."""
        # Convert request to Anthropic format with streaming enabled
        anthropic_request = self._convert_request(request)
        anthropic_request["stream"] = True
        
        try:
            async with self._client.stream(
                "POST",
                "/v1/messages", 
                json=anthropic_request,
            ) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]  # Remove "data: " prefix
                        
                        if data_str == "[DONE]":
                            # Stream completed
                            yield StreamingChunk(
                                request_id=request.request_id,
                                is_complete=True,
                            )
                            break
                        
                        try:
                            data = json.loads(data_str)
                            chunk = self._convert_streaming_chunk(request, data)
                            if chunk:
                                yield chunk
                        except json.JSONDecodeError:
                            # Skip invalid JSON
                            continue
                        
        except httpx.HTTPStatusError as e:
            logger.error(
                "Anthropic streaming error",
                status_code=e.response.status_code,
                error=e.response.text,
            )
            raise
        except Exception as e:
            logger.error("Anthropic streaming failed", error=str(e))
            raise
    
    def _get_supported_capabilities(self) -> List[ProviderCapability]:
        """Get supported capabilities for Anthropic."""
        return [
            ProviderCapability.TEXT_GENERATION,
            ProviderCapability.TOOL_CALLING,
            ProviderCapability.VISION,
            ProviderCapability.STREAMING,
        ]
    
    async def _perform_health_check(self) -> None:
        """Perform health check by calling a simple API endpoint."""
        try:
            # Use a simple completion request for health check
            test_request = {
                "model": "claude-3-haiku-20240307",  # Use cheapest model
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "Hi"}],
            }
            
            response = await self._client.post("/v1/messages", json=test_request)
            response.raise_for_status()
            
            # Update rate limit info if available
            if "x-ratelimit-remaining" in response.headers:
                self._health.rate_limit_remaining = int(
                    response.headers["x-ratelimit-remaining"]
                )
            
            if "x-ratelimit-reset" in response.headers:
                reset_timestamp = int(response.headers["x-ratelimit-reset"])
                self._health.rate_limit_reset = datetime.fromtimestamp(reset_timestamp)
                
        except Exception as e:
            logger.error("Anthropic health check failed", error=str(e))
            raise
    
    def _convert_request(self, request: AIRequest) -> Dict[str, Any]:
        """Convert AIRequest to Anthropic API format."""
        anthropic_request = {
            "model": request.model,
            "messages": self._convert_messages(request.messages),
            "max_tokens": request.max_tokens or 2048,
        }
        
        if request.temperature is not None:
            anthropic_request["temperature"] = request.temperature
        
        if request.tools:
            anthropic_request["tools"] = self._convert_tools(request.tools)
        
        return anthropic_request
    
    def _convert_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert messages to Anthropic format."""
        converted = []
        
        for message in messages:
            role = message.get("role")
            content = message.get("content")
            
            if role == "system":
                # Anthropic handles system messages differently
                # We'll add it as a system parameter or prepend to first user message
                continue
            
            converted_message = {"role": role, "content": content}
            converted.append(converted_message)
        
        return converted
    
    def _convert_tools(self, tools: List[Any]) -> List[Dict[str, Any]]:
        """Convert MCP tools to Anthropic format."""
        converted_tools = []
        
        for tool in tools:
            anthropic_tool = {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema,
            }
            converted_tools.append(anthropic_tool)
        
        return converted_tools
    
    def _convert_response(
        self, request: AIRequest, result: Dict[str, Any], start_time: datetime
    ) -> AIResponse:
        """Convert Anthropic response to AIResponse format."""
        content = ""
        tool_calls = []
        
        # Extract content
        if "content" in result:
            for content_block in result["content"]:
                if content_block.get("type") == "text":
                    content += content_block.get("text", "")
                elif content_block.get("type") == "tool_use":
                    tool_calls.append({
                        "id": content_block.get("id"),
                        "type": "function",
                        "function": {
                            "name": content_block.get("name"),
                            "arguments": json.dumps(content_block.get("input", {})),
                        },
                    })
        
        # Extract usage info
        usage = result.get("usage", {})
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        
        # Calculate cost
        cost = self.get_model_cost(request.model, input_tokens, output_tokens)
        
        # Calculate latency
        latency_ms = (datetime.now() - start_time).total_seconds() * 1000
        
        return AIResponse(
            request_id=request.request_id,
            provider_type=ProviderType.ANTHROPIC,
            model=request.model,
            content=content,
            tool_calls=tool_calls if tool_calls else None,
            finish_reason=result.get("stop_reason"),
            usage={
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
            cost=cost,
            latency_ms=latency_ms,
        )
    
    def _convert_streaming_chunk(
        self, request: AIRequest, data: Dict[str, Any]
    ) -> Optional[StreamingChunk]:
        """Convert Anthropic streaming data to StreamingChunk."""
        event_type = data.get("type")
        
        if event_type == "content_block_delta":
            delta = data.get("delta", {})
            if delta.get("type") == "text_delta":
                return StreamingChunk(
                    request_id=request.request_id,
                    content=delta.get("text", ""),
                )
        elif event_type == "message_delta":
            # Check for stop_reason in message_delta
            delta = data.get("delta", {})
            stop_reason = delta.get("stop_reason")
            if stop_reason:
                return StreamingChunk(
                    request_id=request.request_id,
                    is_complete=True,
                    finish_reason=stop_reason,
                )
        elif event_type == "message_stop":
            # Fallback to message_stop for completion
            return StreamingChunk(
                request_id=request.request_id,
                is_complete=True,
                finish_reason=data.get("stop_reason"),
            )
        
        return None