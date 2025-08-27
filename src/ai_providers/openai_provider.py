"""OpenAI GPT AI provider implementation."""

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


class OpenAIProvider(AbstractProvider):
    """OpenAI GPT AI provider implementation.
    
    Supports GPT models with text generation, tool calling, and vision capabilities.
    Implements streaming responses and comprehensive error handling.
    """
    
    # OpenAI model configurations
    MODELS = {
        "gpt-4o": ModelSpec(
            model_id="gpt-4o",
            provider_type=ProviderType.OPENAI,
            display_name="GPT-4o",
            capabilities=[
                ProviderCapability.TEXT_GENERATION,
                ProviderCapability.TOOL_CALLING,
                ProviderCapability.VISION,
                ProviderCapability.STREAMING,
            ],
            context_length=128000,
            max_output_tokens=4096,
            cost_per_input_token=5.00,  # Per 1K tokens
            cost_per_output_token=15.00,  # Per 1K tokens
            cost_tier=CostTier.HIGH,
            supports_streaming=True,
            supports_tools=True,
            supports_vision=True,
        ),
        "gpt-4o-mini": ModelSpec(
            model_id="gpt-4o-mini",
            provider_type=ProviderType.OPENAI,
            display_name="GPT-4o Mini", 
            capabilities=[
                ProviderCapability.TEXT_GENERATION,
                ProviderCapability.TOOL_CALLING,
                ProviderCapability.VISION,
                ProviderCapability.STREAMING,
            ],
            context_length=128000,
            max_output_tokens=4096,
            cost_per_input_token=0.15,  # Per 1K tokens
            cost_per_output_token=0.60,  # Per 1K tokens
            cost_tier=CostTier.LOW,
            supports_streaming=True,
            supports_tools=True,
            supports_vision=True,
        ),
        "gpt-3.5-turbo": ModelSpec(
            model_id="gpt-3.5-turbo",
            provider_type=ProviderType.OPENAI,
            display_name="GPT-3.5 Turbo",
            capabilities=[
                ProviderCapability.TEXT_GENERATION,
                ProviderCapability.TOOL_CALLING,
                ProviderCapability.STREAMING,
            ],
            context_length=16385,
            max_output_tokens=4096,
            cost_per_input_token=0.50,  # Per 1K tokens
            cost_per_output_token=1.50,  # Per 1K tokens
            cost_tier=CostTier.LOW,
            supports_streaming=True,
            supports_tools=True,
            supports_vision=False,
        ),
    }
    
    def __init__(self, config):
        """Initialize the OpenAI provider."""
        super().__init__(config)
        self._base_url = config.base_url or "https://api.openai.com"
        self._client = None
    
    async def _initialize_client(self) -> None:
        """Initialize the OpenAI HTTP client."""
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=self.config.timeout,
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
        )
    
    async def _cleanup_client(self) -> None:
        """Clean up the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
    
    async def _load_models(self) -> None:
        """Load available OpenAI models."""
        self._models = self.MODELS.copy()
        
        # Optionally load models from API
        try:
            response = await self._client.get("/v1/models")
            if response.status_code == 200:
                models_data = response.json()
                available_models = {m["id"] for m in models_data.get("data", [])}
                
                # Filter out unavailable models
                self._models = {
                    k: v for k, v in self._models.items()
                    if k in available_models
                }
                
                logger.info(
                    "Loaded OpenAI models",
                    available=list(available_models),
                    configured=list(self._models.keys()),
                )
        except Exception as e:
            logger.warning(
                "Could not load OpenAI models from API, using defaults",
                error=str(e),
            )
    
    async def _generate_response_impl(self, request: AIRequest) -> AIResponse:
        """Generate response using OpenAI API."""
        start_time = datetime.now()
        
        # Convert request to OpenAI format
        openai_request = self._convert_request(request)
        
        try:
            response = await self._client.post(
                "/v1/chat/completions",
                json=openai_request,
            )
            response.raise_for_status()
            
            result = response.json()
            
            # Convert response back to standard format
            ai_response = self._convert_response(request, result, start_time)
            
            logger.info(
                "Generated OpenAI response",
                model=request.model,
                tokens=ai_response.usage.get("total_tokens", 0) if ai_response.usage else 0,
                cost=ai_response.cost,
                latency_ms=ai_response.latency_ms,
            )
            
            return ai_response
            
        except httpx.HTTPStatusError as e:
            logger.error(
                "OpenAI API error",
                status_code=e.response.status_code,
                error=e.response.text,
            )
            raise
        except Exception as e:
            logger.error("OpenAI request failed", error=str(e))
            raise
    
    async def _stream_response_impl(
        self, request: AIRequest
    ) -> AsyncGenerator[StreamingChunk, None]:
        """Stream response using OpenAI API."""
        # Convert request to OpenAI format with streaming enabled
        openai_request = self._convert_request(request)
        openai_request["stream"] = True
        
        try:
            async with self._client.stream(
                "POST",
                "/v1/chat/completions",
                json=openai_request,
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
                "OpenAI streaming error",
                status_code=e.response.status_code,
                error=e.response.text,
            )
            raise
        except Exception as e:
            logger.error("OpenAI streaming failed", error=str(e))
            raise
    
    def _get_supported_capabilities(self) -> List[ProviderCapability]:
        """Get supported capabilities for OpenAI."""
        return [
            ProviderCapability.TEXT_GENERATION,
            ProviderCapability.TOOL_CALLING,
            ProviderCapability.VISION,
            ProviderCapability.STREAMING,
            ProviderCapability.BATCH_PROCESSING,
            ProviderCapability.FINE_TUNING,
        ]
    
    async def _perform_health_check(self) -> None:
        """Perform health check by calling the models endpoint."""
        try:
            response = await self._client.get("/v1/models")
            response.raise_for_status()
            
            # Update rate limit info if available
            if "x-ratelimit-remaining-requests" in response.headers:
                self._health.rate_limit_remaining = int(
                    response.headers["x-ratelimit-remaining-requests"]
                )
            
            if "x-ratelimit-reset-requests" in response.headers:
                reset_timestamp = int(response.headers["x-ratelimit-reset-requests"])
                self._health.rate_limit_reset = datetime.fromtimestamp(reset_timestamp)
                
        except Exception as e:
            logger.error("OpenAI health check failed", error=str(e))
            raise
    
    def _convert_request(self, request: AIRequest) -> Dict[str, Any]:
        """Convert AIRequest to OpenAI API format."""
        openai_request = {
            "model": request.model,
            "messages": request.messages,
        }
        
        if request.max_tokens:
            openai_request["max_tokens"] = request.max_tokens
            
        if request.temperature is not None:
            openai_request["temperature"] = request.temperature
        
        if request.tools:
            openai_request["tools"] = self._convert_tools(request.tools)
            openai_request["tool_choice"] = "auto"
        
        return openai_request
    
    def _convert_tools(self, tools: List[Any]) -> List[Dict[str, Any]]:
        """Convert MCP tools to OpenAI format."""
        converted_tools = []
        
        for tool in tools:
            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema,
                },
            }
            converted_tools.append(openai_tool)
        
        return converted_tools
    
    def _convert_response(
        self, request: AIRequest, result: Dict[str, Any], start_time: datetime
    ) -> AIResponse:
        """Convert OpenAI response to AIResponse format."""
        choice = result.get("choices", [{}])[0]
        message = choice.get("message", {})
        
        content = message.get("content", "")
        tool_calls = message.get("tool_calls")
        
        # Extract usage info
        usage = result.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        
        # Calculate cost
        cost = self.get_model_cost(request.model, input_tokens, output_tokens)
        
        # Calculate latency
        latency_ms = (datetime.now() - start_time).total_seconds() * 1000
        
        return AIResponse(
            request_id=request.request_id,
            provider_type=ProviderType.OPENAI,
            model=request.model,
            content=content,
            tool_calls=tool_calls,
            finish_reason=choice.get("finish_reason"),
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
        """Convert OpenAI streaming data to StreamingChunk."""
        choices = data.get("choices", [])
        if not choices:
            return None
        
        choice = choices[0]
        delta = choice.get("delta", {})
        
        content = delta.get("content")
        tool_calls = delta.get("tool_calls")
        finish_reason = choice.get("finish_reason")
        
        if content or tool_calls or finish_reason:
            return StreamingChunk(
                request_id=request.request_id,
                content=content,
                tool_calls=tool_calls,
                finish_reason=finish_reason,
                is_complete=finish_reason is not None,
            )
        
        return None