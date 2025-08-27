"""Google Gemini AI provider implementation."""

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


class GoogleProvider(AbstractProvider):
    """Google Gemini AI provider implementation.
    
    Supports Gemini models with text generation, tool calling, and vision capabilities.
    Implements streaming responses and comprehensive error handling.
    """
    
    # Google Gemini model configurations
    MODELS = {
        "gemini-1.5-pro": ModelSpec(
            model_id="gemini-1.5-pro",
            provider_type=ProviderType.GOOGLE,
            display_name="Gemini 1.5 Pro",
            capabilities=[
                ProviderCapability.TEXT_GENERATION,
                ProviderCapability.TOOL_CALLING,
                ProviderCapability.VISION,
                ProviderCapability.STREAMING,
            ],
            context_length=2048000,  # 2M tokens
            max_output_tokens=8192,
            cost_per_input_token=1.25,  # Per 1K tokens
            cost_per_output_token=5.00,  # Per 1K tokens
            cost_tier=CostTier.MEDIUM,
            supports_streaming=True,
            supports_tools=True,
            supports_vision=True,
        ),
        "gemini-1.5-flash": ModelSpec(
            model_id="gemini-1.5-flash",
            provider_type=ProviderType.GOOGLE,
            display_name="Gemini 1.5 Flash",
            capabilities=[
                ProviderCapability.TEXT_GENERATION,
                ProviderCapability.TOOL_CALLING,
                ProviderCapability.VISION,
                ProviderCapability.STREAMING,
            ],
            context_length=1048576,  # 1M tokens
            max_output_tokens=8192,
            cost_per_input_token=0.075,  # Per 1K tokens
            cost_per_output_token=0.30,  # Per 1K tokens
            cost_tier=CostTier.LOW,
            supports_streaming=True,
            supports_tools=True,
            supports_vision=True,
        ),
        "gemini-pro": ModelSpec(
            model_id="gemini-pro",
            provider_type=ProviderType.GOOGLE,
            display_name="Gemini Pro",
            capabilities=[
                ProviderCapability.TEXT_GENERATION,
                ProviderCapability.TOOL_CALLING,
                ProviderCapability.STREAMING,
            ],
            context_length=32768,
            max_output_tokens=2048,
            cost_per_input_token=0.50,  # Per 1K tokens
            cost_per_output_token=1.50,  # Per 1K tokens
            cost_tier=CostTier.LOW,
            supports_streaming=True,
            supports_tools=True,
            supports_vision=False,
        ),
    }
    
    def __init__(self, config):
        """Initialize the Google provider."""
        super().__init__(config)
        self._base_url = config.base_url or "https://generativelanguage.googleapis.com"
        self._client = None
    
    async def _initialize_client(self) -> None:
        """Initialize the Google HTTP client."""
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=self.config.timeout,
            headers={
                "Content-Type": "application/json",
            },
        )
    
    async def _cleanup_client(self) -> None:
        """Clean up the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
    
    async def _load_models(self) -> None:
        """Load available Google models."""
        self._models = self.MODELS.copy()
        
        # Optionally validate models with API
        try:
            response = await self._client.get(
                f"/v1beta/models?key={self.config.api_key}"
            )
            
            if response.status_code == 200:
                models_data = response.json()
                available_models = set()
                
                for model in models_data.get("models", []):
                    model_name = model.get("name", "").replace("models/", "")
                    if model_name:
                        available_models.add(model_name)
                
                # Filter out unavailable models
                self._models = {
                    k: v for k, v in self._models.items()
                    if k in available_models
                }
                
                logger.info(
                    "Loaded Google models",
                    available=list(available_models),
                    configured=list(self._models.keys()),
                )
        except Exception as e:
            logger.warning(
                "Could not load Google models from API, using defaults",
                error=str(e),
            )
    
    async def _generate_response_impl(self, request: AIRequest) -> AIResponse:
        """Generate response using Google Gemini API."""
        start_time = datetime.now()
        
        # Convert request to Google format
        google_request = self._convert_request(request)
        
        try:
            model_name = f"models/{request.model}"
            endpoint = f"/v1beta/{model_name}:generateContent"
            
            response = await self._client.post(
                endpoint,
                json=google_request,
                params={"key": self.config.api_key},
            )
            response.raise_for_status()
            
            result = response.json()
            
            # Convert response back to standard format
            ai_response = self._convert_response(request, result, start_time)
            
            logger.info(
                "Generated Google response",
                model=request.model,
                tokens=ai_response.usage.get("total_tokens", 0) if ai_response.usage else 0,
                cost=ai_response.cost,
                latency_ms=ai_response.latency_ms,
            )
            
            return ai_response
            
        except httpx.HTTPStatusError as e:
            logger.error(
                "Google API error",
                status_code=e.response.status_code,
                error=e.response.text,
            )
            raise
        except Exception as e:
            logger.error("Google request failed", error=str(e))
            raise
    
    async def _stream_response_impl(
        self, request: AIRequest
    ) -> AsyncGenerator[StreamingChunk, None]:
        """Stream response using Google Gemini API."""
        # Convert request to Google format
        google_request = self._convert_request(request)
        
        try:
            model_name = f"models/{request.model}"
            endpoint = f"/v1beta/{model_name}:streamGenerateContent"
            
            async with self._client.stream(
                "POST",
                endpoint,
                json=google_request,
                params={"key": self.config.api_key},
            ) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if line.strip():
                        try:
                            # Google uses JSON lines format for streaming
                            data = json.loads(line)
                            chunk = self._convert_streaming_chunk(request, data)
                            if chunk:
                                yield chunk
                        except json.JSONDecodeError:
                            # Skip invalid JSON
                            continue
                
                # Send completion chunk
                yield StreamingChunk(
                    request_id=request.request_id,
                    is_complete=True,
                )
                        
        except httpx.HTTPStatusError as e:
            logger.error(
                "Google streaming error",
                status_code=e.response.status_code,
                error=e.response.text,
            )
            raise
        except Exception as e:
            logger.error("Google streaming failed", error=str(e))
            raise
    
    def _get_supported_capabilities(self) -> List[ProviderCapability]:
        """Get supported capabilities for Google."""
        return [
            ProviderCapability.TEXT_GENERATION,
            ProviderCapability.TOOL_CALLING,
            ProviderCapability.VISION,
            ProviderCapability.STREAMING,
        ]
    
    async def _perform_health_check(self) -> None:
        """Perform health check by listing models."""
        try:
            response = await self._client.get(
                f"/v1beta/models?key={self.config.api_key}"
            )
            response.raise_for_status()
            
        except Exception as e:
            logger.error("Google health check failed", error=str(e))
            raise
    
    def _convert_request(self, request: AIRequest) -> Dict[str, Any]:
        """Convert AIRequest to Google Gemini API format."""
        # Convert messages to Google format
        contents = self._convert_messages(request.messages)
        
        google_request = {
            "contents": contents,
        }
        
        # Add generation config
        generation_config = {}
        if request.max_tokens:
            generation_config["maxOutputTokens"] = request.max_tokens
        if request.temperature is not None:
            generation_config["temperature"] = request.temperature
        
        if generation_config:
            google_request["generationConfig"] = generation_config
        
        # Add tools if present
        if request.tools:
            google_request["tools"] = self._convert_tools(request.tools)
        
        return google_request
    
    def _convert_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert messages to Google format."""
        contents = []
        system_content = ""
        
        # First pass: collect system messages
        for message in messages:
            if message.get("role") == "system":
                if system_content:
                    system_content += "\n\n"
                system_content += message.get("content", "")
        
        # Second pass: convert messages
        first_user_message = True
        for message in messages:
            role = message.get("role")
            content = message.get("content", "")
            
            if role == "system":
                # Already collected
                continue
            elif role == "user":
                google_role = "user"
                # Prepend system content to first user message
                if first_user_message and system_content:
                    content = f"{system_content}\n\n{content}"
                    first_user_message = False
            elif role == "assistant":
                google_role = "model"
            else:
                google_role = role
            
            contents.append({
                "role": google_role,
                "parts": [{"text": content}],
            })
        
        # If no user messages but we have system content, create a user message
        if system_content and not contents:
            contents.append({
                "role": "user",
                "parts": [{"text": system_content}],
            })
        
        return contents
    
    def _convert_tools(self, tools: List[Any]) -> List[Dict[str, Any]]:
        """Convert MCP tools to Google format."""
        function_declarations = []
        
        for tool in tools:
            function_declaration = {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema,
            }
            function_declarations.append(function_declaration)
        
        return [{
            "functionDeclarations": function_declarations
        }]
    
    def _convert_response(
        self, request: AIRequest, result: Dict[str, Any], start_time: datetime
    ) -> AIResponse:
        """Convert Google response to AIResponse format."""
        content = ""
        tool_calls = []
        finish_reason = None
        
        # Extract content from candidates
        candidates = result.get("candidates", [])
        if candidates:
            candidate = candidates[0]
            parts = candidate.get("content", {}).get("parts", [])
            
            for part in parts:
                if "text" in part:
                    content += part["text"]
                elif "functionCall" in part:
                    func_call = part["functionCall"]
                    tool_calls.append({
                        "id": f"call_{len(tool_calls)}",
                        "type": "function",
                        "function": {
                            "name": func_call.get("name"),
                            "arguments": json.dumps(func_call.get("args", {})),
                        },
                    })
            
            finish_reason = candidate.get("finishReason")
        
        # Extract usage info (Google doesn't provide detailed token counts in all cases)
        usage_metadata = result.get("usageMetadata", {})
        input_tokens = usage_metadata.get("promptTokenCount", 0)
        output_tokens = usage_metadata.get("candidatesTokenCount", 0)
        
        # Calculate cost
        cost = self.get_model_cost(request.model, input_tokens, output_tokens)
        
        # Calculate latency
        latency_ms = (datetime.now() - start_time).total_seconds() * 1000
        
        return AIResponse(
            request_id=request.request_id,
            provider_type=ProviderType.GOOGLE,
            model=request.model,
            content=content,
            tool_calls=tool_calls if tool_calls else None,
            finish_reason=finish_reason,
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
        """Convert Google streaming data to StreamingChunk."""
        candidates = data.get("candidates", [])
        if not candidates:
            return None
        
        candidate = candidates[0]
        parts = candidate.get("content", {}).get("parts", [])
        
        content = ""
        tool_calls = []
        
        for part in parts:
            if "text" in part:
                content += part["text"]
            elif "functionCall" in part:
                func_call = part["functionCall"]
                tool_calls.append({
                    "id": f"call_{len(tool_calls)}",
                    "type": "function", 
                    "function": {
                        "name": func_call.get("name"),
                        "arguments": json.dumps(func_call.get("args", {})),
                    },
                })
        
        finish_reason = candidate.get("finishReason")
        
        if content or tool_calls or finish_reason:
            return StreamingChunk(
                request_id=request.request_id,
                content=content if content else None,
                tool_calls=tool_calls if tool_calls else None,
                finish_reason=finish_reason,
                is_complete=finish_reason is not None,
            )
        
        return None