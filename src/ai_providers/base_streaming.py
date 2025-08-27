"""Base streaming handler to reduce duplication across providers."""

import json
from abc import abstractmethod
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, Optional

import httpx
from structlog import get_logger

from .models import AIRequest, StreamingChunk

logger = get_logger(__name__)


class BaseStreamingHandler:
    """Base class for handling streaming responses across different providers."""
    
    def __init__(self, provider_name: str):
        """
        Initialize base streaming handler.
        
        Args:
            provider_name: Name of the provider for logging
        """
        self.provider_name = provider_name
        self._total_tokens = 0
        self._start_time: Optional[datetime] = None
    
    async def handle_streaming_response(
        self,
        client: httpx.AsyncClient,
        request: AIRequest,
        endpoint: str,
        request_data: Dict[str, Any],
        headers: Optional[Dict[str, str]] = None,
    ) -> AsyncGenerator[StreamingChunk, None]:
        """
        Handle streaming response from provider.
        
        Args:
            client: HTTP client
            request: Original AI request
            endpoint: API endpoint
            request_data: Request data to send
            headers: Optional additional headers
            
        Yields:
            StreamingChunk: Parsed streaming chunks
        """
        self._start_time = datetime.now()
        self._total_tokens = 0
        
        try:
            async with client.stream(
                "POST",
                endpoint,
                json=request_data,
                headers=headers,
            ) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    chunk = await self._process_stream_line(line, request)
                    if chunk:
                        yield chunk
                        
        except httpx.HTTPStatusError as e:
            logger.error(
                f"{self.provider_name} streaming error",
                status_code=e.response.status_code,
                error=e.response.text,
            )
            raise
        except Exception as e:
            logger.error(f"{self.provider_name} streaming failed", error=str(e))
            raise
    
    async def _process_stream_line(
        self, 
        line: str, 
        request: AIRequest
    ) -> Optional[StreamingChunk]:
        """
        Process a single line from the stream.
        
        Args:
            line: Raw line from stream
            request: Original request
            
        Returns:
            Parsed chunk or None
        """
        # Handle SSE format (data: prefix)
        if line.startswith("data: "):
            data_str = line[6:]  # Remove "data: " prefix
            
            # Check for stream completion
            if data_str.strip() in ["[DONE]", "done", ""]:
                return StreamingChunk(
                    request_id=request.request_id,
                    is_complete=True,
                )
            
            try:
                data = json.loads(data_str)
                return await self._parse_stream_data(data, request)
            except json.JSONDecodeError:
                logger.debug(f"Skipping invalid JSON in {self.provider_name} stream: {data_str[:100]}")
                return None
        
        # Handle raw JSON lines (some providers don't use SSE)
        elif line.strip():
            try:
                data = json.loads(line)
                return await self._parse_stream_data(data, request)
            except json.JSONDecodeError:
                return None
        
        return None
    
    @abstractmethod
    async def _parse_stream_data(
        self, 
        data: Dict[str, Any], 
        request: AIRequest
    ) -> Optional[StreamingChunk]:
        """
        Parse provider-specific streaming data.
        
        Args:
            data: Parsed JSON data from stream
            request: Original request
            
        Returns:
            StreamingChunk or None
        """
        pass
    
    def _create_chunk(
        self,
        request: AIRequest,
        content: Optional[str] = None,
        is_complete: bool = False,
        finish_reason: Optional[str] = None,
        tool_calls: Optional[list] = None,
        usage: Optional[Dict[str, int]] = None,
    ) -> StreamingChunk:
        """
        Create a standardized streaming chunk.
        
        Args:
            request: Original request
            content: Text content
            is_complete: Whether stream is complete
            finish_reason: Reason for completion
            tool_calls: Tool/function calls
            usage: Token usage information
            
        Returns:
            StreamingChunk
        """
        chunk = StreamingChunk(
            request_id=request.request_id,
            content=content,
            is_complete=is_complete,
            finish_reason=finish_reason,
            tool_calls=tool_calls,
            usage=usage,
        )
        
        # Track tokens if provided
        if usage:
            self._total_tokens = usage.get("total_tokens", self._total_tokens)
        
        # Add timing info for complete chunks
        if is_complete and self._start_time:
            elapsed = (datetime.now() - self._start_time).total_seconds()
            chunk.metadata = {
                "elapsed_seconds": elapsed,
                "total_tokens": self._total_tokens,
            }
        
        return chunk


class AnthropicStreamingHandler(BaseStreamingHandler):
    """Anthropic-specific streaming handler."""
    
    def __init__(self):
        super().__init__("Anthropic")
    
    async def _parse_stream_data(
        self, 
        data: Dict[str, Any], 
        request: AIRequest
    ) -> Optional[StreamingChunk]:
        """Parse Anthropic streaming data."""
        event_type = data.get("type")
        
        if event_type == "content_block_delta":
            delta = data.get("delta", {})
            if delta.get("type") == "text_delta":
                return self._create_chunk(
                    request=request,
                    content=delta.get("text", ""),
                )
        
        elif event_type == "message_delta":
            # Check for stop_reason in message_delta
            delta = data.get("delta", {})
            stop_reason = delta.get("stop_reason")
            usage = data.get("usage")
            
            if stop_reason:
                return self._create_chunk(
                    request=request,
                    is_complete=True,
                    finish_reason=stop_reason,
                    usage=usage,
                )
        
        elif event_type == "message_stop":
            # Message completion
            return self._create_chunk(
                request=request,
                is_complete=True,
                finish_reason=data.get("stop_reason"),
            )
        
        return None


class OpenAIStreamingHandler(BaseStreamingHandler):
    """OpenAI-specific streaming handler."""
    
    def __init__(self):
        super().__init__("OpenAI")
        self._accumulated_tools = {}
    
    async def _parse_stream_data(
        self, 
        data: Dict[str, Any], 
        request: AIRequest
    ) -> Optional[StreamingChunk]:
        """Parse OpenAI streaming data."""
        choices = data.get("choices", [])
        
        if not choices:
            return None
        
        choice = choices[0]
        delta = choice.get("delta", {})
        
        # Handle text content
        if "content" in delta:
            return self._create_chunk(
                request=request,
                content=delta["content"],
            )
        
        # Handle tool/function calls
        if "tool_calls" in delta:
            for tool_call in delta["tool_calls"]:
                idx = tool_call.get("index", 0)
                
                if idx not in self._accumulated_tools:
                    self._accumulated_tools[idx] = {
                        "id": tool_call.get("id"),
                        "type": tool_call.get("type", "function"),
                        "function": {
                            "name": "",
                            "arguments": "",
                        },
                    }
                
                if "function" in tool_call:
                    func = tool_call["function"]
                    if "name" in func:
                        self._accumulated_tools[idx]["function"]["name"] = func["name"]
                    if "arguments" in func:
                        self._accumulated_tools[idx]["function"]["arguments"] += func["arguments"]
        
        # Handle completion
        finish_reason = choice.get("finish_reason")
        if finish_reason:
            tool_calls = None
            if self._accumulated_tools:
                tool_calls = list(self._accumulated_tools.values())
                self._accumulated_tools.clear()
            
            usage = data.get("usage")
            
            return self._create_chunk(
                request=request,
                is_complete=True,
                finish_reason=finish_reason,
                tool_calls=tool_calls,
                usage=usage,
            )
        
        return None


class GoogleStreamingHandler(BaseStreamingHandler):
    """Google Gemini-specific streaming handler."""
    
    def __init__(self):
        super().__init__("Google")
    
    async def _parse_stream_data(
        self, 
        data: Dict[str, Any], 
        request: AIRequest
    ) -> Optional[StreamingChunk]:
        """Parse Google Gemini streaming data."""
        candidates = data.get("candidates", [])
        
        if not candidates:
            return None
        
        candidate = candidates[0]
        content = candidate.get("content", {})
        
        # Extract text from parts
        text_content = ""
        parts = content.get("parts", [])
        for part in parts:
            if "text" in part:
                text_content += part["text"]
        
        # Check for completion
        finish_reason = candidate.get("finishReason")
        
        if finish_reason:
            usage_metadata = data.get("usageMetadata", {})
            usage = None
            
            if usage_metadata:
                usage = {
                    "input_tokens": usage_metadata.get("promptTokenCount", 0),
                    "output_tokens": usage_metadata.get("candidatesTokenCount", 0),
                    "total_tokens": usage_metadata.get("totalTokenCount", 0),
                }
            
            return self._create_chunk(
                request=request,
                content=text_content if text_content else None,
                is_complete=True,
                finish_reason=finish_reason,
                usage=usage,
            )
        elif text_content:
            return self._create_chunk(
                request=request,
                content=text_content,
            )
        
        return None


# Factory function to get appropriate handler
def get_streaming_handler(provider_type: str) -> BaseStreamingHandler:
    """
    Get the appropriate streaming handler for a provider.
    
    Args:
        provider_type: Type of provider
        
    Returns:
        Appropriate streaming handler
    """
    handlers = {
        "anthropic": AnthropicStreamingHandler,
        "openai": OpenAIStreamingHandler,
        "google": GoogleStreamingHandler,
        "gemini": GoogleStreamingHandler,
    }
    
    handler_class = handlers.get(provider_type.lower())
    if not handler_class:
        raise ValueError(f"Unknown provider type: {provider_type}")
    
    return handler_class()