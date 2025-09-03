"""Advanced token counting with provider-specific accuracy."""

import json
import threading
from typing import Any, Dict, Generator, Iterator, List, Union

import structlog

from ..models import ProviderType, StreamingChunk
from ..token_estimator import TokenEstimator
from .models import TokenCountMetrics

logger = structlog.get_logger(__name__)


class AdvancedTokenCounter:
    """Advanced token counting with provider-specific accuracy."""
    
    def __init__(self):
        self._estimator = TokenEstimator()
        self._cache: Dict[str, int] = {}
        self._cache_lock = threading.Lock()
        
        # Provider-specific tokenizers
        self._tokenizers: Dict[ProviderType, Any] = {}
        self._initialize_tokenizers()
    
    def _initialize_tokenizers(self) -> None:
        """Initialize provider-specific tokenizers."""
        # Try to load tiktoken for OpenAI
        try:
            import tiktoken
            self._tokenizers[ProviderType.OPENAI] = tiktoken
            logger.info("Loaded tiktoken for accurate OpenAI token counting")
        except ImportError:
            logger.debug("tiktoken not available for OpenAI")
        
        # Try to load Anthropic tokenizer
        try:
            from anthropic import Anthropic
            # Note: Anthropic doesn't expose direct tokenizer, use heuristics
            logger.info("Anthropic SDK available for token estimation")
        except ImportError:
            logger.debug("Anthropic SDK not available")
    
    def _get_cache_key(self, content: str, provider: ProviderType, model: str) -> str:
        """Generate cache key for token counting."""
        # Use first 100 and last 100 chars for key to handle large content
        content_sig = content[:100] + content[-100:] if len(content) > 200 else content
        return f"{provider.value}:{model}:{hash(content_sig)}"
    
    def count_tokens(
        self,
        content: Union[str, List[Dict[str, Any]], Dict[str, Any]],
        provider: ProviderType,
        model: str,
        cache: bool = True,
    ) -> TokenCountMetrics:
        """Count tokens with detailed metrics.
        
        Args:
            content: Content to count tokens for
            provider: AI provider type
            model: Model identifier
            cache: Whether to use caching
            
        Returns:
            Detailed token count metrics
        """
        metrics = TokenCountMetrics()
        
        # Handle different content types
        if isinstance(content, str):
            metrics.text_tokens = self._count_text_tokens(content, provider, model, cache)
        elif isinstance(content, list):
            metrics = self._count_message_tokens(content, provider, model, cache)
        elif isinstance(content, dict):
            metrics = self._count_structured_tokens(content, provider, model, cache)
        
        return metrics
    
    def _count_text_tokens(
        self,
        text: str,
        provider: ProviderType,
        model: str,
        cache: bool = True,
    ) -> int:
        """Count tokens in plain text."""
        if cache:
            cache_key = self._get_cache_key(text, provider, model)
            with self._cache_lock:
                if cache_key in self._cache:
                    return self._cache[cache_key]
        
        # Use provider-specific counting
        if provider == ProviderType.OPENAI and ProviderType.OPENAI in self._tokenizers:
            count = self._count_openai_tokens(text, model)
        else:
            count = self._estimator.estimate_tokens(text, provider, model)
        
        if cache:
            with self._cache_lock:
                self._cache[cache_key] = count
        
        return count
    
    def _count_openai_tokens(self, text: str, model: str) -> int:
        """Count tokens using OpenAI's tiktoken."""
        tiktoken = self._tokenizers[ProviderType.OPENAI]
        
        # Get appropriate encoding
        try:
            if "gpt-4" in model or "gpt-3.5" in model:
                encoding = tiktoken.encoding_for_model(model)
            else:
                encoding = tiktoken.get_encoding("cl100k_base")
            
            tokens = encoding.encode(text)
            return len(tokens)
        except Exception as e:
            logger.warning(f"Failed to use tiktoken: {e}, falling back to estimation")
            return self._estimator.estimate_tokens(text, ProviderType.OPENAI, model)
    
    def _count_message_tokens(
        self,
        messages: List[Dict[str, Any]],
        provider: ProviderType,
        model: str,
        cache: bool = True,
    ) -> TokenCountMetrics:
        """Count tokens in message list."""
        metrics = TokenCountMetrics()
        
        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")
            
            # Handle different content types
            if isinstance(content, str):
                if role == "system":
                    metrics.system_prompt_tokens += self._count_text_tokens(
                        content, provider, model, cache
                    )
                else:
                    metrics.text_tokens += self._count_text_tokens(
                        content, provider, model, cache
                    )
            elif isinstance(content, list):
                # Handle multimodal content
                for item in content:
                    if isinstance(item, dict):
                        item_type = item.get("type", "")
                        if item_type == "text":
                            metrics.text_tokens += self._count_text_tokens(
                                item.get("text", ""), provider, model, cache
                            )
                        elif item_type == "image":
                            metrics.image_tokens += self._estimate_image_tokens(
                                item, provider, model
                            )
                        elif item_type == "audio":
                            metrics.audio_tokens += self._estimate_audio_tokens(
                                item, provider, model
                            )
            
            # Handle tool calls
            if "tool_calls" in message:
                metrics.tool_call_tokens += self._count_tool_tokens(
                    message["tool_calls"], provider, model, cache
                )
            
            if "tool_call_id" in message:
                # This is a tool response
                metrics.tool_response_tokens += self._count_text_tokens(
                    str(content), provider, model, cache
                )
        
        return metrics
    
    def _count_structured_tokens(
        self,
        data: Dict[str, Any],
        provider: ProviderType,
        model: str,
        cache: bool = True,
    ) -> TokenCountMetrics:
        """Count tokens in structured data."""
        # Convert to JSON string for estimation
        json_str = json.dumps(data, separators=(',', ':'))
        metrics = TokenCountMetrics()
        metrics.text_tokens = self._count_text_tokens(json_str, provider, model, cache)
        return metrics
    
    def _count_tool_tokens(
        self,
        tool_calls: List[Dict[str, Any]],
        provider: ProviderType,
        model: str,
        cache: bool = True,
    ) -> int:
        """Count tokens in tool calls."""
        total = 0
        for call in tool_calls:
            # Count function name and arguments
            json_str = json.dumps(call, separators=(',', ':'))
            total += self._count_text_tokens(json_str, provider, model, cache)
        return total
    
    def _estimate_image_tokens(
        self,
        image_data: Dict[str, Any],
        provider: ProviderType,
        model: str,
    ) -> int:
        """Estimate tokens for image content."""
        # Provider-specific image token estimation
        if provider == ProviderType.OPENAI:
            # GPT-4 Vision pricing model
            detail = image_data.get("detail", "auto")
            if detail == "low":
                return 85  # Fixed cost for low detail
            else:
                # High detail: base + tiles
                # Simplified estimation
                return 170 + 85 * 4  # Base + estimated tiles
        elif provider == ProviderType.ANTHROPIC:
            # Claude vision estimation
            return 250  # Approximate
        elif provider == ProviderType.GOOGLE:
            # Gemini vision estimation
            return 258  # Approximate
        return 200  # Default estimation
    
    def _estimate_audio_tokens(
        self,
        audio_data: Dict[str, Any],
        provider: ProviderType,
        model: str,
    ) -> int:
        """Estimate tokens for audio content."""
        # Simplified estimation based on duration
        duration_seconds = audio_data.get("duration_seconds", 0)
        if duration_seconds:
            # Approximate: 1 second of audio ≈ 50 tokens
            return int(duration_seconds * 50)
        return 1000  # Default estimation
    
    def count_streaming_tokens(
        self,
        chunks: Iterator[StreamingChunk],
        provider: ProviderType,
        model: str,
    ) -> Generator[TokenCountMetrics, None, TokenCountMetrics]:
        """Count tokens in streaming response.
        
        Yields token metrics for each chunk and returns total.
        """
        total_metrics = TokenCountMetrics()
        
        for chunk in chunks:
            chunk_metrics = TokenCountMetrics()
            
            if chunk.content:
                chunk_metrics.text_tokens = self._count_text_tokens(
                    chunk.content, provider, model, cache=False
                )
            
            if chunk.tool_calls:
                chunk_metrics.tool_call_tokens = self._count_tool_tokens(
                    chunk.tool_calls, provider, model, cache=False
                )
            
            total_metrics.text_tokens += chunk_metrics.text_tokens
            total_metrics.tool_call_tokens += chunk_metrics.tool_call_tokens
            
            yield chunk_metrics
        
        return total_metrics