"""Enhanced token counting system for all AI providers with accurate estimation."""

import json
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import tiktoken
from structlog import get_logger

from .models import ProviderType

logger = get_logger(__name__)


class TokenCountMethod(Enum):
    """Token counting methods for different providers."""
    
    TIKTOKEN = "tiktoken"  # OpenAI's tokenizer
    SENTENCEPIECE = "sentencepiece"  # Google's tokenizer
    CLAUDE_HEURISTIC = "claude_heuristic"  # Anthropic's estimation
    CHARACTER_BASED = "character_based"  # Fallback method
    OLLAMA_ESTIMATION = "ollama_estimation"  # Local model estimation


@dataclass
class TokenizationConfig:
    """Configuration for tokenization by provider and model."""
    
    provider_type: ProviderType
    model_pattern: str  # Regex pattern for model matching
    method: TokenCountMethod
    encoding_name: Optional[str] = None  # For tiktoken
    chars_per_token: float = 4.0  # For character-based estimation
    message_overhead: int = 4  # Additional tokens per message
    tool_overhead: int = 10  # Additional tokens per tool
    vision_base_tokens: int = 85  # Base tokens for vision processing
    vision_detail_multiplier: float = 1.4  # Multiplier for high detail images


class BaseTokenEstimator(ABC):
    """Abstract base class for token estimation."""
    
    @abstractmethod
    def estimate_input_tokens(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None) -> int:
        """Estimate input token count."""
        pass
    
    @abstractmethod
    def estimate_output_tokens(self, content: str, tool_calls: Optional[List[Dict[str, Any]]] = None) -> int:
        """Estimate output token count."""
        pass


class TikTokenEstimator(BaseTokenEstimator):
    """OpenAI tiktoken-based token estimation."""
    
    def __init__(self, encoding_name: str = "cl100k_base"):
        try:
            self.encoding = tiktoken.get_encoding(encoding_name)
        except Exception as e:
            logger.warning(f"Failed to load tiktoken encoding {encoding_name}: {e}")
            self.encoding = tiktoken.get_encoding("cl100k_base")  # Fallback
    
    def estimate_input_tokens(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None) -> int:
        """Estimate tokens using tiktoken for OpenAI-compatible models."""
        total_tokens = 0
        
        # Count message tokens
        for message in messages:
            # Start with message formatting overhead (role, etc.)
            total_tokens += 4  # Basic message overhead
            
            role = message.get("role", "")
            total_tokens += len(self.encoding.encode(role))
            
            content = message.get("content", "")
            if isinstance(content, str):
                total_tokens += len(self.encoding.encode(content))
            elif isinstance(content, list):
                # Multimodal content
                for item in content:
                    if item.get("type") == "text":
                        text_content = item.get("text", "")
                        total_tokens += len(self.encoding.encode(text_content))
                    elif item.get("type") == "image_url":
                        # Vision model token estimation
                        detail = item.get("image_url", {}).get("detail", "auto")
                        if detail == "high":
                            total_tokens += int(85 * 1.4)  # High detail multiplier
                        else:
                            total_tokens += 85  # Base vision tokens
        
        # Add tool definition tokens
        if tools:
            tool_tokens = 0
            for tool in tools:
                tool_json = json.dumps(tool, separators=(',', ':'))
                tool_tokens += len(self.encoding.encode(tool_json))
            total_tokens += tool_tokens + (len(tools) * 10)  # Tool overhead
        
        return total_tokens
    
    def estimate_output_tokens(self, content: str, tool_calls: Optional[List[Dict[str, Any]]] = None) -> int:
        """Estimate output tokens for generated content."""
        tokens = len(self.encoding.encode(content))
        
        if tool_calls:
            for tool_call in tool_calls:
                # Tool call structure overhead
                tokens += 15
                
                # Function name and arguments
                if "function" in tool_call:
                    func_data = tool_call["function"]
                    tokens += len(self.encoding.encode(func_data.get("name", "")))
                    arguments = func_data.get("arguments", "")
                    if isinstance(arguments, str):
                        tokens += len(self.encoding.encode(arguments))
                    else:
                        tokens += len(self.encoding.encode(json.dumps(arguments, separators=(',', ':'))))
        
        return tokens


class ClaudeHeuristicEstimator(BaseTokenEstimator):
    """Heuristic-based estimation for Anthropic Claude models."""
    
    def __init__(self, chars_per_token: float = 3.5):
        self.chars_per_token = chars_per_token
    
    def estimate_input_tokens(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None) -> int:
        """Estimate tokens using character-based heuristics optimized for Claude."""
        total_chars = 0
        
        for message in messages:
            # Add role overhead
            total_chars += 10  # Role formatting
            
            content = message.get("content", "")
            if isinstance(content, str):
                total_chars += len(content)
            elif isinstance(content, list):
                for item in content:
                    if item.get("type") == "text":
                        total_chars += len(item.get("text", ""))
                    elif item.get("type") == "image":
                        # Claude vision token estimation
                        total_chars += 200 * self.chars_per_token  # ~200 tokens for images
        
        # Tool definitions
        if tools:
            for tool in tools:
                tool_json = json.dumps(tool, separators=(',', ':'))
                total_chars += len(tool_json) + 40  # Tool overhead
        
        return int(total_chars / self.chars_per_token)
    
    def estimate_output_tokens(self, content: str, tool_calls: Optional[List[Dict[str, Any]]] = None) -> int:
        """Estimate output tokens for Claude responses."""
        tokens = int(len(content) / self.chars_per_token)
        
        if tool_calls:
            for tool_call in tool_calls:
                # Claude tool use format estimation
                tool_content = json.dumps(tool_call, separators=(',', ':'))
                tokens += int(len(tool_content) / self.chars_per_token) + 20  # Tool overhead
        
        return tokens


class GoogleTokenEstimator(BaseTokenEstimator):
    """Token estimation for Google's models."""
    
    def __init__(self, chars_per_token: float = 4.2):
        self.chars_per_token = chars_per_token
    
    def estimate_input_tokens(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None) -> int:
        """Estimate tokens for Google models."""
        total_chars = 0
        
        for message in messages:
            role = message.get("role", "")
            total_chars += len(role) + 5  # Role formatting
            
            content = message.get("content", "")
            if isinstance(content, str):
                total_chars += len(content)
            elif isinstance(content, list):
                for item in content:
                    if item.get("type") == "text":
                        total_chars += len(item.get("text", ""))
                    elif item.get("type") == "image":
                        # Google vision token estimation
                        total_chars += 150 * self.chars_per_token  # ~150 tokens
        
        # Function calling overhead
        if tools:
            for tool in tools:
                tool_json = json.dumps(tool, separators=(',', ':'))
                total_chars += len(tool_json) + 30
        
        return int(total_chars / self.chars_per_token)
    
    def estimate_output_tokens(self, content: str, tool_calls: Optional[List[Dict[str, Any]]] = None) -> int:
        """Estimate output tokens for Google responses."""
        tokens = int(len(content) / self.chars_per_token)
        
        if tool_calls:
            for tool_call in tool_calls:
                tool_content = json.dumps(tool_call, separators=(',', ':'))
                tokens += int(len(tool_content) / self.chars_per_token) + 15
        
        return tokens


class OllamaTokenEstimator(BaseTokenEstimator):
    """Token estimation for local Ollama models."""
    
    def __init__(self, chars_per_token: float = 4.5):
        self.chars_per_token = chars_per_token
    
    def estimate_input_tokens(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None) -> int:
        """Estimate tokens for Ollama models (more conservative)."""
        total_chars = 0
        
        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")
            total_chars += len(role) + len(str(content)) + 8  # Conservative overhead
        
        if tools:
            # Ollama tool support varies, use conservative estimation
            for tool in tools:
                tool_json = json.dumps(tool, separators=(',', ':'))
                total_chars += len(tool_json) + 50  # Higher overhead
        
        return int(total_chars / self.chars_per_token)
    
    def estimate_output_tokens(self, content: str, tool_calls: Optional[List[Dict[str, Any]]] = None) -> int:
        """Conservative estimation for Ollama output."""
        tokens = int(len(content) / self.chars_per_token)
        
        if tool_calls:
            for tool_call in tool_calls:
                tokens += int(len(json.dumps(tool_call)) / self.chars_per_token) + 25
        
        return tokens


class EnhancedTokenEstimator:
    """Enhanced token estimator supporting multiple providers with accuracy optimization."""
    
    def __init__(self):
        self.configurations = self._load_configurations()
        self.estimators = {
            TokenCountMethod.TIKTOKEN: TikTokenEstimator(),
            TokenCountMethod.CLAUDE_HEURISTIC: ClaudeHeuristicEstimator(),
            TokenCountMethod.SENTENCEPIECE: GoogleTokenEstimator(),
            TokenCountMethod.OLLAMA_ESTIMATION: OllamaTokenEstimator(),
        }
        
        # Cache for expensive estimations
        self._estimation_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
    
    def _load_configurations(self) -> List[TokenizationConfig]:
        """Load tokenization configurations for different models."""
        return [
            # OpenAI Models
            TokenizationConfig(
                provider_type=ProviderType.OPENAI,
                model_pattern=r"gpt-4|gpt-3.5",
                method=TokenCountMethod.TIKTOKEN,
                encoding_name="cl100k_base"
            ),
            TokenizationConfig(
                provider_type=ProviderType.OPENAI,
                model_pattern=r"text-davinci|text-curie|text-babbage|text-ada",
                method=TokenCountMethod.TIKTOKEN,
                encoding_name="p50k_base"
            ),
            
            # Anthropic Models
            TokenizationConfig(
                provider_type=ProviderType.ANTHROPIC,
                model_pattern=r"claude-3|claude-2|claude-instant",
                method=TokenCountMethod.CLAUDE_HEURISTIC,
                chars_per_token=3.5
            ),
            
            # Google Models
            TokenizationConfig(
                provider_type=ProviderType.GOOGLE,
                model_pattern=r"gemini|palm|bison",
                method=TokenCountMethod.SENTENCEPIECE,
                chars_per_token=4.2
            ),
            
            # Ollama Models (fallback pattern)
            TokenizationConfig(
                provider_type=ProviderType.ANTHROPIC,  # Using ANTHROPIC as Ollama isn't in enum
                model_pattern=r"llama|mistral|codellama|vicuna",
                method=TokenCountMethod.OLLAMA_ESTIMATION,
                chars_per_token=4.5
            ),
        ]
    
    def get_estimator_config(self, provider_type: ProviderType, model: str) -> Optional[TokenizationConfig]:
        """Get the appropriate tokenization configuration for a model."""
        for config in self.configurations:
            if config.provider_type == provider_type and re.search(config.model_pattern, model, re.IGNORECASE):
                return config
        
        # Fallback configuration
        return TokenizationConfig(
            provider_type=provider_type,
            model_pattern=".*",
            method=TokenCountMethod.CHARACTER_BASED,
            chars_per_token=4.0
        )
    
    def estimate_request_tokens(
        self,
        provider_type: ProviderType,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        max_output_tokens: Optional[int] = None
    ) -> Tuple[int, int]:
        """
        Estimate input and potential output tokens for a request.
        
        Returns:
            Tuple of (input_tokens, estimated_output_tokens)
        """
        # Create cache key
        cache_key = self._create_cache_key(provider_type, model, messages, tools, max_output_tokens)
        
        if cache_key in self._estimation_cache:
            self._cache_hits += 1
            return self._estimation_cache[cache_key]
        
        self._cache_misses += 1
        
        config = self.get_estimator_config(provider_type, model)
        estimator = self.estimators.get(config.method)
        
        if not estimator:
            # Fallback to character-based estimation
            input_tokens = self._character_based_estimation(messages, tools)
            output_tokens = max_output_tokens or 150  # Conservative estimate
        else:
            input_tokens = estimator.estimate_input_tokens(messages, tools)
            # For output estimation, we use max_output_tokens or a reasonable default
            output_tokens = max_output_tokens or min(input_tokens // 2, 500)
        
        result = (input_tokens, output_tokens)
        self._estimation_cache[cache_key] = result
        
        # Limit cache size
        if len(self._estimation_cache) > 1000:
            # Remove oldest entries (simple FIFO)
            keys_to_remove = list(self._estimation_cache.keys())[:100]
            for key in keys_to_remove:
                del self._estimation_cache[key]
        
        return result
    
    def estimate_response_tokens(
        self,
        provider_type: ProviderType,
        model: str,
        content: str,
        tool_calls: Optional[List[Dict[str, Any]]] = None
    ) -> int:
        """Estimate actual response tokens."""
        config = self.get_estimator_config(provider_type, model)
        estimator = self.estimators.get(config.method)
        
        if not estimator:
            return int(len(content) / config.chars_per_token)
        
        return estimator.estimate_output_tokens(content, tool_calls)
    
    def _create_cache_key(
        self,
        provider_type: ProviderType,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        max_output_tokens: Optional[int]
    ) -> str:
        """Create a cache key for token estimation."""
        # Create a hash of the input parameters
        import hashlib
        
        content_hash = hashlib.md5()
        content_hash.update(f"{provider_type.value}:{model}".encode())
        content_hash.update(str(len(messages)).encode())
        content_hash.update(str(sum(len(str(m.get('content', ''))) for m in messages)).encode())
        content_hash.update(str(len(tools) if tools else 0).encode())
        content_hash.update(str(max_output_tokens or 0).encode())
        
        return content_hash.hexdigest()
    
    def _character_based_estimation(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        chars_per_token: float = 4.0
    ) -> int:
        """Fallback character-based token estimation."""
        total_chars = 0
        
        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")
            total_chars += len(role) + len(str(content)) + 10  # Overhead
        
        if tools:
            for tool in tools:
                tool_json = json.dumps(tool, separators=(',', ':'))
                total_chars += len(tool_json) + 20  # Tool overhead
        
        return int(total_chars / chars_per_token)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = (self._cache_hits / total_requests) if total_requests > 0 else 0.0
        
        return {
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": hit_rate,
            "cache_size": len(self._estimation_cache),
        }
    
    def clear_cache(self) -> None:
        """Clear the estimation cache."""
        self._estimation_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        logger.info("Token estimation cache cleared")