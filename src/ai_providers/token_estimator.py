"""Token estimation utilities for AI providers."""

import re
from typing import Any, Dict, List, Optional, Union

from structlog import get_logger

from .models import ProviderType

logger = get_logger(__name__)


class TokenEstimator:
    """Advanced token estimation for various AI providers.
    
    Uses provider-specific heuristics and optional tiktoken for accurate estimates.
    Falls back to character-based estimation when libraries are unavailable.
    """
    
    def __init__(self):
        self._tiktoken_available = False
        self._tiktoken_encoders = {}
        self._anthropic_tokenizer = None
        
        # Try to import tiktoken for OpenAI
        try:
            import tiktoken
            self._tiktoken_available = True
            self._tiktoken = tiktoken
            logger.info("Tiktoken available for accurate OpenAI token estimation")
        except ImportError:
            logger.warning("Tiktoken not available, using heuristic estimation for OpenAI")
        
        # Try to import anthropic tokenizer
        try:
            from anthropic import Anthropic
            # Anthropic doesn't expose tokenizer directly, use heuristics
            self._anthropic_available = True
        except ImportError:
            self._anthropic_available = False
    
    def estimate_tokens(
        self,
        content: Union[str, List[Dict[str, Any]], Dict[str, Any]],
        provider_type: ProviderType,
        model: Optional[str] = None,
    ) -> int:
        """Estimate token count for content based on provider type.
        
        Args:
            content: Content to estimate tokens for
            provider_type: AI provider type
            model: Optional model name for more accurate estimation
            
        Returns:
            Estimated token count
        """
        # Convert content to string if needed
        text = self._content_to_text(content)
        
        if provider_type == ProviderType.OPENAI:
            return self._estimate_openai_tokens(text, model)
        elif provider_type == ProviderType.ANTHROPIC:
            return self._estimate_anthropic_tokens(text, model)
        elif provider_type == ProviderType.GOOGLE:
            return self._estimate_google_tokens(text, model)
        else:
            # Fallback to generic estimation
            return self._estimate_generic_tokens(text)
    
    def _content_to_text(self, content: Union[str, List, Dict]) -> str:
        """Convert various content types to text string.
        
        Args:
            content: Content to convert
            
        Returns:
            Text representation
        """
        if isinstance(content, str):
            return content
        
        if isinstance(content, list):
            # Handle message list
            text_parts = []
            for item in content:
                if isinstance(item, dict):
                    # Extract text from message dict
                    if "content" in item:
                        text_parts.append(str(item["content"]))
                    if "text" in item:
                        text_parts.append(str(item["text"]))
                else:
                    text_parts.append(str(item))
            return " ".join(text_parts)
        
        if isinstance(content, dict):
            # Extract text from dict
            if "content" in content:
                return str(content["content"])
            if "text" in content:
                return str(content["text"])
            # Convert entire dict to string
            return str(content)
        
        return str(content)
    
    def _estimate_openai_tokens(self, text: str, model: Optional[str] = None) -> int:
        """Estimate tokens for OpenAI models.
        
        Args:
            text: Text to estimate
            model: OpenAI model name
            
        Returns:
            Estimated token count
        """
        if self._tiktoken_available:
            try:
                # Use tiktoken for accurate estimation
                encoding_name = self._get_openai_encoding(model)
                
                if encoding_name not in self._tiktoken_encoders:
                    self._tiktoken_encoders[encoding_name] = \
                        self._tiktoken.get_encoding(encoding_name)
                
                encoder = self._tiktoken_encoders[encoding_name]
                tokens = encoder.encode(text)
                return len(tokens)
                
            except Exception as e:
                logger.warning(f"Tiktoken encoding failed: {e}, using fallback")
        
        # Fallback: OpenAI heuristics
        # GPT models typically use ~1 token per 4 characters for English
        # Adjust for model specifics
        if model and "gpt-4" in model:
            # GPT-4 is slightly more efficient
            chars_per_token = 3.8
        elif model and "gpt-3.5" in model:
            chars_per_token = 4.0
        else:
            chars_per_token = 4.0
        
        # Account for special tokens and formatting
        base_tokens = len(text) / chars_per_token
        
        # Add overhead for message structure
        overhead = 4  # Typical message wrapper tokens
        
        return int(base_tokens + overhead)
    
    def _estimate_anthropic_tokens(self, text: str, model: Optional[str] = None) -> int:
        """Estimate tokens for Anthropic Claude models.
        
        Args:
            text: Text to estimate
            model: Anthropic model name
            
        Returns:
            Estimated token count
        """
        # Anthropic uses a similar tokenization to GPT models
        # but with some differences in efficiency
        
        # Claude models are generally more efficient with tokens
        if model and "claude-3-opus" in model:
            chars_per_token = 3.5
        elif model and "claude-3-sonnet" in model:
            chars_per_token = 3.7
        elif model and "claude-3-haiku" in model:
            chars_per_token = 3.9
        elif model and "claude-2" in model:
            chars_per_token = 3.8
        else:
            chars_per_token = 3.7
        
        # Account for Claude's specific formatting
        base_tokens = len(text) / chars_per_token
        
        # Claude has less overhead than GPT models
        overhead = 2
        
        # Adjust for complex content
        if self._has_code(text):
            # Code tends to use more tokens
            base_tokens *= 1.15
        
        if self._has_special_characters(text):
            # Special characters can increase token count
            base_tokens *= 1.1
        
        return int(base_tokens + overhead)
    
    def _estimate_google_tokens(self, text: str, model: Optional[str] = None) -> int:
        """Estimate tokens for Google Gemini models.
        
        Args:
            text: Text to estimate
            model: Google model name
            
        Returns:
            Estimated token count
        """
        # Gemini models have different tokenization characteristics
        if model and "gemini-1.5-pro" in model:
            # Very efficient tokenization for long contexts
            chars_per_token = 3.2
        elif model and "gemini-1.5-flash" in model:
            chars_per_token = 3.4
        elif model and "gemini-pro" in model:
            chars_per_token = 3.6
        else:
            chars_per_token = 3.5
        
        base_tokens = len(text) / chars_per_token
        
        # Gemini has minimal overhead
        overhead = 1
        
        # Adjust for content type
        if self._has_markdown(text):
            # Markdown formatting affects tokenization
            base_tokens *= 1.05
        
        return int(base_tokens + overhead)
    
    def _estimate_generic_tokens(self, text: str) -> int:
        """Generic token estimation fallback.
        
        Args:
            text: Text to estimate
            
        Returns:
            Estimated token count
        """
        # Conservative generic estimation
        # Most modern models average around 3.5-4 chars per token
        chars_per_token = 3.75
        
        # Word-based adjustment
        word_count = len(text.split())
        char_count = len(text)
        
        # Use hybrid approach for better accuracy
        char_estimate = char_count / chars_per_token
        word_estimate = word_count * 1.3  # Average 1.3 tokens per word
        
        # Take weighted average
        estimated = (char_estimate * 0.7) + (word_estimate * 0.3)
        
        return int(estimated)
    
    def _get_openai_encoding(self, model: Optional[str] = None) -> str:
        """Get the appropriate tiktoken encoding for OpenAI model.
        
        Args:
            model: OpenAI model name
            
        Returns:
            Encoding name for tiktoken
        """
        if not model:
            return "cl100k_base"  # Default for recent models
        
        model = model.lower()
        
        # GPT-4 and GPT-3.5-turbo use cl100k_base
        if "gpt-4" in model or "gpt-3.5-turbo" in model:
            return "cl100k_base"
        
        # Older models
        if "text-davinci" in model:
            return "p50k_base"
        
        if "code-davinci" in model:
            return "p50k_base"
        
        # Default to most recent encoding
        return "cl100k_base"
    
    def _has_code(self, text: str) -> bool:
        """Check if text contains code patterns.
        
        Args:
            text: Text to check
            
        Returns:
            True if code patterns detected
        """
        code_patterns = [
            r'def\s+\w+\s*\(',  # Python function
            r'function\s+\w+\s*\(',  # JavaScript function
            r'class\s+\w+',  # Class definition
            r'import\s+\w+',  # Import statement
            r'{\s*["\']',  # JSON-like
            r'<[^>]+>',  # HTML/XML tags
            r'\w+\s*=\s*["\']',  # Variable assignment
        ]
        
        for pattern in code_patterns:
            if re.search(pattern, text):
                return True
        
        return False
    
    def _has_special_characters(self, text: str) -> bool:
        """Check if text has high density of special characters.
        
        Args:
            text: Text to check
            
        Returns:
            True if high special character density
        """
        if not text:
            return False
        
        special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
        ratio = special_chars / len(text)
        
        return ratio > 0.15  # More than 15% special characters
    
    def _has_markdown(self, text: str) -> bool:
        """Check if text contains markdown formatting.
        
        Args:
            text: Text to check
            
        Returns:
            True if markdown patterns detected
        """
        markdown_patterns = [
            r'^#{1,6}\s',  # Headers
            r'\*\*[^*]+\*\*',  # Bold
            r'\*[^*]+\*',  # Italic
            r'\[([^\]]+)\]\([^\)]+\)',  # Links
            r'```[^`]*```',  # Code blocks
            r'^\s*[-*+]\s',  # Lists
        ]
        
        for pattern in markdown_patterns:
            if re.search(pattern, text, re.MULTILINE):
                return True
        
        return False
    
    def estimate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        provider_type: ProviderType,
        model: Optional[str] = None,
    ) -> float:
        """Estimate cost based on token counts.
        
        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            provider_type: AI provider type
            model: Optional model name
            
        Returns:
            Estimated cost in USD
        """
        # Basic cost estimation (should be updated with actual pricing)
        pricing = self._get_pricing(provider_type, model)
        
        input_cost = (input_tokens / 1000) * pricing["input"]
        output_cost = (output_tokens / 1000) * pricing["output"]
        
        return input_cost + output_cost
    
    def _get_pricing(
        self,
        provider_type: ProviderType,
        model: Optional[str] = None,
    ) -> Dict[str, float]:
        """Get pricing per 1K tokens for provider/model.
        
        Args:
            provider_type: AI provider type
            model: Optional model name
            
        Returns:
            Dict with input/output pricing
        """
        # Default pricing (should be kept updated)
        default_pricing = {
            ProviderType.OPENAI: {
                "gpt-4": {"input": 0.03, "output": 0.06},
                "gpt-4-turbo": {"input": 0.01, "output": 0.03},
                "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
                "default": {"input": 0.002, "output": 0.002},
            },
            ProviderType.ANTHROPIC: {
                "claude-3-opus": {"input": 0.015, "output": 0.075},
                "claude-3-sonnet": {"input": 0.003, "output": 0.015},
                "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
                "default": {"input": 0.008, "output": 0.024},
            },
            ProviderType.GOOGLE: {
                "gemini-1.5-pro": {"input": 0.00125, "output": 0.005},
                "gemini-1.5-flash": {"input": 0.000075, "output": 0.0003},
                "gemini-pro": {"input": 0.0005, "output": 0.0015},
                "default": {"input": 0.001, "output": 0.002},
            },
        }
        
        provider_pricing = default_pricing.get(provider_type, {})
        
        if model:
            for model_prefix, pricing in provider_pricing.items():
                if model_prefix in model.lower():
                    return pricing
        
        return provider_pricing.get("default", {"input": 0.001, "output": 0.002})


# Global instance for convenience
_default_estimator = TokenEstimator()


def estimate_tokens(
    content: Union[str, List[Dict[str, Any]], Dict[str, Any]],
    provider_type: ProviderType,
    model: Optional[str] = None,
) -> int:
    """Convenience function for token estimation.
    
    Args:
        content: Content to estimate tokens for
        provider_type: AI provider type
        model: Optional model name
        
    Returns:
        Estimated token count
    """
    return _default_estimator.estimate_tokens(content, provider_type, model)