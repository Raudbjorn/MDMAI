"""AI Provider Integration module for TTRPG Assistant MCP Server.

This module provides a unified interface for integrating multiple AI providers
(Anthropic Claude, OpenAI GPT, Google Gemini) with the MCP Bridge infrastructure.

Key Features:
- Provider abstraction layer with unified interface
- Tool format translation between MCP and provider-specific formats  
- Cost optimization with usage tracking and budget enforcement
- Provider selection strategies based on capability and cost
- Response streaming and unified error handling
- Extensible architecture for future provider additions

Architecture Components:
- AbstractProvider: Base class for all AI providers
- ProviderRegistry: Manages available providers and selection
- CostOptimizer: Tracks usage and optimizes costs
- ToolTranslator: Converts between MCP and provider tool formats
- StreamingManager: Handles response streaming across providers
- ErrorHandler: Unified error handling and retry logic
"""

from .abstract_provider import AbstractProvider, ProviderCapability, ProviderStatus
from .anthropic_provider import AnthropicProvider
from .cost_optimizer import CostOptimizer, UsageTracker
from .error_handler import AIProviderError, ErrorHandler
from .google_provider import GoogleProvider
from .openai_provider import OpenAIProvider
from .provider_manager import AIProviderManager
from .provider_registry import ProviderRegistry
from .streaming_manager import StreamingManager, StreamingResponse
from .tool_translator import ToolTranslator

__all__ = [
    "AbstractProvider",
    "ProviderCapability", 
    "ProviderStatus",
    "AnthropicProvider",
    "OpenAIProvider", 
    "GoogleProvider",
    "AIProviderManager",
    "ProviderRegistry",
    "CostOptimizer",
    "UsageTracker", 
    "ToolTranslator",
    "StreamingManager",
    "StreamingResponse",
    "ErrorHandler",
    "AIProviderError",
]