"""Context translation and provider adaptation system."""

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, Union
from uuid import uuid4

from ..ai_providers.models import ProviderType
from .models import Context, ProviderContext, ConversationContext

logger = logging.getLogger(__name__)


class ProviderAdapter(ABC):
    """Abstract base class for provider-specific context adapters."""
    
    def __init__(self, provider_type: ProviderType):
        self.provider_type = provider_type
        self.supported_context_types = []
        self.format_version = "1.0"
    
    @abstractmethod
    async def to_provider_format(
        self, context: Context, options: Optional[Dict[str, Any]] = None
    ) -> ProviderContext:
        """Convert internal context to provider-specific format."""
        pass
    
    @abstractmethod
    async def from_provider_format(
        self, provider_context: ProviderContext, target_type: Type[Context] = Context
    ) -> Context:
        """Convert provider-specific format to internal context."""
        pass
    
    @abstractmethod
    def validate_provider_context(self, provider_context: ProviderContext) -> bool:
        """Validate provider context format and content."""
        pass
    
    @abstractmethod
    def get_size_estimate(self, context: Context) -> int:
        """Estimate the size of context in provider format."""
        pass


class AnthropicAdapter(ProviderAdapter):
    """Adapter for Anthropic Claude provider contexts."""
    
    def __init__(self):
        super().__init__(ProviderType.ANTHROPIC)
        self.supported_context_types = [ConversationContext]
        self.max_context_tokens = 100000  # Claude's context window
    
    async def to_provider_format(
        self, context: Context, options: Optional[Dict[str, Any]] = None
    ) -> ProviderContext:
        """Convert to Anthropic format with message structure."""
        try:
            provider_data = {}
            
            if isinstance(context, ConversationContext):
                # Convert to Anthropic message format
                messages = []
                system_message = None
                
                for msg in context.messages:
                    if msg.get("role") == "system":
                        # Anthropic handles system messages separately
                        system_message = msg.get("content", "")
                    else:
                        # Convert message format
                        anthropic_msg = {
                            "role": msg.get("role", "user"),
                            "content": msg.get("content", ""),
                        }
                        
                        # Handle tool calls and responses
                        if "tool_calls" in msg:
                            anthropic_msg["tool_calls"] = self._convert_tool_calls(
                                msg["tool_calls"]
                            )
                        
                        if "tool_results" in msg:
                            anthropic_msg["tool_results"] = msg["tool_results"]
                        
                        messages.append(anthropic_msg)
                
                provider_data = {
                    "messages": messages,
                    "system": system_message,
                    "model_parameters": context.model_parameters,
                    "conversation_metadata": {
                        "conversation_id": context.context_id,
                        "turn_count": context.current_turn,
                        "participants": context.participants,
                        "created_at": context.created_at.isoformat(),
                    },
                }
                
                # Add provider-specific settings
                if options:
                    provider_data.update(options)
            
            else:
                # Generic context conversion
                provider_data = {
                    "context_type": context.context_type.value,
                    "data": context.data,
                    "metadata": context.metadata,
                    "context_id": context.context_id,
                }
            
            return ProviderContext(
                provider_type=self.provider_type.value,
                context_data=provider_data,
                format_version=self.format_version,
                requires_translation=False,
                size_bytes=len(json.dumps(provider_data).encode('utf-8')),
            )
            
        except Exception as e:
            logger.error(f"Failed to convert to Anthropic format: {e}")
            raise
    
    async def from_provider_format(
        self, provider_context: ProviderContext, target_type: Type[Context] = Context
    ) -> Context:
        """Convert from Anthropic format to internal context."""
        try:
            data = provider_context.context_data
            
            if target_type == ConversationContext or "messages" in data:
                # Convert from Anthropic message format
                messages = []
                
                # Add system message if present
                if "system" in data and data["system"]:
                    messages.append({
                        "role": "system",
                        "content": data["system"],
                        "timestamp": provider_context.created_at.isoformat(),
                    })
                
                # Convert messages
                for msg in data.get("messages", []):
                    internal_msg = {
                        "role": msg.get("role", "user"),
                        "content": msg.get("content", ""),
                        "timestamp": provider_context.updated_at.isoformat(),
                    }
                    
                    # Handle tool calls and results
                    if "tool_calls" in msg:
                        internal_msg["tool_calls"] = self._convert_tool_calls_back(
                            msg["tool_calls"]
                        )
                    
                    if "tool_results" in msg:
                        internal_msg["tool_results"] = msg["tool_results"]
                    
                    messages.append(internal_msg)
                
                # Create conversation context
                conversation_metadata = data.get("conversation_metadata", {})
                
                return ConversationContext(
                    context_id=conversation_metadata.get("context_id", str(uuid4())),
                    messages=messages,
                    current_turn=conversation_metadata.get("turn_count", len(messages)),
                    participants=conversation_metadata.get("participants", []),
                    model_parameters=data.get("model_parameters", {}),
                    provider_settings=data.get("model_parameters", {}),
                    metadata=data.get("metadata", {}),
                )
            
            else:
                # Generic context conversion
                return target_type(
                    context_id=data.get("context_id", str(uuid4())),
                    data=data.get("data", {}),
                    metadata=data.get("metadata", {}),
                )
                
        except Exception as e:
            logger.error(f"Failed to convert from Anthropic format: {e}")
            raise
    
    def validate_provider_context(self, provider_context: ProviderContext) -> bool:
        """Validate Anthropic provider context."""
        try:
            data = provider_context.context_data
            
            # Check required fields for conversation contexts
            if "messages" in data:
                messages = data["messages"]
                if not isinstance(messages, list):
                    return False
                
                for msg in messages:
                    if not isinstance(msg, dict):
                        return False
                    
                    if "role" not in msg or "content" not in msg:
                        return False
                    
                    if msg["role"] not in ["user", "assistant", "system"]:
                        return False
            
            return True
            
        except Exception:
            return False
    
    def get_size_estimate(self, context: Context) -> int:
        """Estimate token usage for Anthropic."""
        if isinstance(context, ConversationContext):
            # Rough token estimation (1 token ≈ 4 characters)
            total_chars = 0
            for msg in context.messages:
                total_chars += len(str(msg.get("content", "")))
            
            estimated_tokens = total_chars // 4
            return min(estimated_tokens, self.max_context_tokens)
        
        # Generic estimation
        return len(json.dumps(context.data).encode('utf-8')) // 4
    
    def _convert_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert tool calls to Anthropic format."""
        converted = []
        for call in tool_calls:
            converted.append({
                "type": "function",
                "function": {
                    "name": call.get("name", ""),
                    "arguments": call.get("arguments", {}),
                },
                "id": call.get("id", str(uuid4())),
            })
        return converted
    
    def _convert_tool_calls_back(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert tool calls back from Anthropic format."""
        converted = []
        for call in tool_calls:
            if call.get("type") == "function" and "function" in call:
                func = call["function"]
                converted.append({
                    "name": func.get("name", ""),
                    "arguments": func.get("arguments", {}),
                    "id": call.get("id", str(uuid4())),
                })
        return converted


class OpenAIAdapter(ProviderAdapter):
    """Adapter for OpenAI provider contexts."""
    
    def __init__(self):
        super().__init__(ProviderType.OPENAI)
        self.supported_context_types = [ConversationContext]
        self.max_context_tokens = 128000  # GPT-4 context window
    
    async def to_provider_format(
        self, context: Context, options: Optional[Dict[str, Any]] = None
    ) -> ProviderContext:
        """Convert to OpenAI format."""
        try:
            provider_data = {}
            
            if isinstance(context, ConversationContext):
                # OpenAI uses a simpler message format
                messages = []
                
                for msg in context.messages:
                    openai_msg = {
                        "role": msg.get("role", "user"),
                        "content": msg.get("content", ""),
                    }
                    
                    # Handle tool calls
                    if "tool_calls" in msg:
                        openai_msg["tool_calls"] = msg["tool_calls"]
                    
                    if "tool_call_id" in msg:
                        openai_msg["tool_call_id"] = msg["tool_call_id"]
                    
                    messages.append(openai_msg)
                
                provider_data = {
                    "messages": messages,
                    "model_parameters": context.model_parameters,
                    "conversation_metadata": {
                        "conversation_id": context.context_id,
                        "turn_count": context.current_turn,
                        "participants": context.participants,
                    },
                }
            
            else:
                provider_data = {
                    "context_type": context.context_type.value,
                    "data": context.data,
                    "metadata": context.metadata,
                }
            
            return ProviderContext(
                provider_type=self.provider_type.value,
                context_data=provider_data,
                format_version=self.format_version,
                requires_translation=False,
                size_bytes=len(json.dumps(provider_data).encode('utf-8')),
            )
            
        except Exception as e:
            logger.error(f"Failed to convert to OpenAI format: {e}")
            raise
    
    async def from_provider_format(
        self, provider_context: ProviderContext, target_type: Type[Context] = Context
    ) -> Context:
        """Convert from OpenAI format."""
        try:
            data = provider_context.context_data
            
            if target_type == ConversationContext or "messages" in data:
                messages = []
                
                for msg in data.get("messages", []):
                    internal_msg = {
                        "role": msg.get("role", "user"),
                        "content": msg.get("content", ""),
                        "timestamp": provider_context.updated_at.isoformat(),
                    }
                    
                    if "tool_calls" in msg:
                        internal_msg["tool_calls"] = msg["tool_calls"]
                    
                    if "tool_call_id" in msg:
                        internal_msg["tool_call_id"] = msg["tool_call_id"]
                    
                    messages.append(internal_msg)
                
                conversation_metadata = data.get("conversation_metadata", {})
                
                return ConversationContext(
                    context_id=conversation_metadata.get("context_id", str(uuid4())),
                    messages=messages,
                    current_turn=conversation_metadata.get("turn_count", len(messages)),
                    participants=conversation_metadata.get("participants", []),
                    model_parameters=data.get("model_parameters", {}),
                    metadata=data.get("metadata", {}),
                )
            
            else:
                return target_type(
                    context_id=data.get("context_id", str(uuid4())),
                    data=data.get("data", {}),
                    metadata=data.get("metadata", {}),
                )
                
        except Exception as e:
            logger.error(f"Failed to convert from OpenAI format: {e}")
            raise
    
    def validate_provider_context(self, provider_context: ProviderContext) -> bool:
        """Validate OpenAI provider context."""
        try:
            data = provider_context.context_data
            
            if "messages" in data:
                messages = data["messages"]
                if not isinstance(messages, list):
                    return False
                
                for msg in messages:
                    if not isinstance(msg, dict):
                        return False
                    
                    if "role" not in msg or "content" not in msg:
                        return False
                    
                    if msg["role"] not in ["user", "assistant", "system", "tool"]:
                        return False
            
            return True
            
        except Exception:
            return False
    
    def get_size_estimate(self, context: Context) -> int:
        """Estimate token usage for OpenAI."""
        if isinstance(context, ConversationContext):
            # OpenAI token estimation (similar to Anthropic)
            total_chars = 0
            for msg in context.messages:
                total_chars += len(str(msg.get("content", "")))
            
            estimated_tokens = total_chars // 4
            return min(estimated_tokens, self.max_context_tokens)
        
        return len(json.dumps(context.data).encode('utf-8')) // 4


class GoogleAdapter(ProviderAdapter):
    """Adapter for Google provider contexts."""
    
    def __init__(self):
        super().__init__(ProviderType.GOOGLE)
        self.supported_context_types = [ConversationContext]
        self.max_context_tokens = 1000000  # Gemini context window
    
    async def to_provider_format(
        self, context: Context, options: Optional[Dict[str, Any]] = None
    ) -> ProviderContext:
        """Convert to Google/Gemini format."""
        try:
            provider_data = {}
            
            if isinstance(context, ConversationContext):
                # Google uses parts-based message format
                contents = []
                
                for msg in context.messages:
                    role = msg.get("role", "user")
                    # Map roles to Google format
                    if role == "assistant":
                        role = "model"
                    elif role == "system":
                        # Google doesn't have explicit system role, merge with first user message
                        continue
                    
                    content = {
                        "role": role,
                        "parts": [{"text": msg.get("content", "")}]
                    }
                    
                    # Handle tool calls (function calling)
                    if "tool_calls" in msg:
                        content["parts"] = []
                        for call in msg["tool_calls"]:
                            content["parts"].append({
                                "functionCall": {
                                    "name": call.get("name", ""),
                                    "args": call.get("arguments", {}),
                                }
                            })
                    
                    if "tool_results" in msg:
                        content["parts"] = []
                        for result in msg["tool_results"]:
                            content["parts"].append({
                                "functionResponse": {
                                    "name": result.get("name", ""),
                                    "response": result.get("content", ""),
                                }
                            })
                    
                    contents.append(content)
                
                provider_data = {
                    "contents": contents,
                    "generation_config": context.model_parameters,
                    "conversation_metadata": {
                        "conversation_id": context.context_id,
                        "turn_count": context.current_turn,
                        "participants": context.participants,
                    },
                }
            
            else:
                provider_data = {
                    "context_type": context.context_type.value,
                    "data": context.data,
                    "metadata": context.metadata,
                }
            
            return ProviderContext(
                provider_type=self.provider_type.value,
                context_data=provider_data,
                format_version=self.format_version,
                requires_translation=False,
                size_bytes=len(json.dumps(provider_data).encode('utf-8')),
            )
            
        except Exception as e:
            logger.error(f"Failed to convert to Google format: {e}")
            raise
    
    async def from_provider_format(
        self, provider_context: ProviderContext, target_type: Type[Context] = Context
    ) -> Context:
        """Convert from Google format."""
        try:
            data = provider_context.context_data
            
            if target_type == ConversationContext or "contents" in data:
                messages = []
                
                for content in data.get("contents", []):
                    role = content.get("role", "user")
                    # Map back from Google format
                    if role == "model":
                        role = "assistant"
                    
                    # Extract text from parts
                    text_content = ""
                    tool_calls = []
                    tool_results = []
                    
                    for part in content.get("parts", []):
                        if "text" in part:
                            text_content += part["text"]
                        elif "functionCall" in part:
                            call = part["functionCall"]
                            tool_calls.append({
                                "name": call.get("name", ""),
                                "arguments": call.get("args", {}),
                                "id": str(uuid4()),
                            })
                        elif "functionResponse" in part:
                            response = part["functionResponse"]
                            tool_results.append({
                                "name": response.get("name", ""),
                                "content": response.get("response", ""),
                            })
                    
                    internal_msg = {
                        "role": role,
                        "content": text_content,
                        "timestamp": provider_context.updated_at.isoformat(),
                    }
                    
                    if tool_calls:
                        internal_msg["tool_calls"] = tool_calls
                    if tool_results:
                        internal_msg["tool_results"] = tool_results
                    
                    messages.append(internal_msg)
                
                conversation_metadata = data.get("conversation_metadata", {})
                
                return ConversationContext(
                    context_id=conversation_metadata.get("context_id", str(uuid4())),
                    messages=messages,
                    current_turn=conversation_metadata.get("turn_count", len(messages)),
                    participants=conversation_metadata.get("participants", []),
                    model_parameters=data.get("generation_config", {}),
                    metadata=data.get("metadata", {}),
                )
            
            else:
                return target_type(
                    context_id=data.get("context_id", str(uuid4())),
                    data=data.get("data", {}),
                    metadata=data.get("metadata", {}),
                )
                
        except Exception as e:
            logger.error(f"Failed to convert from Google format: {e}")
            raise
    
    def validate_provider_context(self, provider_context: ProviderContext) -> bool:
        """Validate Google provider context."""
        try:
            data = provider_context.context_data
            
            if "contents" in data:
                contents = data["contents"]
                if not isinstance(contents, list):
                    return False
                
                for content in contents:
                    if not isinstance(content, dict):
                        return False
                    
                    if "role" not in content or "parts" not in content:
                        return False
                    
                    if content["role"] not in ["user", "model"]:
                        return False
                    
                    if not isinstance(content["parts"], list):
                        return False
            
            return True
            
        except Exception:
            return False
    
    def get_size_estimate(self, context: Context) -> int:
        """Estimate token usage for Google."""
        if isinstance(context, ConversationContext):
            total_chars = 0
            for msg in context.messages:
                total_chars += len(str(msg.get("content", "")))
            
            # Google typically has higher token efficiency
            estimated_tokens = total_chars // 3
            return min(estimated_tokens, self.max_context_tokens)
        
        return len(json.dumps(context.data).encode('utf-8')) // 3


class ContextTranslator:
    """High-performance context translation manager."""
    
    def __init__(self):
        # Register all available adapters
        self._adapters: Dict[ProviderType, ProviderAdapter] = {
            ProviderType.ANTHROPIC: AnthropicAdapter(),
            ProviderType.OPENAI: OpenAIAdapter(),
            ProviderType.GOOGLE: GoogleAdapter(),
        }
        
        # Translation cache
        self._translation_cache: Dict[str, ProviderContext] = {}
        self._cache_stats = {"hits": 0, "misses": 0}
        
        # Performance tracking
        self._translation_stats = {
            "translations_completed": 0,
            "translations_failed": 0,
            "avg_translation_time": 0.0,
            "cache_hit_rate": 0.0,
        }
        
        logger.info(
            "Context translator initialized",
            available_adapters=list(self._adapters.keys()),
        )
    
    async def translate_to_provider(
        self,
        context: Context,
        target_provider: ProviderType,
        options: Optional[Dict[str, Any]] = None,
        use_cache: bool = True,
    ) -> ProviderContext:
        """Translate context to provider-specific format."""
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = f"{context.context_id}:{context.current_version}:{target_provider.value}"
            
            if use_cache and cache_key in self._translation_cache:
                self._cache_stats["hits"] += 1
                return self._translation_cache[cache_key]
            
            self._cache_stats["misses"] += 1
            
            # Get appropriate adapter
            if target_provider not in self._adapters:
                raise ValueError(f"No adapter available for provider: {target_provider}")
            
            adapter = self._adapters[target_provider]
            
            # Perform translation
            provider_context = await adapter.to_provider_format(context, options)
            
            # Validate result
            if not adapter.validate_provider_context(provider_context):
                raise ValueError(f"Translation validation failed for {target_provider}")
            
            # Cache result
            if use_cache:
                self._translation_cache[cache_key] = provider_context
            
            # Update statistics
            execution_time = time.time() - start_time
            self._translation_stats["translations_completed"] += 1
            self._update_avg_translation_time(execution_time)
            self._update_cache_hit_rate()
            
            logger.debug(
                "Context translated to provider format",
                context_id=context.context_id,
                target_provider=target_provider.value,
                execution_time=execution_time,
                cached=False,
            )
            
            return provider_context
            
        except Exception as e:
            self._translation_stats["translations_failed"] += 1
            logger.error(f"Failed to translate context to {target_provider}: {e}")
            raise
    
    async def translate_from_provider(
        self,
        provider_context: ProviderContext,
        target_type: Type[Context] = Context,
    ) -> Context:
        """Translate from provider-specific format to internal context."""
        start_time = time.time()
        
        try:
            # Get provider type
            provider_type = ProviderType(provider_context.provider_type)
            
            if provider_type not in self._adapters:
                raise ValueError(f"No adapter available for provider: {provider_type}")
            
            adapter = self._adapters[provider_type]
            
            # Perform translation
            context = await adapter.from_provider_format(provider_context, target_type)
            
            # Update statistics
            execution_time = time.time() - start_time
            
            logger.debug(
                "Context translated from provider format",
                provider_type=provider_type.value,
                target_type=target_type.__name__,
                execution_time=execution_time,
            )
            
            return context
            
        except Exception as e:
            logger.error(f"Failed to translate context from provider format: {e}")
            raise
    
    async def migrate_context(
        self,
        context: Context,
        source_provider: ProviderType,
        target_provider: ProviderType,
        migration_options: Optional[Dict[str, Any]] = None,
    ) -> Context:
        """Migrate context from one provider format to another."""
        start_time = time.time()
        
        try:
            # First convert to internal format if needed
            if hasattr(context, 'provider_contexts') and source_provider.value in context.provider_contexts:
                # Extract provider-specific context
                source_context = context.provider_contexts[source_provider.value]
                internal_context = await self.translate_from_provider(source_context)
            else:
                internal_context = context
            
            # Then convert to target provider format
            target_provider_context = await self.translate_to_provider(
                internal_context, 
                target_provider,
                migration_options,
            )
            
            # Update context with new provider format
            if not hasattr(internal_context, 'provider_contexts'):
                internal_context.provider_contexts = {}
            
            internal_context.provider_contexts[target_provider.value] = target_provider_context
            
            # Remove old provider context if requested
            if migration_options and migration_options.get("remove_source", False):
                if source_provider.value in internal_context.provider_contexts:
                    del internal_context.provider_contexts[source_provider.value]
            
            execution_time = time.time() - start_time
            
            logger.info(
                "Context migrated between providers",
                context_id=internal_context.context_id,
                source_provider=source_provider.value,
                target_provider=target_provider.value,
                execution_time=execution_time,
            )
            
            return internal_context
            
        except Exception as e:
            logger.error(f"Failed to migrate context: {e}")
            raise
    
    def get_adapter(self, provider_type: ProviderType) -> Optional[ProviderAdapter]:
        """Get adapter for a specific provider."""
        return self._adapters.get(provider_type)
    
    def register_adapter(self, adapter: ProviderAdapter) -> None:
        """Register a new provider adapter."""
        self._adapters[adapter.provider_type] = adapter
        logger.info(f"Registered adapter for {adapter.provider_type}")
    
    def estimate_translation_size(
        self, context: Context, target_provider: ProviderType
    ) -> Optional[int]:
        """Estimate the size of context in target provider format."""
        adapter = self._adapters.get(target_provider)
        if adapter:
            return adapter.get_size_estimate(context)
        return None
    
    def clear_cache(self) -> None:
        """Clear translation cache."""
        self._translation_cache.clear()
        logger.debug("Translation cache cleared")
    
    def _update_avg_translation_time(self, new_time: float) -> None:
        """Update average translation time."""
        count = self._translation_stats["translations_completed"]
        current_avg = self._translation_stats["avg_translation_time"]
        
        if count > 1:
            self._translation_stats["avg_translation_time"] = (
                (current_avg * (count - 1) + new_time) / count
            )
        else:
            self._translation_stats["avg_translation_time"] = new_time
    
    def _update_cache_hit_rate(self) -> None:
        """Update cache hit rate."""
        total = self._cache_stats["hits"] + self._cache_stats["misses"]
        if total > 0:
            self._translation_stats["cache_hit_rate"] = (
                self._cache_stats["hits"] / total
            )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get translation performance statistics."""
        return {
            "translation_stats": self._translation_stats,
            "cache_stats": self._cache_stats,
            "available_providers": [p.value for p in self._adapters.keys()],
            "cache_size": len(self._translation_cache),
        }
    
    def cleanup(self) -> None:
        """Clean up resources."""
        self.clear_cache()
        logger.info("Context translator cleaned up")