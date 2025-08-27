"""Integration layer for Context Management System with existing TTRPG Assistant components."""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Union

from ..ai_providers.models import ProviderType
from ..ai_providers.provider_manager import ProviderManager
from ..bridge.models import BridgeMessage, MCPSession
from ..bridge.session_manager import SessionManager as BridgeSessionManager
from ..security.enhanced_security_manager import EnhancedSecurityManager
from .context_manager import ContextManager
from .models import ConversationContext, SessionContext, CollaborativeContext, ContextEvent
from .config import get_context_config, validate_context_config, create_context_directories

logger = logging.getLogger(__name__)


class ContextIntegrationManager:
    """Integration manager for connecting context system with existing components."""
    
    def __init__(
        self,
        context_manager: ContextManager,
        provider_manager: Optional[ProviderManager] = None,
        bridge_session_manager: Optional[BridgeSessionManager] = None,
        security_manager: Optional[EnhancedSecurityManager] = None,
    ):
        self.context_manager = context_manager
        self.provider_manager = provider_manager
        self.bridge_session_manager = bridge_session_manager
        self.security_manager = security_manager
        
        self.config = get_context_config()
        
        # Integration state
        self._session_contexts: Dict[str, str] = {}  # session_id -> context_id
        self._provider_contexts: Dict[str, Dict[str, str]] = {}  # provider -> {session_id: context_id}
        
        logger.info(
            "Context integration manager initialized",
            provider_integration=provider_manager is not None,
            bridge_integration=bridge_session_manager is not None,
            security_integration=security_manager is not None,
        )
    
    async def initialize(self) -> None:
        """Initialize integration components."""
        try:
            # Validate configuration
            validate_context_config()
            
            # Create necessary directories
            create_context_directories()
            
            # Initialize context manager
            await self.context_manager.initialize()
            
            # Setup integrations
            if self.provider_manager:
                await self._setup_provider_integration()
            
            if self.bridge_session_manager:
                await self._setup_bridge_integration()
            
            if self.security_manager:
                await self._setup_security_integration()
            
            logger.info("Context integration system initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize context integration: {e}")
            raise
    
    async def _setup_provider_integration(self) -> None:
        """Setup integration with AI providers."""
        try:
            # Register context management hooks with provider manager
            if hasattr(self.provider_manager, 'add_request_hook'):
                self.provider_manager.add_request_hook(self._on_provider_request)
            
            if hasattr(self.provider_manager, 'add_response_hook'):
                self.provider_manager.add_response_hook(self._on_provider_response)
            
            logger.info("Provider integration configured")
            
        except Exception as e:
            logger.error(f"Failed to setup provider integration: {e}")
            raise
    
    async def _setup_bridge_integration(self) -> None:
        """Setup integration with MCP Bridge."""
        try:
            # Register session lifecycle hooks
            if hasattr(self.bridge_session_manager, 'add_session_hook'):
                self.bridge_session_manager.add_session_hook('session_created', self._on_session_created)
                self.bridge_session_manager.add_session_hook('session_destroyed', self._on_session_destroyed)
                self.bridge_session_manager.add_session_hook('message_received', self._on_bridge_message)
            
            logger.info("Bridge integration configured")
            
        except Exception as e:
            logger.error(f"Failed to setup bridge integration: {e}")
            raise
    
    async def _setup_security_integration(self) -> None:
        """Setup integration with security system."""
        try:
            # Configure context manager to use security manager
            self.context_manager.security_manager = self.security_manager
            
            logger.info("Security integration configured")
            
        except Exception as e:
            logger.error(f"Failed to setup security integration: {e}")
            raise
    
    async def create_conversation_context(
        self,
        session_id: str,
        user_id: Optional[str] = None,
        title: str = "New Conversation",
        provider_type: Optional[ProviderType] = None,
        model_parameters: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create a new conversation context for a session."""
        try:
            # Create conversation context
            context = ConversationContext(
                title=title,
                description=f"Conversation context for session {session_id}",
                participants=[user_id] if user_id else [],
                primary_provider=provider_type.value if provider_type else self.config.default_provider,
                model_parameters=model_parameters or {},
                metadata={
                    "session_id": session_id,
                    "created_via": "integration_manager",
                },
            )
            
            # Store context
            context_id = await self.context_manager.create_context(
                context, user_id=user_id
            )
            
            # Map session to context
            self._session_contexts[session_id] = context_id
            
            # Map provider context if specified
            if provider_type:
                if provider_type.value not in self._provider_contexts:
                    self._provider_contexts[provider_type.value] = {}
                self._provider_contexts[provider_type.value][session_id] = context_id
            
            logger.info(
                "Conversation context created for session",
                session_id=session_id,
                context_id=context_id,
                user_id=user_id,
            )
            
            return context_id
            
        except Exception as e:
            logger.error(f"Failed to create conversation context: {e}")
            raise
    
    async def create_collaborative_context(
        self,
        room_id: str,
        title: str = "Collaborative Session",
        participants: List[str] = None,
        creator_id: Optional[str] = None,
    ) -> str:
        """Create a collaborative context for multi-user sessions."""
        try:
            context = CollaborativeContext(
                title=title,
                description=f"Collaborative context for room {room_id}",
                room_id=room_id,
                active_participants=participants or [],
                metadata={
                    "room_id": room_id,
                    "created_via": "integration_manager",
                },
            )
            
            context_id = await self.context_manager.create_context(
                context, user_id=creator_id
            )
            
            logger.info(
                "Collaborative context created",
                room_id=room_id,
                context_id=context_id,
                participants=len(participants) if participants else 0,
            )
            
            return context_id
            
        except Exception as e:
            logger.error(f"Failed to create collaborative context: {e}")
            raise
    
    async def add_message_to_conversation(
        self,
        session_id: str,
        message: Dict[str, Any],
        user_id: Optional[str] = None,
        sync_with_provider: bool = True,
    ) -> bool:
        """Add a message to a conversation context."""
        try:
            # Get context for session
            context_id = self._session_contexts.get(session_id)
            if not context_id:
                # Create context if it doesn't exist
                context_id = await self.create_conversation_context(
                    session_id, user_id=user_id
                )
            
            # Get current context
            context = await self.context_manager.get_context(context_id, user_id=user_id)
            if not isinstance(context, ConversationContext):
                raise ValueError(f"Context {context_id} is not a conversation context")
            
            # Add message
            context.add_message(message)
            
            # Update context
            success = await self.context_manager.update_context(
                context_id,
                {
                    "messages": context.messages,
                    "current_turn": context.current_turn,
                    "last_message_at": context.last_message_at,
                },
                user_id=user_id,
                sync=True,
            )
            
            # Sync with provider if requested
            if sync_with_provider and self.provider_manager and context.primary_provider:
                try:
                    provider_type = ProviderType(context.primary_provider)
                    await self._sync_context_with_provider(context_id, provider_type)
                except Exception as e:
                    logger.warning(f"Failed to sync with provider: {e}")
            
            if success:
                logger.debug(
                    "Message added to conversation",
                    session_id=session_id,
                    context_id=context_id,
                    message_role=message.get("role", "unknown"),
                )
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to add message to conversation: {e}")
            raise
    
    async def get_conversation_history(
        self,
        session_id: str,
        user_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Get conversation history for a session."""
        try:
            context_id = self._session_contexts.get(session_id)
            if not context_id:
                return []
            
            context = await self.context_manager.get_context(context_id, user_id=user_id)
            if not isinstance(context, ConversationContext):
                return []
            
            messages = context.messages
            if limit:
                messages = messages[-limit:]
            
            return messages
            
        except Exception as e:
            logger.error(f"Failed to get conversation history: {e}")
            return []
    
    async def translate_context_for_provider(
        self,
        context_id: str,
        provider_type: ProviderType,
        user_id: Optional[str] = None,
    ) -> Any:
        """Translate context to provider-specific format."""
        try:
            if not self.config.enable_translation:
                raise ValueError("Context translation is disabled")
            
            return await self.context_manager.translate_context(
                context_id, provider_type, user_id=user_id
            )
            
        except Exception as e:
            logger.error(f"Failed to translate context for provider: {e}")
            raise
    
    async def migrate_context_between_providers(
        self,
        context_id: str,
        source_provider: ProviderType,
        target_provider: ProviderType,
        user_id: Optional[str] = None,
    ) -> bool:
        """Migrate context between providers."""
        try:
            if not self.config.enable_cross_provider_sync:
                raise ValueError("Cross-provider sync is disabled")
            
            await self.context_manager.migrate_context(
                context_id,
                source_provider,
                target_provider,
                user_id=user_id,
            )
            
            logger.info(
                "Context migrated between providers",
                context_id=context_id,
                source=source_provider.value,
                target=target_provider.value,
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to migrate context: {e}")
            return False
    
    async def join_collaborative_session(
        self,
        context_id: str,
        user_id: str,
        room_id: Optional[str] = None,
    ) -> bool:
        """Join a collaborative context session."""
        try:
            return await self.context_manager.join_collaborative_session(
                context_id, user_id, room_id=room_id
            )
            
        except Exception as e:
            logger.error(f"Failed to join collaborative session: {e}")
            return False
    
    async def leave_collaborative_session(
        self,
        context_id: str,
        user_id: str,
        room_id: Optional[str] = None,
    ) -> bool:
        """Leave a collaborative context session."""
        try:
            return await self.context_manager.leave_collaborative_session(
                context_id, user_id, room_id=room_id
            )
            
        except Exception as e:
            logger.error(f"Failed to leave collaborative session: {e}")
            return False
    
    async def _sync_context_with_provider(
        self,
        context_id: str,
        provider_type: ProviderType,
    ) -> None:
        """Sync context with specific AI provider."""
        try:
            # Translate context for provider
            provider_context = await self.context_manager.translate_context(
                context_id, provider_type
            )
            
            # Update provider-specific context storage
            context = await self.context_manager.get_context(context_id)
            if context:
                await self.context_manager.update_context(
                    context_id,
                    {
                        "provider_contexts": {
                            **context.provider_contexts,
                            provider_type.value: provider_context,
                        }
                    },
                    sync=False,  # Avoid recursion
                )
            
        except Exception as e:
            logger.warning(f"Provider sync failed: {e}")
    
    async def _on_provider_request(self, provider_type: ProviderType, request_data: Dict[str, Any]) -> None:
        """Handle AI provider request."""
        try:
            session_id = request_data.get("session_id")
            if not session_id:
                return
            
            # Update context with request information
            context_id = self._session_contexts.get(session_id)
            if context_id:
                await self.context_manager.update_context(
                    context_id,
                    {
                        "metadata": {
                            "last_provider_request": {
                                "provider": provider_type.value,
                                "timestamp": request_data.get("timestamp"),
                                "model": request_data.get("model"),
                            }
                        }
                    },
                    sync=False,
                )
            
        except Exception as e:
            logger.warning(f"Provider request hook failed: {e}")
    
    async def _on_provider_response(self, provider_type: ProviderType, response_data: Dict[str, Any]) -> None:
        """Handle AI provider response."""
        try:
            session_id = response_data.get("session_id")
            if not session_id:
                return
            
            # Add response to conversation if available
            if "content" in response_data:
                message = {
                    "role": "assistant",
                    "content": response_data["content"],
                    "metadata": {
                        "provider": provider_type.value,
                        "model": response_data.get("model"),
                        "usage": response_data.get("usage"),
                    },
                }
                
                await self.add_message_to_conversation(
                    session_id, message, sync_with_provider=False
                )
            
        except Exception as e:
            logger.warning(f"Provider response hook failed: {e}")
    
    async def _on_session_created(self, session: MCPSession) -> None:
        """Handle MCP session creation."""
        try:
            if self.config.integrate_with_bridge:
                # Create session context
                session_context = SessionContext(
                    session_id=session.session_id,
                    user_id=session.client_id,
                    metadata={
                        "mcp_session_id": session.session_id,
                        "created_via": "bridge_integration",
                        "transport": session.transport.value if session.transport else None,
                    },
                )
                
                context_id = await self.context_manager.create_context(
                    session_context, user_id=session.client_id
                )
                
                logger.info(
                    "Session context created for MCP session",
                    mcp_session_id=session.session_id,
                    context_id=context_id,
                )
            
        except Exception as e:
            logger.warning(f"Session creation hook failed: {e}")
    
    async def _on_session_destroyed(self, session: MCPSession) -> None:
        """Handle MCP session destruction."""
        try:
            # Clean up session mappings
            if session.session_id in self._session_contexts:
                context_id = self._session_contexts[session.session_id]
                
                # Archive the context instead of deleting
                await self.context_manager.update_context(
                    context_id,
                    {"state": "archived"},
                    user_id=session.client_id,
                )
                
                del self._session_contexts[session.session_id]
                
                logger.info(
                    "Session context archived for destroyed MCP session",
                    mcp_session_id=session.session_id,
                    context_id=context_id,
                )
            
        except Exception as e:
            logger.warning(f"Session destruction hook failed: {e}")
    
    async def _on_bridge_message(self, session: MCPSession, message: BridgeMessage) -> None:
        """Handle bridge message."""
        try:
            if message.type == "request" and hasattr(message.data, 'method'):
                # Track request in context
                context_id = self._session_contexts.get(session.session_id)
                if context_id:
                    await self.context_manager.update_context(
                        context_id,
                        {
                            "metadata": {
                                "last_mcp_request": {
                                    "method": message.data.method,
                                    "timestamp": message.timestamp.isoformat(),
                                }
                            }
                        },
                        user_id=session.client_id,
                        sync=False,
                    )
            
        except Exception as e:
            logger.warning(f"Bridge message hook failed: {e}")
    
    def get_integration_stats(self) -> Dict[str, Any]:
        """Get integration statistics."""
        return {
            "session_contexts": len(self._session_contexts),
            "provider_contexts": {
                provider: len(contexts)
                for provider, contexts in self._provider_contexts.items()
            },
            "configuration": {
                "real_time_sync": self.config.enable_real_time_sync,
                "compression": self.config.enable_compression,
                "validation": self.config.enable_validation,
                "versioning": self.config.enable_versioning,
                "translation": self.config.enable_translation,
            },
            "integrations": {
                "provider_manager": self.provider_manager is not None,
                "bridge_session_manager": self.bridge_session_manager is not None,
                "security_manager": self.security_manager is not None,
            },
        }
    
    async def cleanup(self) -> None:
        """Clean up integration resources."""
        try:
            # Clean up mappings
            self._session_contexts.clear()
            self._provider_contexts.clear()
            
            # Clean up context manager
            await self.context_manager.cleanup()
            
            logger.info("Context integration manager cleaned up")
            
        except Exception as e:
            logger.error(f"Integration cleanup failed: {e}")


# Global integration manager instance (initialized when needed)
_integration_manager: Optional[ContextIntegrationManager] = None


async def get_integration_manager(
    provider_manager: Optional[ProviderManager] = None,
    bridge_session_manager: Optional[BridgeSessionManager] = None,
    security_manager: Optional[EnhancedSecurityManager] = None,
) -> ContextIntegrationManager:
    """Get or create the global context integration manager."""
    global _integration_manager
    
    if _integration_manager is None:
        config = get_context_config()
        
        # Create context manager
        context_manager = ContextManager(
            database_url=config.database_url,
            redis_url=config.redis_url,
            security_manager=security_manager,
            enable_real_time_sync=config.enable_real_time_sync,
            enable_compression=config.enable_compression,
            enable_validation=config.enable_validation,
            enable_versioning=config.enable_versioning,
        )
        
        # Create integration manager
        _integration_manager = ContextIntegrationManager(
            context_manager=context_manager,
            provider_manager=provider_manager,
            bridge_session_manager=bridge_session_manager,
            security_manager=security_manager,
        )
        
        await _integration_manager.initialize()
    
    return _integration_manager


async def cleanup_integration_manager() -> None:
    """Clean up the global integration manager."""
    global _integration_manager
    
    if _integration_manager:
        await _integration_manager.cleanup()
        _integration_manager = None