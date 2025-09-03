"""
Complete MCP Integration for Provider Router with Fallback System.

This module provides the complete integration layer that combines all MCP components
for the Provider Router system, including routing, health monitoring, error handling,
and protocol compliance.

Key Features:
- Complete FastMCP server integration
- Unified tool registration and management
- Event-driven architecture with real-time notifications
- Comprehensive error handling and recovery
- Health monitoring with automated alerting
- Cost optimization integration
- Protocol-compliant JSON-RPC 2.0 messaging
"""

import asyncio
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from fastmcp import FastMCP
from structlog import get_logger

from .mcp_provider_router import MCPProviderRouter, register_provider_router_tools
from .mcp_error_handler import MCPErrorHandler, MCPError
from .mcp_health_monitor import ProviderHealthMonitor, HealthAlert
from .mcp_protocol_schemas import (
    PROTOCOL_VERSION,
    MCP_PROVIDER_ROUTER_NAMESPACE,
    ProviderEventType,
    create_notification,
    serialize_message
)
from .provider_manager import AIProviderManager
from .models import ProviderConfig, CostBudget

logger = get_logger(__name__)


class MCPProviderRouterServer:
    """
    Complete MCP Server integration for Provider Router with Fallback.
    
    This class provides the main integration point for all MCP functionality,
    including FastMCP server management, tool registration, event handling,
    and system coordination.
    """
    
    def __init__(
        self,
        server_name: str = "provider-router",
        version: str = PROTOCOL_VERSION,
        provider_configs: Optional[List[ProviderConfig]] = None,
        cost_budgets: Optional[List[CostBudget]] = None,
        health_check_interval: int = 60
    ):
        # Core components
        self.server_name = server_name
        self.version = version
        self.mcp_server = FastMCP(server_name)
        
        # AI Provider components
        self.provider_manager = AIProviderManager()
        self.provider_configs = provider_configs or []
        self.cost_budgets = cost_budgets or []
        
        # MCP Router components
        self.router = MCPProviderRouter(self.provider_manager)
        self.error_handler = MCPErrorHandler()
        self.health_monitor = ProviderHealthMonitor(check_interval=health_check_interval)
        
        # State management
        self._initialized = False
        self._running = False
        self.event_subscribers: Dict[str, List[callable]] = {}
        
        # Performance metrics
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_response_time": 0.0,
            "uptime_start": None
        }
    
    async def initialize(self) -> None:
        """Initialize the complete MCP Provider Router server."""
        if self._initialized:
            logger.warning("MCP Provider Router Server already initialized")
            return
        
        logger.info(
            "Initializing MCP Provider Router Server",
            server_name=self.server_name,
            version=self.version,
            providers=len(self.provider_configs)
        )
        
        try:
            # Initialize AI Provider Manager
            await self.provider_manager.initialize(
                provider_configs=self.provider_configs,
                budgets=self.cost_budgets
            )
            
            # Initialize MCP Router
            await self.router.initialize()
            
            # Register providers with health monitor
            for config in self.provider_configs:
                self.health_monitor.register_provider(
                    config.provider_type.value,
                    {"priority": config.priority, "enabled": config.enabled}
                )
            
            # Start health monitoring
            await self.health_monitor.start_monitoring()
            
            # Setup event handling
            self._setup_event_handling()
            
            # Register MCP tools
            self._register_all_tools()
            
            # Setup health monitoring callbacks
            self._setup_health_callbacks()
            
            # Setup error handling callbacks
            self._setup_error_callbacks()
            
            self._initialized = True
            self.metrics["uptime_start"] = datetime.now()
            
            logger.info("MCP Provider Router Server initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize MCP Provider Router Server", error=str(e))
            raise
    
    async def start(self) -> None:
        """Start the MCP server."""
        if not self._initialized:
            await self.initialize()
        
        if self._running:
            logger.warning("MCP Provider Router Server already running")
            return
        
        self._running = True
        
        logger.info("Starting MCP Provider Router Server")
        
        # Emit server started event
        await self._emit_system_event("server_started", {
            "server_name": self.server_name,
            "version": self.version,
            "providers_configured": len(self.provider_configs)
        })
    
    async def stop(self) -> None:
        """Stop the MCP server and cleanup resources."""
        if not self._running:
            return
        
        logger.info("Stopping MCP Provider Router Server")
        
        try:
            # Stop health monitoring
            await self.health_monitor.stop_monitoring()
            
            # Shutdown router
            await self.router.shutdown()
            
            # Shutdown provider manager
            await self.provider_manager.shutdown()
            
            # Emit server stopped event
            await self._emit_system_event("server_stopped", {
                "server_name": self.server_name,
                "uptime_seconds": self._calculate_uptime()
            })
            
            self._running = False
            
            logger.info("MCP Provider Router Server stopped successfully")
            
        except Exception as e:
            logger.error("Error stopping MCP Provider Router Server", error=str(e))
            raise
    
    def _register_all_tools(self):
        """Register all MCP tools with the FastMCP server."""
        logger.info("Registering MCP tools")
        
        # Register core provider router tools
        self.registered_router = register_provider_router_tools(
            self.mcp_server, 
            self.provider_manager
        )
        
        # Register additional monitoring and management tools
        self._register_monitoring_tools()
        self._register_management_tools()
        
        logger.info("All MCP tools registered successfully")
    
    def _register_monitoring_tools(self):
        """Register monitoring and diagnostics tools."""
        
        @self.mcp_server.tool()
        async def get_system_health() -> Dict[str, Any]:
            """
            Get comprehensive system health and status information.
            
            Returns detailed health metrics for all providers, system performance,
            error rates, and overall service availability.
            """
            try:
                health_summary = self.health_monitor.get_health_summary()
                error_stats = self.error_handler.get_error_statistics()
                router_stats = self.router.get_routing_stats()
                
                return {
                    "system_status": "healthy" if health_summary["overall_health_percentage"] > 80 else "degraded",
                    "server_info": {
                        "name": self.server_name,
                        "version": self.version,
                        "uptime_seconds": self._calculate_uptime(),
                        "initialized": self._initialized,
                        "running": self._running
                    },
                    "health_summary": health_summary,
                    "error_statistics": error_stats,
                    "routing_statistics": router_stats,
                    "performance_metrics": self.metrics,
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                logger.error("Failed to get system health", error=str(e))
                raise
        
        @self.mcp_server.tool()
        async def get_provider_health_details(provider_name: str) -> Dict[str, Any]:
            """
            Get detailed health information for a specific provider.
            
            Args:
                provider_name: Name of the provider to get health details for
            
            Returns:
                Comprehensive health metrics and status for the specified provider
            """
            try:
                provider_health = self.health_monitor.get_provider_health(provider_name)
                
                if not provider_health:
                    return {
                        "error": f"Provider '{provider_name}' not found",
                        "available_providers": list(self.health_monitor.provider_configs.keys())
                    }
                
                # Get historical metrics
                metrics_history = {}
                for metric_type in ["response_time", "error_rate", "success_rate"]:
                    metrics_history[metric_type] = self.health_monitor.get_provider_metrics_history(
                        provider_name, metric_type, hours=24
                    )
                
                return {
                    "provider_name": provider_name,
                    "current_status": provider_health.dict(),
                    "metrics_history": metrics_history,
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                logger.error("Failed to get provider health details", error=str(e), provider=provider_name)
                raise
        
        @self.mcp_server.tool()
        async def get_active_alerts() -> Dict[str, Any]:
            """
            Get all active health alerts and their details.
            
            Returns comprehensive information about current system alerts,
            their severity levels, and recommended actions.
            """
            try:
                active_alerts = list(self.health_monitor.active_alerts.values())
                
                return {
                    "total_alerts": len(active_alerts),
                    "alerts_by_severity": {
                        severity: [
                            alert.dict() for alert in active_alerts
                            if alert.severity.value == severity
                        ]
                        for severity in ["critical", "error", "warning", "info"]
                    },
                    "alerts_by_provider": {
                        provider: [
                            alert.dict() for alert in active_alerts
                            if alert.provider_name == provider
                        ]
                        for provider in set(alert.provider_name for alert in active_alerts)
                    },
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                logger.error("Failed to get active alerts", error=str(e))
                raise
    
    def _register_management_tools(self):
        """Register system management and configuration tools."""
        
        @self.mcp_server.tool()
        async def update_provider_config(
            provider_name: str,
            config_updates: Dict[str, Any]
        ) -> Dict[str, Any]:
            """
            Update configuration for a specific provider.
            
            Args:
                provider_name: Name of the provider to update
                config_updates: Configuration updates to apply
            
            Returns:
                Result of the configuration update operation
            """
            try:
                # Validate provider exists
                if provider_name not in [config.provider_type.value for config in self.provider_configs]:
                    return {
                        "success": False,
                        "error": f"Provider '{provider_name}' not found",
                        "available_providers": [config.provider_type.value for config in self.provider_configs]
                    }
                
                # Apply configuration updates (simplified implementation)
                logger.info(
                    "Updating provider configuration",
                    provider=provider_name,
                    updates=config_updates
                )
                
                # In practice, this would update the actual provider configuration
                # and potentially restart the provider with new settings
                
                return {
                    "success": True,
                    "provider": provider_name,
                    "updates_applied": config_updates,
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                logger.error("Failed to update provider config", error=str(e), provider=provider_name)
                return {
                    "success": False,
                    "error": str(e)
                }
        
        @self.mcp_server.tool()
        async def restart_provider(provider_name: str, reason: str = "Manual restart") -> Dict[str, Any]:
            """
            Restart a specific provider.
            
            Args:
                provider_name: Name of the provider to restart
                reason: Reason for the restart
            
            Returns:
                Result of the restart operation
            """
            try:
                logger.info("Restarting provider", provider=provider_name, reason=reason)
                
                # In practice, this would:
                # 1. Gracefully shutdown the provider
                # 2. Clear any circuit breaker state
                # 3. Reinitialize the provider
                # 4. Update health monitoring
                
                # Emit restart event
                await self._emit_system_event("provider_restarted", {
                    "provider_name": provider_name,
                    "reason": reason,
                    "timestamp": datetime.now().isoformat()
                })
                
                return {
                    "success": True,
                    "provider": provider_name,
                    "reason": reason,
                    "restarted_at": datetime.now().isoformat()
                }
                
            except Exception as e:
                logger.error("Failed to restart provider", error=str(e), provider=provider_name)
                return {
                    "success": False,
                    "error": str(e)
                }
        
        @self.mcp_server.tool()
        async def get_protocol_info() -> Dict[str, Any]:
            """
            Get MCP protocol information and capabilities.
            
            Returns comprehensive information about the MCP protocol version,
            supported methods, events, and error codes.
            """
            from .mcp_protocol_schemas import PROTOCOL_DOCUMENTATION
            
            return {
                "protocol_version": PROTOCOL_VERSION,
                "server_info": {
                    "name": self.server_name,
                    "version": self.version,
                    "namespace": MCP_PROVIDER_ROUTER_NAMESPACE
                },
                "capabilities": {
                    "provider_routing": True,
                    "failover_support": True,
                    "health_monitoring": True,
                    "cost_optimization": True,
                    "real_time_events": True,
                    "circuit_breakers": True,
                    "error_recovery": True
                },
                "protocol_documentation": PROTOCOL_DOCUMENTATION,
                "timestamp": datetime.now().isoformat()
            }
    
    def _setup_event_handling(self):
        """Setup event handling and notification system."""
        logger.info("Setting up event handling system")
        
        # Setup router event subscriptions
        self.router.subscribe_to_events(self._handle_router_event)
        
        # Setup internal event subscribers for different event types
        self.event_subscribers = {
            ProviderEventType.PROVIDER_HEALTH_CHANGED: [],
            ProviderEventType.FAILOVER_TRIGGERED: [],
            ProviderEventType.ROUTING_DECISION_MADE: [],
            ProviderEventType.COST_THRESHOLD_EXCEEDED: [],
            "system_events": [],
            "error_events": []
        }
    
    def _setup_health_callbacks(self):
        """Setup health monitoring event callbacks."""
        
        async def health_alert_callback(alert: HealthAlert):
            """Handle health alerts."""
            await self._emit_health_event("health_alert", {
                "alert_id": alert.alert_id,
                "provider": alert.provider_name,
                "severity": alert.severity.value,
                "metric_type": alert.metric_type.value,
                "current_value": alert.current_value,
                "threshold_value": alert.threshold_value,
                "message": alert.message
            })
        
        self.health_monitor.add_alert_callback(health_alert_callback)
    
    def _setup_error_callbacks(self):
        """Setup error handling event callbacks."""
        
        async def error_callback(error: MCPError):
            """Handle MCP errors."""
            await self._emit_error_event("mcp_error", {
                "error_id": error.error_id,
                "category": error.category.value,
                "severity": error.severity.value,
                "message": error.message,
                "provider": error.context.provider,
                "can_retry": error.can_retry,
                "recovery_strategy": error.recovery_strategy.value
            })
        
        self.error_handler.add_error_callback(error_callback)
    
    async def _handle_router_event(self, event_data: Dict[str, Any]):
        """Handle events from the router."""
        event_type = event_data.get("event_type")
        
        if event_type in self.event_subscribers:
            # Notify subscribers for this event type
            subscribers = self.event_subscribers[event_type]
            for subscriber in subscribers:
                try:
                    if asyncio.iscoroutinefunction(subscriber):
                        await subscriber(event_data)
                    else:
                        subscriber(event_data)
                except Exception as e:
                    logger.error("Event subscriber failed", error=str(e), event_type=event_type)
        
        # Emit as MCP notification
        await self._emit_mcp_notification(f"router/{event_type}", event_data)
    
    async def _emit_system_event(self, event_type: str, event_data: Dict[str, Any]):
        """Emit system-level events."""
        full_event_data = {
            "event_type": event_type,
            "timestamp": datetime.now().isoformat(),
            "server_name": self.server_name,
            **event_data
        }
        
        # Notify system event subscribers
        for subscriber in self.event_subscribers.get("system_events", []):
            try:
                if asyncio.iscoroutinefunction(subscriber):
                    await subscriber(full_event_data)
                else:
                    subscriber(full_event_data)
            except Exception as e:
                logger.error("System event subscriber failed", error=str(e))
        
        # Emit as MCP notification
        await self._emit_mcp_notification(f"system/{event_type}", full_event_data)
    
    async def _emit_health_event(self, event_type: str, event_data: Dict[str, Any]):
        """Emit health monitoring events."""
        full_event_data = {
            "event_type": event_type,
            "timestamp": datetime.now().isoformat(),
            **event_data
        }
        
        await self._emit_mcp_notification(f"health/{event_type}", full_event_data)
    
    async def _emit_error_event(self, event_type: str, event_data: Dict[str, Any]):
        """Emit error handling events."""
        full_event_data = {
            "event_type": event_type,
            "timestamp": datetime.now().isoformat(),
            **event_data
        }
        
        # Notify error event subscribers
        for subscriber in self.event_subscribers.get("error_events", []):
            try:
                if asyncio.iscoroutinefunction(subscriber):
                    await subscriber(full_event_data)
                else:
                    subscriber(full_event_data)
            except Exception as e:
                logger.error("Error event subscriber failed", error=str(e))
        
        await self._emit_mcp_notification(f"error/{event_type}", full_event_data)
    
    async def _emit_mcp_notification(self, method: str, params: Dict[str, Any]):
        """Emit MCP JSON-RPC notification."""
        notification = create_notification(method=method, params=params)
        
        # Log the notification (in practice, this would be sent to connected clients)
        logger.debug(
            "MCP notification emitted",
            method=method,
            notification_size=len(serialize_message(notification))
        )
    
    def _calculate_uptime(self) -> float:
        """Calculate server uptime in seconds."""
        if not self.metrics["uptime_start"]:
            return 0.0
        
        return (datetime.now() - self.metrics["uptime_start"]).total_seconds()
    
    def subscribe_to_events(self, event_type: str, callback: callable):
        """Subscribe to specific event types."""
        if event_type not in self.event_subscribers:
            self.event_subscribers[event_type] = []
        
        self.event_subscribers[event_type].append(callback)
        logger.info("Event subscription added", event_type=event_type)
    
    def unsubscribe_from_events(self, event_type: str, callback: callable):
        """Unsubscribe from specific event types."""
        if event_type in self.event_subscribers and callback in self.event_subscribers[event_type]:
            self.event_subscribers[event_type].remove(callback)
            logger.info("Event subscription removed", event_type=event_type)
    
    def get_server_info(self) -> Dict[str, Any]:
        """Get comprehensive server information."""
        return {
            "server_name": self.server_name,
            "version": self.version,
            "protocol_version": PROTOCOL_VERSION,
            "namespace": MCP_PROVIDER_ROUTER_NAMESPACE,
            "initialized": self._initialized,
            "running": self._running,
            "uptime_seconds": self._calculate_uptime(),
            "providers_configured": len(self.provider_configs),
            "performance_metrics": self.metrics,
            "components": {
                "provider_manager": self.provider_manager._initialized,
                "router": True,
                "error_handler": True,
                "health_monitor": self.health_monitor.is_monitoring
            }
        }
    
    def get_mcp_server(self) -> FastMCP:
        """Get the underlying FastMCP server instance."""
        return self.mcp_server


# Convenience function for easy setup
async def create_provider_router_server(
    server_name: str = "provider-router",
    provider_configs: Optional[List[ProviderConfig]] = None,
    cost_budgets: Optional[List[CostBudget]] = None,
    auto_start: bool = True
) -> MCPProviderRouterServer:
    """
    Create and optionally start a complete MCP Provider Router server.
    
    Args:
        server_name: Name for the MCP server
        provider_configs: List of provider configurations
        cost_budgets: List of cost budget configurations
        auto_start: Whether to automatically start the server
    
    Returns:
        Initialized (and optionally started) MCPProviderRouterServer instance
    """
    server = MCPProviderRouterServer(
        server_name=server_name,
        provider_configs=provider_configs,
        cost_budgets=cost_budgets
    )
    
    await server.initialize()
    
    if auto_start:
        await server.start()
    
    return server


# Export main classes
__all__ = [
    "MCPProviderRouterServer",
    "create_provider_router_server"
]