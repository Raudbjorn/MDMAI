"""Comprehensive Usage Tracking and Cost Management System Integration."""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from structlog import get_logger

from .models import ProviderType, AIRequest, AIResponse, UsageRecord
from .enhanced_token_estimator import EnhancedTokenEstimator
from .pricing_engine import PricingEngine
from .user_usage_tracker import UserUsageTracker, UserProfile, UserSpendingLimits
from .budget_enforcer import BudgetEnforcer, EnforcementResult, EnforcementAction
from .analytics_dashboard import AnalyticsDashboard
from .cost_optimization_engine import CostOptimizationEngine, OptimizationRecommendation
from .metrics_collector import MetricsCollector
from .alert_system import AlertSystem, AlertSeverity, AlertType
from ..core.database import get_db_manager

logger = get_logger(__name__)


@dataclass
class SystemConfiguration:
    """Comprehensive system configuration."""
    
    # Storage configuration
    storage_base_path: str = "./data"
    use_chromadb: bool = True
    
    # Token estimation settings
    enable_token_estimation_cache: bool = True
    token_cache_size: int = 10000
    
    # Pricing settings
    pricing_update_interval: int = 300  # 5 minutes
    enable_dynamic_pricing: bool = True
    
    # Budget enforcement settings
    enable_budget_enforcement: bool = True
    default_user_tier: str = "free"
    grace_period_enabled: bool = True
    
    # Analytics settings
    enable_analytics: bool = True
    dashboard_refresh_interval: int = 60  # seconds
    metrics_aggregation_interval: int = 300  # 5 minutes
    
    # Cost optimization settings
    enable_cost_optimization: bool = True
    optimization_analysis_interval: int = 3600  # 1 hour
    min_optimization_savings: float = 5.0  # USD
    
    # Alerting settings
    enable_alerting: bool = True
    alert_evaluation_interval: int = 30  # seconds
    default_notification_channels: List[str] = field(default_factory=lambda: ["email", "log_only"])
    
    # Retention settings
    raw_data_retention_days: int = 7
    aggregated_data_retention_days: int = 365
    alert_history_retention_days: int = 90
    
    # Performance settings
    async_processing: bool = True
    batch_processing_size: int = 100
    max_concurrent_operations: int = 10


class ComprehensiveUsageSystem:
    """
    Comprehensive Usage Tracking and Cost Management System.
    
    This is the main integration class that orchestrates all components:
    - Token counting and estimation
    - Real-time cost calculation
    - Per-user usage tracking
    - Budget enforcement with graceful degradation
    - Analytics dashboard and reporting
    - Cost optimization recommendations
    - Metrics collection and retention
    - Alerting and notifications
    """
    
    def __init__(self, config: Optional[SystemConfiguration] = None):
        self.config = config or SystemConfiguration()
        
        # Initialize storage paths
        self.storage_path = Path(self.config.storage_base_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize core components
        self.token_estimator = EnhancedTokenEstimator()
        self.pricing_engine = PricingEngine(
            str(self.storage_path / "pricing")
        )
        self.usage_tracker = UserUsageTracker(
            str(self.storage_path / "users"),
            self.config.use_chromadb
        )
        
        # Initialize advanced components
        self.budget_enforcer = BudgetEnforcer(
            self.usage_tracker,
            self.pricing_engine
        )
        
        if self.config.enable_analytics:
            self.analytics_dashboard = AnalyticsDashboard(
                self.usage_tracker,
                self.pricing_engine,
                str(self.storage_path / "analytics"),
                self.config.use_chromadb
            )
        else:
            self.analytics_dashboard = None
        
        if self.config.enable_cost_optimization:
            self.cost_optimizer = CostOptimizationEngine(
                self.usage_tracker,
                self.pricing_engine,
                self.analytics_dashboard,
                str(self.storage_path / "optimization")
            )
        else:
            self.cost_optimizer = None
        
        self.metrics_collector = MetricsCollector(
            self.usage_tracker,
            str(self.storage_path / "metrics"),
            self.config.use_chromadb
        )
        
        if self.config.enable_alerting:
            self.alert_system = AlertSystem(
                self.usage_tracker,
                self.metrics_collector,
                self.budget_enforcer,
                str(self.storage_path / "alerts")
            )
        else:
            self.alert_system = None
        
        # Background tasks
        self._background_tasks: List[asyncio.Task] = []
        self._is_running = False
        
        # System statistics
        self.system_stats = {
            "start_time": datetime.now(),
            "requests_processed": 0,
            "total_cost_tracked": 0.0,
            "users_managed": 0,
            "alerts_generated": 0,
            "optimizations_suggested": 0,
            "budget_enforcements": 0
        }
        
        logger.info("Comprehensive Usage System initialized", config=self.config)
    
    async def start(self) -> None:
        """Start the comprehensive usage system."""
        if self._is_running:
            logger.warning("System is already running")
            return
        
        try:
            # Start background processes
            if self.config.async_processing:
                await self._start_background_processes()
            
            self._is_running = True
            
            logger.info("Comprehensive Usage System started successfully")
            
        except Exception as e:
            logger.error("Failed to start system", error=str(e))
            raise
    
    async def _start_background_processes(self) -> None:
        """Start all background processing tasks."""
        # Start metrics collection
        if self.metrics_collector:
            metrics_task = asyncio.create_task(self._run_metrics_collection())
            self._background_tasks.append(metrics_task)
        
        # Start cost optimization analysis
        if self.cost_optimizer:
            optimization_task = asyncio.create_task(self._run_cost_optimization())
            self._background_tasks.append(optimization_task)
        
        # Start periodic cleanup
        cleanup_task = asyncio.create_task(self._run_periodic_cleanup())
        self._background_tasks.append(cleanup_task)
        
        logger.info("Background processes started", count=len(self._background_tasks))
    
    async def _run_metrics_collection(self) -> None:
        """Run periodic metrics collection."""
        while self._is_running:
            try:
                # Collect system-wide metrics
                await self._collect_system_metrics()
                
                await asyncio.sleep(self.config.metrics_aggregation_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in metrics collection", error=str(e))
                await asyncio.sleep(60)
    
    async def _run_cost_optimization(self) -> None:
        """Run periodic cost optimization analysis."""
        while self._is_running:
            try:
                if self.cost_optimizer:
                    recommendations = await self.cost_optimizer.generate_recommendations()
                    self.system_stats["optimizations_suggested"] += len(recommendations)
                
                await asyncio.sleep(self.config.optimization_analysis_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in cost optimization", error=str(e))
                await asyncio.sleep(300)
    
    async def _run_periodic_cleanup(self) -> None:
        """Run periodic system cleanup."""
        while self._is_running:
            try:
                await self._perform_system_cleanup()
                
                # Run cleanup daily
                await asyncio.sleep(86400)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in periodic cleanup", error=str(e))
                await asyncio.sleep(3600)
    
    async def process_ai_request(
        self,
        request: AIRequest,
        provider_type: ProviderType,
        user_id: str,
        check_budget: bool = True
    ) -> Tuple[bool, Optional[EnforcementResult], Dict[str, Any]]:
        """
        Process an AI request with comprehensive tracking and enforcement.
        
        Args:
            request: The AI request to process
            provider_type: Provider that will handle the request
            user_id: User making the request
            check_budget: Whether to enforce budget limits
            
        Returns:
            Tuple of (allowed, enforcement_result, metadata)
        """
        try:
            processing_start = datetime.now()
            
            # Estimate request cost
            estimated_cost, cost_breakdown = self.pricing_engine.estimate_request_cost(
                provider_type, request.model, request.messages, request.tools, request.max_tokens
            )
            
            enforcement_result = None
            allowed = True
            metadata = {
                "estimated_cost": estimated_cost,
                "cost_breakdown": cost_breakdown,
                "processing_time": 0.0
            }
            
            # Check budget enforcement if enabled
            if check_budget and self.config.enable_budget_enforcement:
                enforcement_result = await self.budget_enforcer.check_budget_enforcement(
                    user_id, request, estimated_cost, provider_type
                )
                
                allowed = enforcement_result.allowed
                
                if not allowed:
                    self.system_stats["budget_enforcements"] += 1
                    
                    # Log enforcement action
                    await self.metrics_collector.collect_metric(
                        "budget_enforcement",
                        1.0,
                        user_id=user_id,
                        provider_type=provider_type.value,
                        metadata={"action": enforcement_result.action.value}
                    )
                    
                    # Use alternative if suggested
                    if enforcement_result.modified_request:
                        request = enforcement_result.modified_request
                        provider_type = enforcement_result.suggested_provider or provider_type
                        allowed = True
            
            # Update metadata
            processing_time = (datetime.now() - processing_start).total_seconds() * 1000
            metadata["processing_time"] = processing_time
            
            # Collect processing metrics
            await self.metrics_collector.collect_metric(
                "request_processing_time",
                processing_time,
                user_id=user_id,
                provider_type=provider_type.value
            )
            
            self.system_stats["requests_processed"] += 1
            
            return allowed, enforcement_result, metadata
            
        except Exception as e:
            logger.error("Failed to process AI request", 
                        request_id=request.request_id, 
                        user_id=user_id, 
                        error=str(e))
            
            # Return safe default
            return False, None, {"error": str(e)}
    
    async def record_ai_response(
        self,
        request: AIRequest,
        response: Optional[AIResponse],
        provider_type: ProviderType,
        user_id: str,
        success: bool = True,
        error_message: Optional[str] = None
    ) -> None:
        """
        Record an AI response with comprehensive tracking.
        
        Args:
            request: The original AI request
            response: The AI response (None if failed)
            provider_type: Provider that handled the request
            user_id: User who made the request
            success: Whether the request was successful
            error_message: Error message if failed
        """
        try:
            # Calculate actual cost if response available
            actual_cost = 0.0
            if response and response.usage:
                input_tokens = response.usage.get("input_tokens", 0)
                output_tokens = response.usage.get("output_tokens", 0)
                actual_cost, _ = self.pricing_engine.calculate_request_cost(
                    provider_type, request.model, input_tokens, output_tokens, response.latency_ms
                )
            
            # Create usage record
            usage_record = UsageRecord(
                request_id=request.request_id,
                session_id=request.session_id,
                provider_type=provider_type,
                model=request.model,
                input_tokens=response.usage.get("input_tokens", 0) if response and response.usage else 0,
                output_tokens=response.usage.get("output_tokens", 0) if response and response.usage else 0,
                cost=actual_cost,
                latency_ms=response.latency_ms if response else 0.0,
                success=success,
                error_message=error_message,
                metadata={
                    "user_id": user_id,
                    "temperature": request.temperature,
                    "max_tokens": request.max_tokens,
                    "has_tools": bool(request.tools),
                    "streaming": request.stream,
                }
            )
            
            # Record usage
            await self.usage_tracker.record_user_usage(user_id, usage_record, request, response)
            
            # Collect detailed metrics
            await self._collect_response_metrics(usage_record, user_id, provider_type)
            
            # Update system statistics
            self.system_stats["total_cost_tracked"] += actual_cost
            
            logger.debug("AI response recorded",
                        request_id=request.request_id,
                        user_id=user_id,
                        cost=actual_cost,
                        success=success)
            
        except Exception as e:
            logger.error("Failed to record AI response",
                        request_id=request.request_id,
                        user_id=user_id,
                        error=str(e))
    
    async def _collect_response_metrics(
        self,
        usage_record: UsageRecord,
        user_id: str,
        provider_type: ProviderType
    ) -> None:
        """Collect detailed metrics from usage record."""
        # Basic metrics
        await self.metrics_collector.collect_metric(
            "requests_total", 1.0, user_id=user_id, provider_type=provider_type.value
        )
        
        if usage_record.success:
            await self.metrics_collector.collect_metric(
                "requests_successful", 1.0, user_id=user_id, provider_type=provider_type.value
            )
        else:
            await self.metrics_collector.collect_metric(
                "requests_failed", 1.0, user_id=user_id, provider_type=provider_type.value
            )
        
        # Token metrics
        total_tokens = usage_record.input_tokens + usage_record.output_tokens
        await self.metrics_collector.collect_metric(
            "tokens_total", float(total_tokens), user_id=user_id, provider_type=provider_type.value
        )
        await self.metrics_collector.collect_metric(
            "tokens_input", float(usage_record.input_tokens), user_id=user_id, provider_type=provider_type.value
        )
        await self.metrics_collector.collect_metric(
            "tokens_output", float(usage_record.output_tokens), user_id=user_id, provider_type=provider_type.value
        )
        
        # Cost metrics
        await self.metrics_collector.collect_metric(
            "cost_total", usage_record.cost, user_id=user_id, provider_type=provider_type.value
        )
        
        if total_tokens > 0:
            cost_per_token = usage_record.cost / total_tokens
            await self.metrics_collector.collect_metric(
                "cost_per_token", cost_per_token, user_id=user_id, provider_type=provider_type.value
            )
        
        # Performance metrics
        await self.metrics_collector.collect_metric(
            "latency_avg", usage_record.latency_ms, user_id=user_id, provider_type=provider_type.value
        )
        
        # Model-specific metrics
        await self.metrics_collector.collect_metric(
            "model_requests", 1.0, user_id=user_id, model=usage_record.model
        )
    
    async def _collect_system_metrics(self) -> None:
        """Collect system-wide metrics."""
        current_time = datetime.now()
        
        # System uptime
        uptime = (current_time - self.system_stats["start_time"]).total_seconds()
        await self.metrics_collector.collect_metric("system_uptime_seconds", uptime)
        
        # Active users
        active_users = len(self.usage_tracker.user_profiles)
        await self.metrics_collector.collect_metric("active_users", float(active_users))
        self.system_stats["users_managed"] = active_users
        
        # Alert metrics
        if self.alert_system:
            alert_stats = self.alert_system.get_alert_statistics()
            await self.metrics_collector.collect_metric("active_alerts", float(alert_stats["active_alerts"]["total"]))
            self.system_stats["alerts_generated"] = alert_stats["total_alerts"]
        
        # Budget enforcement metrics
        enforcement_stats = self.budget_enforcer.get_enforcement_stats()
        await self.metrics_collector.collect_metric("budget_enforcements", float(enforcement_stats["total_checks"]))
        
        # Memory usage (simplified)
        import psutil
        import os
        process = psutil.Process(os.getpid())
        memory_usage = process.memory_info().rss / 1024 / 1024  # MB
        await self.metrics_collector.collect_metric("system_memory_usage_mb", memory_usage)
    
    async def create_user(
        self,
        username: str,
        email: Optional[str] = None,
        user_tier: str = None,
        daily_limit: Optional[float] = None,
        monthly_limit: Optional[float] = None
    ) -> str:
        """
        Create a new user with default settings.
        
        Args:
            username: User's display name
            email: User's email address
            user_tier: User tier (free, premium, enterprise)
            daily_limit: Daily spending limit in USD
            monthly_limit: Monthly spending limit in USD
            
        Returns:
            User ID of the created user
        """
        try:
            # Create user profile
            user_profile = UserProfile(
                user_id="",  # Will be generated
                username=username,
                email=email,
                user_tier=user_tier or self.config.default_user_tier
            )
            
            # Create user
            created_profile = await self.usage_tracker.create_user_profile(user_profile)
            
            # Update spending limits if specified
            if daily_limit or monthly_limit:
                limits = UserSpendingLimits(
                    user_id=created_profile.user_id,
                    daily_limit=daily_limit,
                    monthly_limit=monthly_limit
                )
                self.usage_tracker.user_limits[created_profile.user_id] = limits
            
            logger.info("User created", user_id=created_profile.user_id, username=username)
            
            return created_profile.user_id
            
        except Exception as e:
            logger.error("Failed to create user", username=username, error=str(e))
            raise
    
    async def get_user_dashboard_data(
        self,
        user_id: str,
        time_range: str = "24h",
        dashboard_id: str = "overview"
    ) -> Dict[str, Any]:
        """
        Get comprehensive dashboard data for a user.
        
        Args:
            user_id: User ID
            time_range: Time range (1h, 24h, 7d, 30d, 90d, 1y)
            dashboard_id: Dashboard to generate
            
        Returns:
            Complete dashboard data
        """
        try:
            if not self.analytics_dashboard:
                raise ValueError("Analytics dashboard not enabled")
            
            # Get dashboard data
            dashboard_data = await self.analytics_dashboard.get_dashboard_data(
                dashboard_id, time_range, user_id
            )
            
            # Add user-specific information
            user_summary = self.usage_tracker.get_user_usage_summary(user_id)
            dashboard_data["user_summary"] = user_summary
            
            # Add cost optimization recommendations
            if self.cost_optimizer:
                recommendations = await self.cost_optimizer.generate_recommendations(user_id)
                dashboard_data["recommendations"] = [rec.to_dict() for rec in recommendations[:5]]  # Top 5
            
            # Add active alerts for this user
            if self.alert_system:
                user_alerts = [
                    alert.to_dict() for alert in self.alert_system.get_active_alerts()
                    if alert.user_id == user_id
                ]
                dashboard_data["active_alerts"] = user_alerts
            
            return dashboard_data
            
        except Exception as e:
            logger.error("Failed to get user dashboard data", user_id=user_id, error=str(e))
            raise
    
    async def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health information."""
        try:
            health_data = {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "uptime_seconds": (datetime.now() - self.system_stats["start_time"]).total_seconds(),
                "components": {},
                "statistics": dict(self.system_stats),
                "alerts": []
            }
            
            # Check component health
            components = {
                "token_estimator": self.token_estimator is not None,
                "pricing_engine": self.pricing_engine is not None,
                "usage_tracker": self.usage_tracker is not None,
                "budget_enforcer": self.budget_enforcer is not None,
                "analytics_dashboard": self.analytics_dashboard is not None,
                "cost_optimizer": self.cost_optimizer is not None,
                "metrics_collector": self.metrics_collector is not None,
                "alert_system": self.alert_system is not None,
            }
            
            for component, status in components.items():
                health_data["components"][component] = {
                    "status": "healthy" if status else "disabled",
                    "enabled": status
                }
            
            # Get metrics summary
            if self.metrics_collector:
                metrics_summary = await self.metrics_collector.get_metrics_summary()
                health_data["metrics_summary"] = metrics_summary
            
            # Get alert summary
            if self.alert_system:
                alert_stats = self.alert_system.get_alert_statistics()
                health_data["alert_summary"] = alert_stats
                
                # Get critical alerts
                critical_alerts = self.alert_system.get_active_alerts(AlertSeverity.CRITICAL)
                if critical_alerts:
                    health_data["status"] = "degraded"
                    health_data["alerts"] = [alert.to_dict() for alert in critical_alerts[:5]]
            
            # Check background task health
            failed_tasks = [task for task in self._background_tasks if task.done() and task.exception()]
            if failed_tasks:
                health_data["status"] = "degraded"
                health_data["failed_background_tasks"] = len(failed_tasks)
            
            return health_data
            
        except Exception as e:
            logger.error("Failed to get system health", error=str(e))
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _perform_system_cleanup(self) -> None:
        """Perform comprehensive system cleanup."""
        try:
            # Clean up components
            await self.usage_tracker.cleanup()
            
            if self.analytics_dashboard:
                await self.analytics_dashboard.cleanup()
            
            if self.cost_optimizer:
                await self.cost_optimizer.cleanup()
            
            await self.metrics_collector.cleanup()
            
            if self.alert_system:
                await self.alert_system.cleanup()
            
            await self.budget_enforcer.cleanup()
            
            logger.info("System cleanup completed")
            
        except Exception as e:
            logger.error("Failed to perform system cleanup", error=str(e))
    
    async def export_user_data(
        self,
        user_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        format_type: str = "json"
    ) -> Dict[str, Any]:
        """Export comprehensive user data for analysis or compliance."""
        try:
            if not start_date:
                start_date = datetime.now() - timedelta(days=30)
            if not end_date:
                end_date = datetime.now()
            
            export_data = {
                "user_id": user_id,
                "export_timestamp": datetime.now().isoformat(),
                "date_range": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat()
                },
                "user_profile": None,
                "usage_summary": None,
                "cost_breakdown": None,
                "alerts": [],
                "recommendations": []
            }
            
            # Get user profile
            if user_id in self.usage_tracker.user_profiles:
                profile = self.usage_tracker.user_profiles[user_id]
                export_data["user_profile"] = {
                    "user_id": profile.user_id,
                    "username": profile.username,
                    "email": profile.email,
                    "user_tier": profile.user_tier,
                    "created_at": profile.created_at.isoformat(),
                    "is_active": profile.is_active
                }
            
            # Get usage summary
            export_data["usage_summary"] = self.usage_tracker.get_user_usage_summary(user_id)
            
            # Get cost optimization recommendations
            if self.cost_optimizer:
                recommendations = await self.cost_optimizer.generate_recommendations(user_id)
                export_data["recommendations"] = [rec.to_dict() for rec in recommendations]
            
            # Get user alerts
            if self.alert_system:
                user_alerts = [
                    alert.to_dict() for alert in self.alert_system.get_active_alerts()
                    if alert.user_id == user_id
                ]
                export_data["alerts"] = user_alerts
            
            return export_data
            
        except Exception as e:
            logger.error("Failed to export user data", user_id=user_id, error=str(e))
            raise
    
    async def stop(self) -> None:
        """Stop the comprehensive usage system."""
        if not self._is_running:
            return
        
        try:
            self._is_running = False
            
            # Cancel background tasks
            for task in self._background_tasks:
                if not task.done():
                    task.cancel()
            
            # Wait for tasks to complete
            if self._background_tasks:
                await asyncio.gather(*self._background_tasks, return_exceptions=True)
            
            # Cleanup all components
            await self._perform_system_cleanup()
            
            logger.info("Comprehensive Usage System stopped")
            
        except Exception as e:
            logger.error("Error stopping system", error=str(e))
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()
        return False


# Convenience functions for common operations

async def create_usage_system(
    storage_path: str = "./data",
    enable_all_features: bool = True
) -> ComprehensiveUsageSystem:
    """Create and start a comprehensive usage system with default configuration."""
    config = SystemConfiguration(
        storage_base_path=storage_path,
        enable_analytics=enable_all_features,
        enable_cost_optimization=enable_all_features,
        enable_alerting=enable_all_features,
        enable_budget_enforcement=enable_all_features
    )
    
    system = ComprehensiveUsageSystem(config)
    await system.start()
    
    return system


async def process_request_with_tracking(
    system: ComprehensiveUsageSystem,
    request: AIRequest,
    provider_type: ProviderType,
    user_id: str
) -> Tuple[bool, Optional[EnforcementResult], Dict[str, Any]]:
    """Convenience function to process a request with full tracking."""
    return await system.process_ai_request(request, provider_type, user_id)


async def record_response_with_tracking(
    system: ComprehensiveUsageSystem,
    request: AIRequest,
    response: Optional[AIResponse],
    provider_type: ProviderType,
    user_id: str,
    success: bool = True,
    error_message: Optional[str] = None
) -> None:
    """Convenience function to record a response with full tracking."""
    await system.record_ai_response(request, response, provider_type, user_id, success, error_message)


# Example usage and integration patterns
if __name__ == "__main__":
    async def main():
        """Example of how to use the comprehensive system."""
        
        # Create and start the system
        async with ComprehensiveUsageSystem() as system:
            # Create a test user
            user_id = await system.create_user(
                username="test_user",
                email="test@example.com",
                user_tier="premium",
                daily_limit=50.0,
                monthly_limit=500.0
            )
            
            # Create a test request
            test_request = AIRequest(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hello, world!"}],
                max_tokens=100
            )
            
            # Process the request
            allowed, enforcement, metadata = await system.process_ai_request(
                test_request, ProviderType.OPENAI, user_id
            )
            
            if allowed:
                print("Request allowed, estimated cost:", metadata.get("estimated_cost"))
                
                # Simulate response
                test_response = AIResponse(
                    request_id=test_request.request_id,
                    provider_type=ProviderType.OPENAI,
                    model="gpt-3.5-turbo",
                    content="Hello! How can I help you?",
                    usage={"input_tokens": 10, "output_tokens": 8},
                    cost=0.00003,  # Example cost
                    latency_ms=250.0
                )
                
                # Record the response
                await system.record_ai_response(
                    test_request, test_response, ProviderType.OPENAI, user_id, success=True
                )
                
                print("Response recorded successfully")
            else:
                print("Request denied:", enforcement.reason if enforcement else "Unknown reason")
            
            # Get user dashboard data
            dashboard_data = await system.get_user_dashboard_data(user_id)
            print("Dashboard data keys:", list(dashboard_data.keys()))
            
            # Get system health
            health = await system.get_system_health()
            print("System status:", health["status"])
    
    # Run the example
    asyncio.run(main())