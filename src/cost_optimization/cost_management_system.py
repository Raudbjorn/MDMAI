"""
Comprehensive Cost Management System for MDMAI Multi-Provider LLM System.

This is the main orchestrator that integrates all cost optimization components:
- Advanced cost optimization with ML-based routing
- Budget enforcement and limits management
- Real-time alerting and notifications
- Cost prediction and forecasting
- Provider pricing management
- Token optimization and caching
"""

import asyncio
import time
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple, Union

from structlog import get_logger

from .advanced_optimizer import AdvancedCostOptimizer, CostOptimizationStrategy
from .alert_system import AlertSystem, AlertType, AlertSeverity
from .budget_enforcer import BudgetEnforcer, BudgetAction, BudgetLimitType
from .cost_predictor import CostPredictor, ForecastHorizon
from .pricing_engine import PricingEngine, PricingTier, CostComponent
from .token_optimizer import TokenOptimizer, CompressionStrategy

from ..ai_providers.models import ProviderType
from ..usage_tracking.storage.models import UsageRecord

logger = get_logger(__name__)


class CostManagementConfig:
    """Configuration for the cost management system."""
    
    def __init__(self):
        # Optimization settings
        self.default_optimization_strategy = CostOptimizationStrategy.BALANCED
        self.enable_ml_routing = True
        self.enable_caching = True
        self.cache_ttl_hours = 24
        
        # Budget enforcement settings
        self.enable_budget_enforcement = True
        self.default_daily_budget = Decimal("100.0")
        self.default_monthly_budget = Decimal("3000.0")
        self.emergency_brake_multiplier = 2.0
        
        # Alert settings
        self.enable_alerts = True
        self.alert_thresholds = [0.5, 0.8, 0.95]  # 50%, 80%, 95% of budget
        self.anomaly_sensitivity = 2.0
        
        # Token optimization settings
        self.enable_token_optimization = True
        self.default_compression_strategy = CompressionStrategy.PRESERVE_RECENT
        self.max_context_tokens = 8000
        
        # Prediction settings
        self.enable_cost_prediction = True
        self.prediction_update_hours = 6
        self.forecast_horizons = [
            ForecastHorizon.DAILY,
            ForecastHorizon.WEEKLY,
            ForecastHorizon.MONTHLY
        ]
        
        # Provider pricing settings
        self.pricing_update_hours = 24
        self.enable_volume_discounts = True
        self.enable_time_based_pricing = False


class RequestOptimizationResult:
    """Result of request optimization."""
    
    def __init__(self):
        self.approved = False
        self.provider = None
        self.model = None
        self.estimated_cost = Decimal("0.0")
        self.optimization_strategy = None
        self.token_optimization = None
        self.cached_response = None
        self.budget_status = None
        self.recommendations = []
        self.warnings = []
        self.metadata = {}


class CostManagementSystem:
    """Main cost management system orchestrator."""
    
    def __init__(self, config: Optional[CostManagementConfig] = None):
        self.config = config or CostManagementConfig()
        
        # Initialize all subsystems
        self.advanced_optimizer = AdvancedCostOptimizer()
        self.budget_enforcer = BudgetEnforcer()
        self.alert_system = AlertSystem()
        self.cost_predictor = CostPredictor()
        self.pricing_engine = PricingEngine()
        self.token_optimizer = TokenOptimizer()
        
        # System state
        self.initialized = False
        self.last_prediction_update = {}  # user_id -> timestamp
        self.system_metrics = {
            'requests_processed': 0,
            'tokens_optimized': 0,
            'costs_saved': Decimal("0.0"),
            'budget_violations_prevented': 0,
            'alerts_sent': 0
        }
        
        logger.info("Cost Management System initialized")
    
    async def initialize(self) -> None:
        """Initialize the cost management system."""
        if self.initialized:
            return
        
        try:
            # Configure alert channels (example configurations)
            await self._configure_alert_channels()
            
            # Set up default budget rules
            await self._setup_default_budget_rules()
            
            # Initialize pricing models
            await self._initialize_pricing_models()
            
            # Start background tasks
            asyncio.create_task(self._background_maintenance())
            
            self.initialized = True
            logger.info("Cost Management System fully initialized")
            
        except Exception as e:
            logger.error(f"Error initializing Cost Management System: {e}")
            raise
    
    async def optimize_request(
        self,
        user_id: str,
        messages: List[Dict[str, Any]],
        available_providers: List[Tuple[ProviderType, str]],
        max_tokens: int = 1024,
        strategy: Optional[str] = None,
        budget_override: Optional[Decimal] = None
    ) -> RequestOptimizationResult:
        """Optimize a request through the complete cost management pipeline."""
        
        if not self.initialized:
            await self.initialize()
        
        result = RequestOptimizationResult()
        start_time = time.time()
        
        try:
            # Step 1: Token optimization
            if self.config.enable_token_optimization:
                optimized_messages, token_info = await self._optimize_tokens(
                    messages, max_tokens, user_id
                )
                result.token_optimization = token_info
                messages = optimized_messages
            
            # Step 2: Check cache
            if self.config.enable_caching:
                cached_response = self.token_optimizer.semantic_cache.get_cached_response(
                    messages, max_age_hours=self.config.cache_ttl_hours
                )
                if cached_response:
                    result.cached_response = cached_response
                    result.approved = True
                    result.estimated_cost = Decimal("0.0")
                    result.metadata['cache_hit'] = True
                    logger.debug(f"Served cached response for user {user_id}")
                    return result
            
            # Step 3: Provider optimization
            optimization_strategy = strategy or self.config.default_optimization_strategy
            
            optimization_result = self.advanced_optimizer.optimize_provider_selection(
                messages, available_providers, optimization_strategy, max_tokens
            )
            
            if not optimization_result:
                result.approved = False
                result.warnings.append("No suitable provider found")
                return result
            
            result.provider = optimization_result['provider']
            result.model = optimization_result['model']
            result.estimated_cost = Decimal(str(optimization_result['estimated_cost']))
            result.optimization_strategy = optimization_strategy
            
            # Step 4: Budget enforcement
            if self.config.enable_budget_enforcement:
                budget_decision, modifications, violations = await self.budget_enforcer.check_budget_approval(
                    user_id,
                    result.estimated_cost,
                    {
                        'provider': result.provider,
                        'model': result.model,
                        'max_tokens': max_tokens
                    }
                )
                
                result.budget_status = {
                    'action': budget_decision.value,
                    'modifications': modifications,
                    'violations': violations
                }
                
                if budget_decision in [BudgetAction.EMERGENCY_STOP, BudgetAction.BLOCK]:
                    result.approved = False
                    result.warnings.extend(violations)
                    
                    # Send alert
                    if self.config.enable_alerts:
                        await self.alert_system.create_alert(
                            AlertType.BUDGET_EXCEEDED,
                            AlertSeverity.CRITICAL,
                            "Budget Limit Exceeded",
                            f"Request blocked due to budget violation: {', '.join(violations)}",
                            user_id
                        )
                    
                    return result
                
                elif budget_decision == BudgetAction.DOWNGRADE and modifications:
                    # Apply modifications
                    if 'model' in modifications:
                        result.model = modifications['model']
                    if 'provider' in modifications:
                        result.provider = ProviderType(modifications['provider'])
                    if 'max_tokens' in modifications:
                        max_tokens = modifications['max_tokens']
                    
                    # Recalculate cost with modifications
                    result.estimated_cost = await self._recalculate_cost(
                        result.provider, result.model, messages, max_tokens, user_id
                    )
                    
                    result.recommendations.append("Request modified to stay within budget")
            
            # Step 5: Final approval
            result.approved = True
            result.metadata.update({
                'processing_time_ms': (time.time() - start_time) * 1000,
                'original_messages': len(messages),
                'optimization_applied': True
            })
            
            # Update system metrics
            self.system_metrics['requests_processed'] += 1
            
            logger.info(f"Request optimized for user {user_id}: {result.provider.value}:{result.model}, ${result.estimated_cost}")
            
        except Exception as e:
            logger.error(f"Error optimizing request for user {user_id}: {e}")
            result.approved = False
            result.warnings.append(f"Optimization error: {str(e)}")
        
        return result
    
    async def record_usage(
        self,
        user_id: str,
        usage_record: Dict[str, Any],
        response_data: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record usage data across all subsystems."""
        
        try:
            # Record in advanced optimizer
            self.advanced_optimizer.record_usage(usage_record)
            
            # Record in cost predictor
            self.cost_predictor.add_usage_data(user_id, [usage_record])
            
            # Update budget enforcement
            if 'cost' in usage_record:
                cost = Decimal(str(usage_record['cost']))
                self.budget_enforcer.velocity_monitor.add_spending(user_id, cost)
            
            # Cache response if provided
            if (response_data and 'messages' in usage_record and 
                self.config.enable_caching):
                
                self.token_optimizer.semantic_cache.cache_response(
                    usage_record['messages'],
                    response_data,
                    usage_record.get('model', '')
                )
            
            # Check for alerts
            if self.config.enable_alerts:
                await self._check_usage_alerts(user_id, usage_record)
            
            # Update predictions if needed
            await self._update_predictions_if_needed(user_id)
            
            # Update system metrics
            if 'cost' in usage_record:
                self.system_metrics['costs_saved'] += Decimal(str(usage_record.get('cost_saved', 0)))
            
            logger.debug(f"Recorded usage for user {user_id}")
            
        except Exception as e:
            logger.error(f"Error recording usage for user {user_id}: {e}")
    
    async def get_cost_forecast(
        self,
        user_id: str,
        horizon: str = ForecastHorizon.MONTHLY,
        periods_ahead: int = 30
    ) -> Optional[Dict[str, Any]]:
        """Get cost forecast for a user."""
        
        try:
            # Get cost prediction
            forecast = await self.cost_predictor.predict_costs(
                user_id, horizon, periods_ahead
            )
            
            if not forecast:
                return None
            
            # Get insights
            insights = self.cost_predictor.get_prediction_insights(user_id, forecast)
            
            # Get budget status
            budget_status = self.budget_enforcer.get_budget_status(user_id)
            
            # Get spending forecast
            spending_forecast = self.budget_enforcer.get_spending_forecast(user_id, periods_ahead)
            
            return {
                'forecast': {
                    'horizon': forecast.horizon,
                    'predictions': forecast.predictions,
                    'timestamps': [ts.isoformat() for ts in forecast.timestamps],
                    'confidence_intervals': forecast.confidence_intervals,
                    'total_predicted_cost': forecast.get_total_predicted_cost()
                },
                'insights': insights,
                'budget_status': budget_status,
                'spending_forecast': spending_forecast,
                'generated_at': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating cost forecast for user {user_id}: {e}")
            return None
    
    async def create_user_budget(
        self,
        user_id: str,
        daily_limit: Optional[Decimal] = None,
        monthly_limit: Optional[Decimal] = None,
        custom_limits: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Create standard budget limits for a user."""
        
        try:
            # Use defaults if not specified
            daily_limit = daily_limit or self.config.default_daily_budget
            monthly_limit = monthly_limit or self.config.default_monthly_budget
            emergency_limit = daily_limit * Decimal(str(self.config.emergency_brake_multiplier))
            
            # Create standard budget limits
            self.budget_enforcer.create_standard_budget_limits(
                user_id,
                daily_soft=daily_limit * Decimal("0.8"),  # 80% warning
                daily_hard=daily_limit,
                monthly_soft=monthly_limit * Decimal("0.8"),
                monthly_hard=monthly_limit,
                emergency_daily=emergency_limit
            )
            
            # Set up alerts
            if self.config.enable_alerts:
                for threshold in self.config.alert_thresholds:
                    await self._setup_budget_alert(user_id, threshold)
            
            logger.info(f"Created budget limits for user {user_id}: daily=${daily_limit}, monthly=${monthly_limit}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating budget for user {user_id}: {e}")
            return False
    
    async def get_user_analytics(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive analytics for a user."""
        
        try:
            analytics = {
                'user_id': user_id,
                'generated_at': datetime.utcnow().isoformat(),
                'budget_status': {},
                'usage_patterns': {},
                'cost_predictions': {},
                'optimization_stats': {},
                'alerts': {},
                'recommendations': []
            }
            
            # Budget status
            analytics['budget_status'] = self.budget_enforcer.get_budget_status(user_id)
            
            # Usage patterns
            analytics['usage_patterns'] = self.cost_predictor.analyze_usage_patterns(user_id)
            
            # Model performance
            analytics['model_performance'] = self.cost_predictor.get_model_performance(user_id)
            
            # Cost predictions for different horizons
            for horizon in self.config.forecast_horizons:
                forecast = await self.cost_predictor.predict_costs(user_id, horizon, 7)
                if forecast:
                    analytics['cost_predictions'][horizon] = {
                        'total_predicted': forecast.get_total_predicted_cost(),
                        'next_period': forecast.predictions[0] if forecast.predictions else 0
                    }
            
            # Optimization stats
            analytics['optimization_stats'] = self.advanced_optimizer.get_optimization_insights()
            
            # Token optimization recommendations
            recent_messages = []  # Would get from user's recent activity
            if recent_messages:
                analytics['token_recommendations'] = self.token_optimizer.get_optimization_recommendations(
                    recent_messages
                )
            
            # Recent alerts
            analytics['recent_alerts'] = self.alert_system.get_user_notifications(user_id, 10)
            
            # Generate recommendations
            analytics['recommendations'] = await self._generate_recommendations(user_id, analytics)
            
            return analytics
            
        except Exception as e:
            logger.error(f"Error generating analytics for user {user_id}: {e}")
            return {'error': str(e)}
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status and metrics."""
        
        try:
            return {
                'system_status': 'operational' if self.initialized else 'initializing',
                'metrics': dict(self.system_metrics),
                'subsystem_status': {
                    'advanced_optimizer': 'active',
                    'budget_enforcer': 'active' if self.config.enable_budget_enforcement else 'disabled',
                    'alert_system': 'active' if self.config.enable_alerts else 'disabled',
                    'cost_predictor': 'active' if self.config.enable_cost_prediction else 'disabled',
                    'pricing_engine': 'active',
                    'token_optimizer': 'active' if self.config.enable_token_optimization else 'disabled'
                },
                'cache_stats': self.token_optimizer.semantic_cache.get_cache_stats(),
                'alert_stats': self.alert_system.get_alert_statistics(),
                'pricing_stats': self.pricing_engine.get_pricing_analytics(),
                'configuration': {
                    'ml_routing_enabled': self.config.enable_ml_routing,
                    'budget_enforcement_enabled': self.config.enable_budget_enforcement,
                    'alerts_enabled': self.config.enable_alerts,
                    'token_optimization_enabled': self.config.enable_token_optimization,
                    'caching_enabled': self.config.enable_caching
                },
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {'error': str(e)}
    
    # Private helper methods
    
    async def _optimize_tokens(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        user_id: str
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Optimize tokens for the conversation."""
        
        target_tokens = min(max_tokens, self.config.max_context_tokens)
        
        optimized_messages, optimization_info = self.token_optimizer.optimize_conversation(
            messages,
            target_tokens,
            self.config.default_compression_strategy,
            'default',  # Would determine based on selected provider
            preserve_system=True
        )
        
        # Update metrics
        self.system_metrics['tokens_optimized'] += optimization_info.get('tokens_saved', 0)
        
        return optimized_messages, optimization_info
    
    async def _recalculate_cost(
        self,
        provider: ProviderType,
        model: str,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        user_id: str
    ) -> Decimal:
        """Recalculate cost after modifications."""
        
        # Estimate input tokens
        input_tokens = sum(
            self.token_optimizer.token_estimator.estimate_message_tokens(msg)
            for msg in messages
        )
        
        # Calculate cost using pricing engine
        usage_amounts = {
            CostComponent.INPUT_TOKENS: input_tokens,
            CostComponent.OUTPUT_TOKENS: max_tokens
        }
        
        cost_result = self.pricing_engine.calculate_cost(
            provider, model, usage_amounts, PricingTier.PRO
        )
        
        if cost_result:
            return Decimal(str(cost_result['total_cost']))
        
        return Decimal("0.0")
    
    async def _check_usage_alerts(
        self,
        user_id: str,
        usage_record: Dict[str, Any]
    ) -> None:
        """Check if usage should trigger alerts."""
        
        try:
            # Check for cost anomalies
            if 'cost' in usage_record:
                cost = float(usage_record['cost'])
                
                # Simple anomaly detection - would be more sophisticated in production
                user_trend = self.alert_system.get_trend_analyzer(f"cost_{user_id}")
                user_trend.add_data_point(datetime.utcnow(), cost)
                
                anomalies = user_trend.detect_anomalies(self.config.anomaly_sensitivity)
                
                for anomaly in anomalies:
                    await self.alert_system.check_cost_anomaly_alert(
                        user_id,
                        f"API Cost",
                        anomaly['value'],
                        (0, anomaly['expected_max'])
                    )
            
            # Check budget thresholds
            budget_status = self.budget_enforcer.get_budget_status(user_id)
            
            for limit_info in budget_status['limits']:
                if limit_info['enabled']:
                    utilization = limit_info['utilization_percentage']
                    
                    for threshold in self.config.alert_thresholds:
                        if utilization >= threshold * 100:
                            await self.alert_system.check_budget_threshold_alert(
                                user_id,
                                limit_info['name'],
                                Decimal(str(limit_info['spent'])),
                                Decimal(str(limit_info['amount'])),
                                threshold * 100
                            )
                            break
            
            # Check spending velocity
            velocity = budget_status['spending_velocity']
            daily_limits = [
                limit for limit in budget_status['limits']
                if limit['period'] == 'daily' and limit['enabled']
            ]
            
            if daily_limits and velocity > 0:
                daily_budget = daily_limits[0]['amount']
                predicted_daily = velocity * 24 * 60  # Convert per-minute to per-day
                
                await self.alert_system.check_velocity_warning_alert(
                    user_id, velocity, predicted_daily, daily_budget
                )
                
        except Exception as e:
            logger.error(f"Error checking usage alerts for user {user_id}: {e}")
    
    async def _update_predictions_if_needed(self, user_id: str) -> None:
        """Update cost predictions if needed."""
        
        if not self.config.enable_cost_prediction:
            return
        
        current_time = time.time()
        last_update = self.last_prediction_update.get(user_id, 0)
        
        if current_time - last_update > (self.config.prediction_update_hours * 3600):
            try:
                # Retrain models if needed
                self.cost_predictor.retrain_models_if_needed(user_id)
                
                self.last_prediction_update[user_id] = current_time
                logger.debug(f"Updated predictions for user {user_id}")
                
            except Exception as e:
                logger.error(f"Error updating predictions for user {user_id}: {e}")
    
    async def _generate_recommendations(
        self,
        user_id: str,
        analytics: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate personalized cost optimization recommendations."""
        
        recommendations = []
        
        try:
            # Budget utilization recommendations
            budget_status = analytics.get('budget_status', {})
            for limit_info in budget_status.get('limits', []):
                utilization = limit_info.get('utilization_percentage', 0)
                
                if utilization > 90:
                    recommendations.append({
                        'type': 'budget_warning',
                        'priority': 'high',
                        'title': f"High {limit_info['name']} Utilization",
                        'description': f"You've used {utilization:.1f}% of your {limit_info['name']}",
                        'action': 'Consider upgrading your budget or optimizing usage patterns'
                    })
                elif utilization < 30:
                    recommendations.append({
                        'type': 'budget_optimization',
                        'priority': 'low',
                        'title': f"Low {limit_info['name']} Utilization",
                        'description': f"You're only using {utilization:.1f}% of your {limit_info['name']}",
                        'action': 'Consider reducing your budget limit to save costs'
                    })
            
            # Usage pattern recommendations
            usage_patterns = analytics.get('usage_patterns', {})
            pattern_type = usage_patterns.get('pattern_type')
            
            if pattern_type == 'burst':
                recommendations.append({
                    'type': 'usage_pattern',
                    'priority': 'medium',
                    'title': 'Burst Usage Pattern Detected',
                    'description': 'Your usage shows irregular bursts of activity',
                    'action': 'Consider using request batching or caching to smooth out usage'
                })
            elif pattern_type == 'growing':
                recommendations.append({
                    'type': 'usage_pattern',
                    'priority': 'medium',
                    'title': 'Growing Usage Trend',
                    'description': 'Your usage is steadily increasing',
                    'action': 'Monitor budget limits and consider volume discounts'
                })
            
            # Cost prediction recommendations
            cost_predictions = analytics.get('cost_predictions', {})
            
            monthly_prediction = cost_predictions.get(ForecastHorizon.MONTHLY, {})
            if monthly_prediction.get('total_predicted', 0) > 1000:
                recommendations.append({
                    'type': 'cost_forecast',
                    'priority': 'medium',
                    'title': 'High Predicted Monthly Cost',
                    'description': f"Predicted monthly cost: ${monthly_prediction['total_predicted']:.2f}",
                    'action': 'Review volume discounts and consider cheaper model alternatives'
                })
            
            # Token optimization recommendations
            token_recommendations = analytics.get('token_recommendations', {})
            for opportunity in token_recommendations.get('optimization_opportunities', []):
                recommendations.append({
                    'type': 'token_optimization',
                    'priority': 'medium',
                    'title': f"Token Optimization: {opportunity['type']}",
                    'description': opportunity['description'],
                    'action': f"Potential savings: {opportunity['potential_savings']}"
                })
            
        except Exception as e:
            logger.error(f"Error generating recommendations for user {user_id}: {e}")
        
        return recommendations
    
    async def _configure_alert_channels(self) -> None:
        """Configure alert notification channels."""
        
        # Example configurations - would be customized per deployment
        
        # Email channel
        email_config = {
            'smtp_server': 'localhost',
            'smtp_port': 587,
            'from_email': 'noreply@mdmai.com',
            'to_emails': ['admin@mdmai.com']
        }
        
        # Webhook channel  
        webhook_config = {
            'url': 'https://api.mdmai.com/webhooks/cost-alerts',
            'headers': {'Content-Type': 'application/json'},
            'timeout': 30
        }
        
        from .alert_system import AlertChannel
        
        self.alert_system.configure_channel(AlertChannel.EMAIL, email_config)
        self.alert_system.configure_channel(AlertChannel.WEBHOOK, webhook_config)
        
        logger.info("Configured alert channels")
    
    async def _setup_default_budget_rules(self) -> None:
        """Set up default budget alert rules."""
        
        from .alert_system import AlertRule
        
        # Budget threshold rule
        budget_rule = AlertRule(
            rule_id="default_budget_threshold",
            name="Budget Threshold Alert",
            alert_type=AlertType.BUDGET_THRESHOLD,
            condition={'threshold': 0.8},
            severity=AlertSeverity.WARNING,
            channels=[AlertChannel.EMAIL, AlertChannel.IN_APP],
            rate_limit_minutes=60
        )
        
        self.alert_system.add_alert_rule(budget_rule)
        
        logger.info("Set up default budget rules")
    
    async def _setup_budget_alert(self, user_id: str, threshold: float) -> None:
        """Set up budget alert for specific user and threshold."""
        
        from .alert_system import AlertRule
        
        rule_id = f"budget_alert_{user_id}_{int(threshold*100)}"
        
        alert_rule = AlertRule(
            rule_id=rule_id,
            name=f"Budget Alert {int(threshold*100)}%",
            alert_type=AlertType.BUDGET_THRESHOLD,
            condition={'threshold': threshold},
            severity=AlertSeverity.WARNING if threshold < 0.9 else AlertSeverity.CRITICAL,
            channels=[AlertChannel.IN_APP],
            rate_limit_minutes=30
        )
        
        self.alert_system.add_alert_rule(alert_rule)
    
    async def _initialize_pricing_models(self) -> None:
        """Initialize pricing models (already done in PricingEngine.__init__)."""
        
        # Pricing models are initialized in PricingEngine constructor
        # This method could be used to load custom pricing from configuration
        logger.info("Pricing models initialized")
    
    async def _background_maintenance(self) -> None:
        """Background maintenance tasks."""
        
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                # Update pricing from APIs
                if self.config.pricing_update_hours:
                    await self.pricing_engine.update_all_pricing()
                
                # Clean up caches
                self.token_optimizer.semantic_cache._cleanup_cache()
                self.pricing_engine.clear_pricing_cache()
                
                # Train ML models
                self.advanced_optimizer.train_models()
                
                logger.debug("Background maintenance completed")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in background maintenance: {e}")