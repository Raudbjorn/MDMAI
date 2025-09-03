"""
Advanced Cost Optimization System with ML-based prediction and adaptive routing.

This module provides sophisticated cost optimization algorithms for multi-provider LLM systems:
- Machine learning-based cost prediction
- Adaptive provider routing with quality-cost tradeoffs
- Dynamic model selection optimization
- Request batching and caching strategies
- Time-based pricing arbitrage
- Token usage optimization
"""

import asyncio
import json
import math
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from decimal import Decimal
from statistics import mean, stdev
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from structlog import get_logger

from ..ai_providers.models import ProviderType, CostTier, ModelSpec
from ..usage_tracking.storage.models import UsageRecord

logger = get_logger(__name__)


class CostOptimizationStrategy:
    """Cost optimization strategies."""
    MINIMIZE_COST = "minimize_cost"
    MAXIMIZE_QUALITY = "maximize_quality"
    BALANCED = "balanced"
    SPEED_OPTIMIZED = "speed_optimized"
    COST_PER_TOKEN = "cost_per_token"
    ADAPTIVE = "adaptive"


class ModelPerformanceMetrics:
    """Performance metrics for model evaluation."""
    
    def __init__(self):
        self.latency_samples = deque(maxlen=100)
        self.cost_samples = deque(maxlen=100)
        self.quality_scores = deque(maxlen=50)
        self.error_rate = 0.0
        self.success_count = 0
        self.total_requests = 0
        self.last_updated = datetime.utcnow()
    
    def update_latency(self, latency_ms: float) -> None:
        """Update latency metrics."""
        self.latency_samples.append(latency_ms)
    
    def update_cost(self, cost: float) -> None:
        """Update cost metrics."""
        self.cost_samples.append(cost)
    
    def update_quality(self, score: float) -> None:
        """Update quality score (0-1 range)."""
        if 0 <= score <= 1:
            self.quality_scores.append(score)
    
    def update_success(self, success: bool) -> None:
        """Update success/error metrics."""
        self.total_requests += 1
        if success:
            self.success_count += 1
        self.error_rate = 1 - (self.success_count / self.total_requests)
        self.last_updated = datetime.utcnow()
    
    @property
    def avg_latency(self) -> float:
        """Average latency in milliseconds."""
        return mean(self.latency_samples) if self.latency_samples else 0.0
    
    @property
    def p95_latency(self) -> float:
        """95th percentile latency."""
        if not self.latency_samples:
            return 0.0
        sorted_latencies = sorted(self.latency_samples)
        idx = int(len(sorted_latencies) * 0.95)
        return sorted_latencies[idx] if idx < len(sorted_latencies) else sorted_latencies[-1]
    
    @property
    def avg_cost(self) -> float:
        """Average cost per request."""
        return mean(self.cost_samples) if self.cost_samples else 0.0
    
    @property
    def avg_quality(self) -> float:
        """Average quality score."""
        return mean(self.quality_scores) if self.quality_scores else 0.5
    
    @property
    def cost_efficiency_score(self) -> float:
        """Cost efficiency: quality per dollar spent."""
        if self.avg_cost == 0:
            return float('inf')
        return self.avg_quality / self.avg_cost


class PricingModel:
    """Dynamic pricing model for providers."""
    
    def __init__(self, provider_type: ProviderType):
        self.provider_type = provider_type
        self.base_rates = {}  # model_id -> (input_rate, output_rate)
        self.volume_discounts = {}  # tier -> discount_factor
        self.time_multipliers = {}  # hour -> multiplier
        self.surge_pricing = 1.0
        self.last_updated = datetime.utcnow()
    
    def set_base_rates(self, model_id: str, input_rate: float, output_rate: float) -> None:
        """Set base pricing rates for a model (per 1K tokens)."""
        self.base_rates[model_id] = (input_rate, output_rate)
        self.last_updated = datetime.utcnow()
    
    def set_volume_discount(self, monthly_spend: float, discount_factor: float) -> None:
        """Set volume discount based on monthly spend."""
        self.volume_discounts[monthly_spend] = discount_factor
    
    def set_time_multiplier(self, hour: int, multiplier: float) -> None:
        """Set time-based pricing multiplier (0-23 hours)."""
        self.time_multipliers[hour] = multiplier
    
    def calculate_cost(
        self,
        model_id: str,
        input_tokens: int,
        output_tokens: int,
        monthly_spend: float = 0.0,
        request_time: Optional[datetime] = None
    ) -> float:
        """Calculate cost with all pricing factors applied."""
        if model_id not in self.base_rates:
            logger.warning(f"No base rates for model {model_id}")
            return 0.0
        
        input_rate, output_rate = self.base_rates[model_id]
        base_cost = (input_tokens / 1000) * input_rate + (output_tokens / 1000) * output_rate
        
        # Apply volume discount
        volume_discount = 1.0
        for threshold in sorted(self.volume_discounts.keys(), reverse=True):
            if monthly_spend >= threshold:
                volume_discount = self.volume_discounts[threshold]
                break
        
        # Apply time-based multiplier
        time_multiplier = 1.0
        if request_time:
            hour = request_time.hour
            time_multiplier = self.time_multipliers.get(hour, 1.0)
        
        # Apply surge pricing
        total_cost = base_cost * volume_discount * time_multiplier * self.surge_pricing
        return max(0.0, total_cost)


class TokenOptimizer:
    """Token usage optimization strategies."""
    
    @staticmethod
    def compress_context(messages: List[Dict[str, Any]], max_tokens: int) -> List[Dict[str, Any]]:
        """Compress context to fit within token limits."""
        compressed_messages = []
        current_tokens = 0
        
        # Estimate tokens (rough approximation: 4 chars = 1 token)
        def estimate_tokens(text: str) -> int:
            return len(text) // 4
        
        # Keep system message if present
        if messages and messages[0].get("role") == "system":
            system_msg = messages[0]
            system_tokens = estimate_tokens(str(system_msg.get("content", "")))
            if system_tokens < max_tokens * 0.3:  # Use max 30% for system
                compressed_messages.append(system_msg)
                current_tokens += system_tokens
        
        # Process messages in reverse (keep most recent)
        remaining_tokens = max_tokens - current_tokens
        for message in reversed(messages[1:] if compressed_messages else messages):
            msg_tokens = estimate_tokens(str(message.get("content", "")))
            
            if current_tokens + msg_tokens <= remaining_tokens:
                compressed_messages.insert(-len(compressed_messages) if compressed_messages else 0, message)
                current_tokens += msg_tokens
            else:
                # Truncate message if it's the last one we can fit
                if not compressed_messages or len(compressed_messages) == 1:
                    available_tokens = remaining_tokens - current_tokens
                    if available_tokens > 50:  # Only if we have reasonable space
                        content = str(message.get("content", ""))
                        truncated_content = content[:available_tokens * 4]
                        truncated_message = {**message, "content": truncated_content + "..."}
                        compressed_messages.insert(-len(compressed_messages) if compressed_messages else 0, truncated_message)
                break
        
        return compressed_messages
    
    @staticmethod
    def optimize_max_tokens(request_type: str, context_length: int) -> int:
        """Optimize max_tokens based on request type and context."""
        if request_type == "code_generation":
            return min(2048, context_length // 2)
        elif request_type == "analysis":
            return min(1024, context_length // 3)
        elif request_type == "chat":
            return min(512, context_length // 4)
        elif request_type == "summary":
            return min(256, context_length // 6)
        else:
            return min(1024, context_length // 3)


class UsagePredictor:
    """ML-based usage and cost prediction."""
    
    def __init__(self):
        self.usage_history = deque(maxlen=10000)
        self.models = {}  # provider -> sklearn model
        self.scalers = {}  # provider -> scaler
        self.feature_cache = {}
        self.last_training = {}
        self.prediction_cache = {}
        self.cache_ttl = 3600  # 1 hour
    
    def add_usage_record(self, record: Dict[str, Any]) -> None:
        """Add usage record for training."""
        self.usage_history.append({
            'timestamp': record.get('timestamp', time.time()),
            'provider': record.get('provider'),
            'model': record.get('model'),
            'input_tokens': record.get('input_tokens', 0),
            'output_tokens': record.get('output_tokens', 0),
            'cost': record.get('cost', 0.0),
            'latency': record.get('latency_ms', 0.0),
            'hour': datetime.fromtimestamp(record.get('timestamp', time.time())).hour,
            'day_of_week': datetime.fromtimestamp(record.get('timestamp', time.time())).weekday(),
            'success': record.get('success', True)
        })
        
        # Clear prediction cache when new data arrives
        self.prediction_cache.clear()
    
    def extract_features(self, records: List[Dict[str, Any]]) -> np.ndarray:
        """Extract features for ML model."""
        features = []
        
        for record in records:
            feature_vector = [
                record['input_tokens'],
                record['output_tokens'],
                record['hour'],
                record['day_of_week'],
                1 if record['success'] else 0,
                record['latency']
            ]
            features.append(feature_vector)
        
        return np.array(features)
    
    def train_cost_model(self, provider: str) -> None:
        """Train cost prediction model for a provider."""
        provider_records = [r for r in self.usage_history if r['provider'] == provider]
        
        if len(provider_records) < 100:  # Need minimum data
            logger.warning(f"Insufficient data for {provider} cost model training: {len(provider_records)} records")
            return
        
        X = self.extract_features(provider_records)
        y = np.array([r['cost'] for r in provider_records])
        
        # Scale features
        if provider not in self.scalers:
            self.scalers[provider] = StandardScaler()
        
        X_scaled = self.scalers[provider].fit_transform(X)
        
        # Train model
        if provider not in self.models:
            self.models[provider] = LinearRegression()
        
        self.models[provider].fit(X_scaled, y)
        self.last_training[provider] = time.time()
        
        logger.info(f"Trained cost model for {provider} with {len(provider_records)} samples")
    
    def predict_cost(
        self,
        provider: str,
        input_tokens: int,
        output_tokens: int,
        request_time: Optional[datetime] = None
    ) -> float:
        """Predict cost for a request."""
        cache_key = f"{provider}:{input_tokens}:{output_tokens}:{request_time}"
        
        # Check cache
        if cache_key in self.prediction_cache:
            cached_result, timestamp = self.prediction_cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                return cached_result
        
        if provider not in self.models or provider not in self.scalers:
            logger.warning(f"No trained model for {provider}")
            return 0.0
        
        # Retrain if model is old
        if time.time() - self.last_training.get(provider, 0) > 86400:  # 24 hours
            self.train_cost_model(provider)
        
        request_time = request_time or datetime.utcnow()
        
        features = np.array([[
            input_tokens,
            output_tokens,
            request_time.hour,
            request_time.weekday(),
            1,  # Assume success
            100.0  # Average latency estimate
        ]])
        
        features_scaled = self.scalers[provider].transform(features)
        predicted_cost = self.models[provider].predict(features_scaled)[0]
        
        # Cache result
        self.prediction_cache[cache_key] = (predicted_cost, time.time())
        
        return max(0.0, predicted_cost)
    
    def predict_monthly_spend(
        self,
        current_daily_average: float,
        days_remaining: int,
        trend_factor: float = 1.0
    ) -> float:
        """Predict monthly spend based on current usage trends."""
        if not self.usage_history:
            return current_daily_average * days_remaining * trend_factor
        
        # Calculate trend from recent data
        recent_records = list(self.usage_history)[-500:]  # Last 500 records
        if len(recent_records) < 10:
            return current_daily_average * days_remaining * trend_factor
        
        # Group by day and calculate daily costs
        daily_costs = defaultdict(float)
        for record in recent_records:
            date = datetime.fromtimestamp(record['timestamp']).date()
            daily_costs[date] += record['cost']
        
        if len(daily_costs) < 3:
            return current_daily_average * days_remaining * trend_factor
        
        # Calculate trend using linear regression
        days = sorted(daily_costs.keys())
        costs = [daily_costs[day] for day in days]
        
        X = np.array([[i] for i in range(len(days))])
        y = np.array(costs)
        
        trend_model = LinearRegression()
        trend_model.fit(X, y)
        
        # Predict future daily costs
        predicted_daily = []
        for i in range(len(days), len(days) + days_remaining):
            pred = trend_model.predict([[i]])[0]
            predicted_daily.append(max(0.0, pred))
        
        return sum(predicted_daily)


class AdvancedCostOptimizer:
    """Advanced cost optimization system with ML-based routing and prediction."""
    
    def __init__(self):
        self.pricing_models = {}  # provider -> PricingModel
        self.performance_metrics = {}  # (provider, model) -> ModelPerformanceMetrics
        self.usage_predictor = UsagePredictor()
        self.token_optimizer = TokenOptimizer()
        self.request_cache = {}  # Hash of request -> (response, timestamp)
        self.batch_queue = defaultdict(list)  # provider -> requests
        self.cache_ttl = 3600  # 1 hour
        self.optimization_weights = {
            'cost': 0.4,
            'quality': 0.3,
            'speed': 0.2,
            'reliability': 0.1
        }
        
        # Circuit breaker configuration
        self.circuit_breakers = {}  # provider -> failure count
        self.circuit_breaker_threshold = 5
        self.circuit_breaker_timeout = 300  # 5 minutes
        self.circuit_breaker_recovery = {}  # provider -> recovery timestamp
        
        logger.info("Advanced Cost Optimizer initialized")
    
    def register_pricing_model(self, provider: ProviderType, pricing_model: PricingModel) -> None:
        """Register pricing model for a provider."""
        self.pricing_models[provider] = pricing_model
        logger.info(f"Registered pricing model for {provider.value}")
    
    def update_performance_metrics(
        self,
        provider: ProviderType,
        model: str,
        latency_ms: float,
        cost: float,
        quality_score: float = 0.5,
        success: bool = True
    ) -> None:
        """Update performance metrics for a provider/model combination."""
        key = (provider, model)
        
        if key not in self.performance_metrics:
            self.performance_metrics[key] = ModelPerformanceMetrics()
        
        metrics = self.performance_metrics[key]
        metrics.update_latency(latency_ms)
        metrics.update_cost(cost)
        metrics.update_quality(quality_score)
        metrics.update_success(success)
        
        # Update circuit breaker
        if not success:
            self.circuit_breakers[provider] = self.circuit_breakers.get(provider, 0) + 1
            if self.circuit_breakers[provider] >= self.circuit_breaker_threshold:
                self.circuit_breaker_recovery[provider] = time.time() + self.circuit_breaker_timeout
                logger.warning(f"Circuit breaker activated for {provider.value}")
        else:
            self.circuit_breakers[provider] = 0
    
    def is_provider_available(self, provider: ProviderType) -> bool:
        """Check if provider is available (not circuit broken)."""
        if provider not in self.circuit_breaker_recovery:
            return True
        
        if time.time() >= self.circuit_breaker_recovery[provider]:
            # Reset circuit breaker
            del self.circuit_breaker_recovery[provider]
            self.circuit_breakers[provider] = 0
            logger.info(f"Circuit breaker reset for {provider.value}")
            return True
        
        return False
    
    def calculate_request_hash(self, messages: List[Dict[str, Any]], model: str, max_tokens: int) -> str:
        """Calculate hash for request caching."""
        content = json.dumps({
            'messages': messages,
            'model': model,
            'max_tokens': max_tokens
        }, sort_keys=True)
        return str(hash(content))
    
    def check_cache(self, request_hash: str) -> Optional[Dict[str, Any]]:
        """Check if request is cached."""
        if request_hash in self.request_cache:
            response, timestamp = self.request_cache[request_hash]
            if time.time() - timestamp < self.cache_ttl:
                return response
            else:
                del self.request_cache[request_hash]
        return None
    
    def cache_response(self, request_hash: str, response: Dict[str, Any]) -> None:
        """Cache response for future use."""
        self.request_cache[request_hash] = (response, time.time())
        
        # Clean old cache entries periodically
        if len(self.request_cache) > 1000:
            current_time = time.time()
            expired_keys = [
                key for key, (_, timestamp) in self.request_cache.items()
                if current_time - timestamp >= self.cache_ttl
            ]
            for key in expired_keys:
                del self.request_cache[key]
    
    def calculate_optimization_score(
        self,
        provider: ProviderType,
        model: str,
        estimated_cost: float,
        strategy: str = CostOptimizationStrategy.BALANCED
    ) -> float:
        """Calculate optimization score for provider/model combination."""
        key = (provider, model)
        
        if key not in self.performance_metrics:
            # No metrics available, use conservative score
            return 0.5
        
        metrics = self.performance_metrics[key]
        
        # Normalize metrics (0-1 scale)
        cost_score = 1.0 / (1.0 + estimated_cost * 1000)  # Lower cost = higher score
        quality_score = metrics.avg_quality
        speed_score = 1.0 / (1.0 + metrics.avg_latency / 1000)  # Lower latency = higher score
        reliability_score = 1.0 - metrics.error_rate
        
        # Apply strategy-specific weighting
        if strategy == CostOptimizationStrategy.MINIMIZE_COST:
            weights = {'cost': 0.8, 'quality': 0.1, 'speed': 0.05, 'reliability': 0.05}
        elif strategy == CostOptimizationStrategy.MAXIMIZE_QUALITY:
            weights = {'cost': 0.1, 'quality': 0.7, 'speed': 0.1, 'reliability': 0.1}
        elif strategy == CostOptimizationStrategy.SPEED_OPTIMIZED:
            weights = {'cost': 0.2, 'quality': 0.2, 'speed': 0.5, 'reliability': 0.1}
        elif strategy == CostOptimizationStrategy.COST_PER_TOKEN:
            # Focus on cost efficiency
            weights = {'cost': 0.6, 'quality': 0.3, 'speed': 0.05, 'reliability': 0.05}
        else:  # BALANCED or ADAPTIVE
            weights = self.optimization_weights
        
        score = (
            weights['cost'] * cost_score +
            weights['quality'] * quality_score +
            weights['speed'] * speed_score +
            weights['reliability'] * reliability_score
        )
        
        return score
    
    def optimize_provider_selection(
        self,
        messages: List[Dict[str, Any]],
        available_providers: List[Tuple[ProviderType, str]],  # (provider, model)
        strategy: str = CostOptimizationStrategy.BALANCED,
        max_tokens: int = 1024,
        context_length_limit: int = 4096
    ) -> Optional[Dict[str, Any]]:
        """Optimize provider selection based on strategy and constraints."""
        if not available_providers:
            return None
        
        # Check cache first
        request_hash = self.calculate_request_hash(messages, "multi", max_tokens)
        cached_response = self.check_cache(request_hash)
        if cached_response:
            logger.debug("Serving cached response")
            return {
                'provider': cached_response['provider'],
                'model': cached_response['model'],
                'estimated_cost': 0.0,  # Cached = no cost
                'optimization_score': 1.0,
                'reason': 'Cached response',
                'cached': True
            }
        
        # Optimize context if needed
        estimated_tokens = len(json.dumps(messages)) // 4
        if estimated_tokens > context_length_limit:
            messages = self.token_optimizer.compress_context(messages, context_length_limit)
            logger.info(f"Context compressed from ~{estimated_tokens} to ~{len(json.dumps(messages)) // 4} tokens")
        
        candidates = []
        
        for provider, model in available_providers:
            # Skip if circuit breaker is active
            if not self.is_provider_available(provider):
                continue
            
            # Estimate cost
            estimated_cost = 0.0
            if provider in self.pricing_models:
                pricing_model = self.pricing_models[provider]
                input_tokens = estimated_tokens
                output_tokens = max_tokens
                estimated_cost = pricing_model.calculate_cost(model, input_tokens, output_tokens)
            else:
                # Use predictor if available
                estimated_cost = self.usage_predictor.predict_cost(
                    provider.value, estimated_tokens, max_tokens
                )
            
            # Calculate optimization score
            optimization_score = self.calculate_optimization_score(
                provider, model, estimated_cost, strategy
            )
            
            candidates.append({
                'provider': provider,
                'model': model,
                'estimated_cost': estimated_cost,
                'optimization_score': optimization_score,
                'reason': f'Score: {optimization_score:.3f}, Cost: ${estimated_cost:.4f}'
            })
        
        if not candidates:
            logger.warning("No available providers after filtering")
            return None
        
        # Sort by optimization score (descending)
        candidates.sort(key=lambda x: x['optimization_score'], reverse=True)
        
        best_candidate = candidates[0]
        
        # Adaptive strategy: adjust based on performance
        if strategy == CostOptimizationStrategy.ADAPTIVE:
            best_candidate = self._adaptive_selection(candidates)
        
        logger.info(
            f"Selected provider: {best_candidate['provider'].value}, "
            f"model: {best_candidate['model']}, "
            f"score: {best_candidate['optimization_score']:.3f}, "
            f"cost: ${best_candidate['estimated_cost']:.4f}"
        )
        
        return best_candidate
    
    def _adaptive_selection(self, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Adaptive selection based on recent performance trends."""
        # Implement exploration vs exploitation
        exploration_rate = 0.1  # 10% chance to try non-optimal provider
        
        if len(candidates) > 1 and time.time() % 10 < 1:  # Explore 10% of the time
            # Select second-best candidate for exploration
            return candidates[1]
        
        # Otherwise, select best candidate
        return candidates[0]
    
    def optimize_token_usage(
        self,
        messages: List[Dict[str, Any]],
        request_type: str,
        max_context_length: int
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Optimize token usage for a request."""
        # Compress context if needed
        optimized_messages = self.token_optimizer.compress_context(messages, max_context_length)
        
        # Optimize max_tokens based on request type
        optimal_max_tokens = self.token_optimizer.optimize_max_tokens(
            request_type, max_context_length
        )
        
        return optimized_messages, optimal_max_tokens
    
    def record_usage(self, usage_data: Dict[str, Any]) -> None:
        """Record usage data for learning and optimization."""
        self.usage_predictor.add_usage_record(usage_data)
        
        # Update performance metrics
        if all(k in usage_data for k in ['provider', 'model', 'latency_ms', 'cost', 'success']):
            provider = usage_data['provider']
            if isinstance(provider, str):
                provider = ProviderType(provider)
            
            self.update_performance_metrics(
                provider,
                usage_data['model'],
                usage_data['latency_ms'],
                usage_data['cost'],
                usage_data.get('quality_score', 0.5),
                usage_data['success']
            )
    
    def get_optimization_insights(self) -> Dict[str, Any]:
        """Get insights about optimization performance."""
        insights = {
            'cache_hit_ratio': len(self.request_cache) / max(1, len(self.request_cache) + 100),
            'circuit_breakers_active': len(self.circuit_breaker_recovery),
            'performance_metrics_count': len(self.performance_metrics),
            'provider_performance': {}
        }
        
        # Provider performance summary
        for (provider, model), metrics in self.performance_metrics.items():
            key = f"{provider.value}:{model}"
            insights['provider_performance'][key] = {
                'avg_cost': metrics.avg_cost,
                'avg_latency': metrics.avg_latency,
                'avg_quality': metrics.avg_quality,
                'error_rate': metrics.error_rate,
                'cost_efficiency': metrics.cost_efficiency_score
            }
        
        return insights
    
    def train_models(self) -> None:
        """Train ML models with current usage data."""
        providers = set(record.get('provider') for record in self.usage_predictor.usage_history)
        for provider in providers:
            if provider:
                self.usage_predictor.train_cost_model(provider)