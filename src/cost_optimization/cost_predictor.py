"""
Cost Prediction Engine with Pattern Recognition and Forecasting Models.

This module provides sophisticated cost prediction capabilities:
- Machine learning-based cost forecasting
- Usage pattern recognition and classification
- Seasonal trend analysis
- Anomaly detection in spending patterns
- Predictive budget planning
- Multi-horizon forecasting (hourly, daily, weekly, monthly)
"""

import asyncio
import json
import math
import pickle
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from decimal import Decimal
from statistics import mean, median, stdev
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from structlog import get_logger

logger = get_logger(__name__)


class ForecastHorizon:
    """Forecast horizon definitions."""
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


class UsagePattern:
    """Usage pattern classification."""
    STEADY = "steady"
    BURST = "burst"
    PERIODIC = "periodic"
    DECLINING = "declining"
    GROWING = "growing"
    IRREGULAR = "irregular"


class SeasonalityType:
    """Types of seasonality detected."""
    HOURLY = "hourly"  # Different usage by hour
    DAILY = "daily"    # Different usage by day of week
    WEEKLY = "weekly"  # Different usage by week
    MONTHLY = "monthly"  # Different usage by month


class CostForecast:
    """Cost forecast result."""
    
    def __init__(
        self,
        horizon: str,
        predictions: List[float],
        confidence_intervals: List[Tuple[float, float]],
        timestamps: List[datetime],
        metadata: Dict[str, Any] = None
    ):
        self.horizon = horizon
        self.predictions = predictions
        self.confidence_intervals = confidence_intervals
        self.timestamps = timestamps
        self.metadata = metadata or {}
        self.created_at = datetime.utcnow()
    
    def get_total_predicted_cost(self) -> float:
        """Get total predicted cost across all time periods."""
        return sum(self.predictions)
    
    def get_prediction_for_date(self, target_date: datetime) -> Optional[float]:
        """Get prediction for a specific date."""
        for timestamp, prediction in zip(self.timestamps, self.predictions):
            if timestamp.date() == target_date.date():
                return prediction
        return None


class PatternRecognizer:
    """Recognize and classify usage patterns."""
    
    def __init__(self, min_data_points: int = 50):
        self.min_data_points = min_data_points
        self.pattern_cache = {}
        self.cache_ttl = 3600  # 1 hour
    
    def analyze_pattern(
        self,
        usage_data: List[Dict[str, Any]],
        user_id: str
    ) -> Dict[str, Any]:
        """Analyze usage pattern for a user."""
        cache_key = f"{user_id}_{len(usage_data)}"
        
        # Check cache
        if cache_key in self.pattern_cache:
            cached_result, timestamp = self.pattern_cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                return cached_result
        
        if len(usage_data) < self.min_data_points:
            return {
                'pattern_type': UsagePattern.IRREGULAR,
                'confidence': 0.0,
                'characteristics': {},
                'seasonality': [],
                'trend': 'insufficient_data'
            }
        
        # Extract time series
        timestamps = [datetime.fromisoformat(d['timestamp']) if isinstance(d['timestamp'], str) else d['timestamp'] for d in usage_data]
        values = [d['cost'] for d in usage_data]
        
        # Sort by timestamp
        sorted_data = sorted(zip(timestamps, values))
        timestamps, values = zip(*sorted_data)
        
        # Analyze different aspects
        pattern_type = self._classify_pattern(values)
        seasonality = self._detect_seasonality(timestamps, values)
        trend = self._analyze_trend(values)
        characteristics = self._calculate_characteristics(values)
        
        result = {
            'pattern_type': pattern_type,
            'confidence': self._calculate_confidence(values),
            'characteristics': characteristics,
            'seasonality': seasonality,
            'trend': trend,
            'analysis_timestamp': datetime.utcnow().isoformat()
        }
        
        # Cache result
        self.pattern_cache[cache_key] = (result, time.time())
        return result
    
    def _classify_pattern(self, values: List[float]) -> str:
        """Classify the usage pattern."""
        if len(values) < 10:
            return UsagePattern.IRREGULAR
        
        # Calculate statistics
        mean_val = mean(values)
        std_val = stdev(values) if len(values) > 1 else 0
        coefficient_of_variation = std_val / mean_val if mean_val > 0 else 0
        
        # Calculate trend
        x = list(range(len(values)))
        trend_slope = np.polyfit(x, values, 1)[0] if len(values) > 1 else 0
        
        # Classify based on characteristics
        if coefficient_of_variation < 0.2:
            return UsagePattern.STEADY
        elif coefficient_of_variation > 1.0:
            # Check for burst pattern (high values followed by low values)
            peaks = self._find_peaks(values)
            if len(peaks) > 0 and max(values) > mean_val * 3:
                return UsagePattern.BURST
            else:
                return UsagePattern.IRREGULAR
        elif trend_slope > mean_val * 0.01:  # Growing trend
            return UsagePattern.GROWING
        elif trend_slope < -mean_val * 0.01:  # Declining trend
            return UsagePattern.DECLINING
        else:
            # Check for periodic pattern
            if self._has_periodic_pattern(values):
                return UsagePattern.PERIODIC
            else:
                return UsagePattern.STEADY
    
    def _find_peaks(self, values: List[float], prominence: float = 0.5) -> List[int]:
        """Find peaks in the data."""
        peaks = []
        if len(values) < 3:
            return peaks
        
        mean_val = mean(values)
        threshold = mean_val * (1 + prominence)
        
        for i in range(1, len(values) - 1):
            if values[i] > values[i-1] and values[i] > values[i+1] and values[i] > threshold:
                peaks.append(i)
        
        return peaks
    
    def _has_periodic_pattern(self, values: List[float]) -> bool:
        """Check if data has periodic pattern using autocorrelation."""
        if len(values) < 20:
            return False
        
        # Simple autocorrelation check
        n = len(values)
        mean_val = mean(values)
        
        # Check for patterns at different lags
        for lag in range(2, min(n // 4, 50)):
            correlation = 0
            count = 0
            
            for i in range(lag, n):
                correlation += (values[i] - mean_val) * (values[i - lag] - mean_val)
                count += 1
            
            if count > 0:
                correlation /= count
                # Normalize
                variance = sum((v - mean_val) ** 2 for v in values) / len(values)
                # Check for zero variance to prevent division by zero
                if variance > 1e-10:  # Use small epsilon instead of exact zero check
                if variance > 1e-10:  # Use small epsilon instead of exact zero check
                    correlation /= variance
                    # Strong correlation indicates periodicity
                    if correlation > 0.5:
                        return True
        
        return False
    
    def _detect_seasonality(
        self,
        timestamps: List[datetime],
        values: List[float]
    ) -> List[Dict[str, Any]]:
        """Detect seasonal patterns in usage."""
        seasonality = []
        
        if len(timestamps) < 50:
            return seasonality
        
        # Group by hour of day
        hourly_data = defaultdict(list)
        for ts, val in zip(timestamps, values):
            hourly_data[ts.hour].append(val)
        
        # Check hourly seasonality
        if len(hourly_data) >= 12:  # At least half the hours
            hourly_means = [mean(hourly_data.get(h, [0])) for h in range(24)]
            hourly_std = stdev(hourly_means) if len(hourly_means) > 1 else 0
            hourly_mean = mean(hourly_means)
            
            if hourly_std > hourly_mean * 0.3:  # Significant variation
                seasonality.append({
                    'type': SeasonalityType.HOURLY,
                    'strength': min(1.0, hourly_std / hourly_mean),
                    'peak_hours': [h for h in range(24) if hourly_means[h] > hourly_mean * 1.5],
                    'low_hours': [h for h in range(24) if hourly_means[h] < hourly_mean * 0.5]
                })
        
        # Group by day of week
        daily_data = defaultdict(list)
        for ts, val in zip(timestamps, values):
            daily_data[ts.weekday()].append(val)
        
        # Check daily seasonality
        if len(daily_data) >= 5:  # At least 5 days
            daily_means = [mean(daily_data.get(d, [0])) for d in range(7)]
            daily_std = stdev(daily_means) if len(daily_means) > 1 else 0
            daily_mean = mean(daily_means)
            
            if daily_std > daily_mean * 0.2:  # Significant variation
                day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
                seasonality.append({
                    'type': SeasonalityType.DAILY,
                    'strength': min(1.0, daily_std / daily_mean),
                    'peak_days': [day_names[d] for d in range(7) if daily_means[d] > daily_mean * 1.3],
                    'low_days': [day_names[d] for d in range(7) if daily_means[d] < daily_mean * 0.7]
                })
        
        return seasonality
    
    def _analyze_trend(self, values: List[float]) -> str:
        """Analyze trend in the data."""
        if len(values) < 10:
            return "insufficient_data"
        
        # Use linear regression to find trend
        x = np.array(range(len(values))).reshape(-1, 1)
        y = np.array(values)
        
        model = LinearRegression()
        model.fit(x, y)
        
        slope = model.coef_[0]
        mean_val = mean(values)
        
        # Classify trend
        if slope > mean_val * 0.01:
            return "increasing"
        elif slope < -mean_val * 0.01:
            return "decreasing"
        else:
            return "stable"
    
    def _calculate_characteristics(self, values: List[float]) -> Dict[str, float]:
        """Calculate statistical characteristics."""
        if not values:
            return {}
        
        characteristics = {
            'mean': mean(values),
            'median': median(values),
            'std': stdev(values) if len(values) > 1 else 0,
            'min': min(values),
            'max': max(values),
            'range': max(values) - min(values)
        }
        
        # Coefficient of variation
        if characteristics['mean'] > 0:
            characteristics['cv'] = characteristics['std'] / characteristics['mean']
        else:
            characteristics['cv'] = 0
        
        # Skewness (simplified calculation)
        mean_val = characteristics['mean']
        std_val = characteristics['std']
        if std_val > 0:
            skewness = sum((v - mean_val) ** 3 for v in values) / (len(values) * std_val ** 3)
            characteristics['skewness'] = skewness
        
        return characteristics
    
    def _calculate_confidence(self, values: List[float]) -> float:
        """Calculate confidence in pattern classification."""
        if len(values) < 10:
            return 0.0
        
        # Base confidence on data amount and regularity
        data_factor = min(1.0, len(values) / 100)  # More data = higher confidence
        
        # Regularity factor
        cv = stdev(values) / mean(values) if mean(values) > 0 else float('inf')
        regularity_factor = max(0.0, 1.0 - cv)  # Lower variation = higher confidence
        
        return (data_factor + regularity_factor) / 2


class CostPredictor:
    """Advanced cost prediction engine with ML models."""
    
    def __init__(self):
        self.models = {}  # user_id -> {horizon: model}
        self.scalers = {}  # user_id -> scaler
        self.pattern_recognizer = PatternRecognizer()
        self.training_data = {}  # user_id -> training data
        self.model_performance = {}  # user_id -> performance metrics
        self.prediction_cache = {}  # Cache for predictions
        self.cache_ttl = 1800  # 30 minutes
        
        logger.info("Cost Predictor initialized")
    
    def add_usage_data(self, user_id: str, usage_records: List[Dict[str, Any]]) -> None:
        """Add usage data for training."""
        if user_id not in self.training_data:
            self.training_data[user_id] = []
        
        # Add new records
        for record in usage_records:
            # Ensure timestamp is datetime
            if isinstance(record.get('timestamp'), str):
                record['timestamp'] = datetime.fromisoformat(record['timestamp'])
            
            self.training_data[user_id].append(record)
        
        # Keep only recent data (last 90 days)
        cutoff_date = datetime.utcnow() - timedelta(days=90)
        self.training_data[user_id] = [
            record for record in self.training_data[user_id]
            if record['timestamp'] >= cutoff_date
        ]
        
        # Sort by timestamp
        self.training_data[user_id].sort(key=lambda x: x['timestamp'])
        
        # Clear prediction cache when new data arrives
        if user_id in self.prediction_cache:
            del self.prediction_cache[user_id]
    
    def prepare_features(
        self,
        usage_data: List[Dict[str, Any]],
        horizon: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features for ML model training."""
        if len(usage_data) < 10:
            return np.array([]), np.array([])
        
        features = []
        targets = []
        
        # Group data by time period based on horizon
        if horizon == ForecastHorizon.HOURLY:
            time_groups = self._group_by_hour(usage_data)
            lookback_periods = 24  # Look back 24 hours
        elif horizon == ForecastHorizon.DAILY:
            time_groups = self._group_by_day(usage_data)
            lookback_periods = 14  # Look back 14 days
        elif horizon == ForecastHorizon.WEEKLY:
            time_groups = self._group_by_week(usage_data)
            lookback_periods = 8   # Look back 8 weeks
        else:  # Monthly
            time_groups = self._group_by_month(usage_data)
            lookback_periods = 6   # Look back 6 months
        
        # Create features from time series
        time_keys = sorted(time_groups.keys())
        
        for i in range(lookback_periods, len(time_keys)):
            # Historical values as features
            historical_values = []
            for j in range(lookback_periods):
                period_key = time_keys[i - lookback_periods + j]
                period_cost = sum(record['cost'] for record in time_groups[period_key])
                historical_values.append(period_cost)
            
            # Time-based features
            current_time = time_keys[i]
            time_features = self._extract_time_features(current_time, horizon)
            
            # Combine all features
            feature_vector = historical_values + time_features
            features.append(feature_vector)
            
            # Target is the next period's cost
            if i < len(time_keys):
                target_cost = sum(record['cost'] for record in time_groups[time_keys[i]])
                targets.append(target_cost)
        
        return np.array(features), np.array(targets)
    
    def _group_by_hour(self, usage_data: List[Dict[str, Any]]) -> Dict[datetime, List[Dict[str, Any]]]:
        """Group usage data by hour."""
        groups = defaultdict(list)
        for record in usage_data:
            hour_key = record['timestamp'].replace(minute=0, second=0, microsecond=0)
            groups[hour_key].append(record)
        return groups
    
    def _group_by_day(self, usage_data: List[Dict[str, Any]]) -> Dict[datetime, List[Dict[str, Any]]]:
        """Group usage data by day."""
        groups = defaultdict(list)
        for record in usage_data:
            day_key = record['timestamp'].replace(hour=0, minute=0, second=0, microsecond=0)
            groups[day_key].append(record)
        return groups
    
    def _group_by_week(self, usage_data: List[Dict[str, Any]]) -> Dict[datetime, List[Dict[str, Any]]]:
        """Group usage data by week."""
        groups = defaultdict(list)
        for record in usage_data:
            # Get Monday of the week
            days_since_monday = record['timestamp'].weekday()
            monday = record['timestamp'] - timedelta(days=days_since_monday)
            week_key = monday.replace(hour=0, minute=0, second=0, microsecond=0)
            groups[week_key].append(record)
        return groups
    
    def _group_by_month(self, usage_data: List[Dict[str, Any]]) -> Dict[datetime, List[Dict[str, Any]]]:
        """Group usage data by month."""
        groups = defaultdict(list)
        for record in usage_data:
            month_key = record['timestamp'].replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            groups[month_key].append(record)
        return groups
    
    def _extract_time_features(self, timestamp: datetime, horizon: str) -> List[float]:
        """Extract time-based features."""
        features = []
        
        if horizon == ForecastHorizon.HOURLY:
            features.extend([
                timestamp.hour / 23.0,  # Hour of day (normalized)
                timestamp.weekday() / 6.0,  # Day of week (normalized)
                math.sin(2 * math.pi * timestamp.hour / 24),  # Cyclic hour
                math.cos(2 * math.pi * timestamp.hour / 24),
            ])
        elif horizon == ForecastHorizon.DAILY:
            features.extend([
                timestamp.weekday() / 6.0,  # Day of week
                timestamp.day / 31.0,  # Day of month
                math.sin(2 * math.pi * timestamp.weekday() / 7),  # Cyclic weekday
                math.cos(2 * math.pi * timestamp.weekday() / 7),
            ])
        elif horizon == ForecastHorizon.WEEKLY:
            features.extend([
                timestamp.isocalendar()[1] / 52.0,  # Week of year
                timestamp.month / 12.0,  # Month
                math.sin(2 * math.pi * timestamp.isocalendar()[1] / 52),  # Cyclic week
                math.cos(2 * math.pi * timestamp.isocalendar()[1] / 52),
            ])
        else:  # Monthly
            features.extend([
                timestamp.month / 12.0,  # Month of year
                timestamp.quarter / 4.0,  # Quarter
                math.sin(2 * math.pi * timestamp.month / 12),  # Cyclic month
                math.cos(2 * math.pi * timestamp.month / 12),
            ])
        
        return features
    
    async def train_model(self, user_id: str, horizon: str) -> bool:
        """Train prediction model for a user and horizon."""
        if user_id not in self.training_data:
            logger.warning(f"No training data for user {user_id}")
            return False
        
        usage_data = self.training_data[user_id]
        X, y = self.prepare_features(usage_data, horizon)
        
        if len(X) == 0 or len(y) == 0:
            logger.warning(f"Insufficient processed data for {user_id} - {horizon}")
            return False
        
        # Initialize user models if not exists
        if user_id not in self.models:
            self.models[user_id] = {}
            self.scalers[user_id] = {}
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers[user_id][horizon] = scaler
        
        # Choose model based on data size and complexity
        if len(X) < 50:
            model = LinearRegression()
        elif len(X) < 200:
            model = Ridge(alpha=1.0)
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        # Train model in executor to prevent blocking
        def train_model_sync():
            model.fit(X_scaled, y)
            return model
        
        loop = asyncio.get_event_loop()
        trained_model = await loop.run_in_executor(None, train_model_sync)
        self.models[user_id][horizon] = trained_model
        
        # Evaluate model performance
        await self._evaluate_model_performance(user_id, horizon, X_scaled, y, trained_model)
        
        logger.info(f"Trained {type(model).__name__} for {user_id} - {horizon} with {len(X)} samples")
        return True
    
    async def _evaluate_model_performance(
        self,
        user_id: str,
        horizon: str,
        X: np.ndarray,
        y: np.ndarray,
        model
    ) -> None:
        """Evaluate and store model performance metrics."""
        # Run predictions in executor for CPU-intensive operation
        def evaluate_sync():
            predictions = model.predict(X)
            
            mae = mean_absolute_error(y, predictions)
            mse = mean_squared_error(y, predictions)
            rmse = math.sqrt(mse)
            
            # MAPE (Mean Absolute Percentage Error)
            mape = np.mean(np.abs((y - predictions) / np.where(y != 0, y, 1))) * 100
            
            # R² score
            ss_res = np.sum((y - predictions) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            return mae, mse, rmse, mape, r2
        
        loop = asyncio.get_event_loop()
        mae, mse, rmse, mape, r2 = await loop.run_in_executor(None, evaluate_sync)
        
        if user_id not in self.model_performance:
            self.model_performance[user_id] = {}
        
        self.model_performance[user_id][horizon] = {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'mape': mape,
            'r2': r2,
            'model_type': type(model).__name__,
            'training_samples': len(X),
            'last_trained': datetime.utcnow().isoformat()
        }
    
    async def predict_costs(
        self,
        user_id: str,
        horizon: str,
        periods_ahead: int = 30,
        confidence_level: float = 0.95
    ) -> Optional[CostForecast]:
        """Predict costs for specified periods ahead."""
        
        # Check cache
        cache_key = f"{user_id}_{horizon}_{periods_ahead}_{confidence_level}"
        if cache_key in self.prediction_cache:
            cached_result, timestamp = self.prediction_cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                return cached_result
        
        # Check if model exists
        if (user_id not in self.models or 
            horizon not in self.models[user_id] or
            user_id not in self.scalers or
            horizon not in self.scalers[user_id]):
            
            # Try to train model
            if not await self.train_model(user_id, horizon):
                return None
        
        model = self.models[user_id][horizon]
        scaler = self.scalers[user_id][horizon]
        
        # Get recent usage data for context
        if user_id not in self.training_data:
            return None
        
        usage_data = self.training_data[user_id]
        if not usage_data:
            return None
        
        # Generate predictions
        predictions = []
        confidence_intervals = []
        timestamps = []
        
        # Get the most recent data point as starting context
        current_time = usage_data[-1]['timestamp']
        
        # Prepare initial context from recent data
        if horizon == ForecastHorizon.HOURLY:
            lookback_periods = 24
            time_delta = timedelta(hours=1)
        elif horizon == ForecastHorizon.DAILY:
            lookback_periods = 14
            time_delta = timedelta(days=1)
        elif horizon == ForecastHorizon.WEEKLY:
            lookback_periods = 8
            time_delta = timedelta(weeks=1)
        else:  # Monthly
            lookback_periods = 6
            time_delta = timedelta(days=30)
        
        # Get recent costs for initial context
        recent_costs = self._get_recent_costs(usage_data, lookback_periods, horizon)
        if len(recent_costs) < lookback_periods:
            # Pad with zeros if insufficient data
            recent_costs = [0.0] * (lookback_periods - len(recent_costs)) + recent_costs
        
        # Generate predictions iteratively
        context = recent_costs.copy()
        
        for i in range(periods_ahead):
            next_time = current_time + (time_delta * (i + 1))
            time_features = self._extract_time_features(next_time, horizon)
            
            # Create feature vector
            feature_vector = context[-lookback_periods:] + time_features
            feature_array = np.array([feature_vector])
            
            # Scale features
            feature_scaled = scaler.transform(feature_array)
            
            # Make prediction
            prediction = model.predict(feature_scaled)[0]
            predictions.append(max(0.0, prediction))  # Ensure non-negative
            
            # Update context with prediction
            context.append(prediction)
            
            # Generate timestamp
            timestamps.append(next_time)
            
            # Calculate confidence interval (simplified)
            # For more sophisticated models, use prediction intervals
            if hasattr(model, 'predict') and horizon in self.model_performance.get(user_id, {}):
                rmse = self.model_performance[user_id][horizon]['rmse']
                z_score = 1.96  # For 95% confidence
                margin = z_score * rmse
                confidence_intervals.append((
                    max(0.0, prediction - margin),
                    prediction + margin
                ))
            else:
                # Default confidence interval
                margin = prediction * 0.2  # 20% margin
                confidence_intervals.append((
                    max(0.0, prediction - margin),
                    prediction + margin
                ))
        
        # Create forecast object
        forecast = CostForecast(
            horizon=horizon,
            predictions=predictions,
            confidence_intervals=confidence_intervals,
            timestamps=timestamps,
            metadata={
                'model_type': type(model).__name__,
                'training_samples': len(usage_data),
                'confidence_level': confidence_level,
                'performance_metrics': self.model_performance.get(user_id, {}).get(horizon, {})
            }
        )
        
        # Cache result
        self.prediction_cache[cache_key] = (forecast, time.time())
        
        logger.info(f"Generated {horizon} forecast for {user_id}: {periods_ahead} periods ahead")
        return forecast
    
    def _get_recent_costs(
        self,
        usage_data: List[Dict[str, Any]],
        lookback_periods: int,
        horizon: str
    ) -> List[float]:
        """Get recent costs grouped by time period."""
        
        # Group data by time period
        if horizon == ForecastHorizon.HOURLY:
            groups = self._group_by_hour(usage_data)
        elif horizon == ForecastHorizon.DAILY:
            groups = self._group_by_day(usage_data)
        elif horizon == ForecastHorizon.WEEKLY:
            groups = self._group_by_week(usage_data)
        else:  # Monthly
            groups = self._group_by_month(usage_data)
        
        # Get costs for each period
        period_costs = []
        for period_key in sorted(groups.keys())[-lookback_periods:]:
            total_cost = sum(record['cost'] for record in groups[period_key])
            period_costs.append(total_cost)
        
        return period_costs
    
    def analyze_usage_patterns(self, user_id: str) -> Dict[str, Any]:
        """Analyze usage patterns for a user."""
        if user_id not in self.training_data:
            return {'error': 'No training data available'}
        
        usage_data = self.training_data[user_id]
        return self.pattern_recognizer.analyze_pattern(usage_data, user_id)
    
    def get_model_performance(self, user_id: str) -> Dict[str, Any]:
        """Get model performance metrics for a user."""
        return self.model_performance.get(user_id, {})
    
    async def retrain_models_if_needed(self, user_id: str) -> None:
        """Retrain models if performance has degraded or data is stale."""
        if user_id not in self.model_performance:
            return
        
        current_time = datetime.utcnow()
        
        for horizon in self.model_performance[user_id]:
            performance = self.model_performance[user_id][horizon]
            
            # Check if model is old (> 7 days)
            last_trained = datetime.fromisoformat(performance['last_trained'])
            if (current_time - last_trained).days > 7:
                logger.info(f"Retraining stale model for {user_id} - {horizon}")
                await self.train_model(user_id, horizon)
            
            # Check if performance is poor (MAPE > 50%)
            elif performance.get('mape', 0) > 50:
                logger.info(f"Retraining poorly performing model for {user_id} - {horizon}")
                await self.train_model(user_id, horizon)
    
    def get_prediction_insights(self, user_id: str, forecast: CostForecast) -> Dict[str, Any]:
        """Generate insights from cost predictions."""
        insights = {
            'total_predicted_cost': forecast.get_total_predicted_cost(),
            'average_period_cost': mean(forecast.predictions),
            'peak_periods': [],
            'low_periods': [],
            'trend_analysis': {},
            'risk_assessment': {}
        }
        
        # Find peak and low periods
        avg_cost = insights['average_period_cost']
        for i, (timestamp, cost) in enumerate(zip(forecast.timestamps, forecast.predictions)):
            if cost > avg_cost * 1.5:
                insights['peak_periods'].append({
                    'timestamp': timestamp.isoformat(),
                    'cost': cost,
                    'deviation_pct': ((cost - avg_cost) / avg_cost) * 100
                })
            elif cost < avg_cost * 0.5:
                insights['low_periods'].append({
                    'timestamp': timestamp.isoformat(),
                    'cost': cost,
                    'deviation_pct': ((avg_cost - cost) / avg_cost) * 100
                })
        
        # Trend analysis
        if len(forecast.predictions) > 1:
            early_avg = mean(forecast.predictions[:len(forecast.predictions)//2])
            late_avg = mean(forecast.predictions[len(forecast.predictions)//2:])
            
            trend_change = ((late_avg - early_avg) / early_avg) * 100 if early_avg > 0 else 0
            
            insights['trend_analysis'] = {
                'direction': 'increasing' if trend_change > 5 else 'decreasing' if trend_change < -5 else 'stable',
                'change_percentage': trend_change,
                'early_period_avg': early_avg,
                'late_period_avg': late_avg
            }
        
        # Risk assessment
        max_prediction = max(forecast.predictions)
        confidence_widths = [high - low for low, high in forecast.confidence_intervals]
        avg_uncertainty = mean(confidence_widths) if confidence_widths else 0
        
        insights['risk_assessment'] = {
            'maximum_predicted_cost': max_prediction,
            'average_uncertainty': avg_uncertainty,
            'uncertainty_ratio': avg_uncertainty / avg_cost if avg_cost > 0 else 0,
            'risk_level': 'high' if avg_uncertainty > avg_cost * 0.5 else 'medium' if avg_uncertainty > avg_cost * 0.2 else 'low'
        }
        
        return insights