"""User preference learning system with feedback loops for AI model selection."""

import asyncio
import json
import math
import statistics
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import uuid4

import structlog
from .task_categorizer import TTRPGTaskType
from ..ai_providers.models import ProviderType

logger = structlog.get_logger(__name__)


class FeedbackType(Enum):
    """Types of user feedback."""
    
    EXPLICIT_RATING = "explicit_rating"        # Direct rating (1-5 stars)
    IMPLICIT_USAGE = "implicit_usage"          # Usage patterns (time spent, regeneration)
    REGENERATION_REQUEST = "regeneration"      # User asked for different response
    ACCEPTANCE = "acceptance"                  # User used the response without changes
    MODIFICATION = "modification"              # User edited the response
    BOOKMARK = "bookmark"                      # User saved/bookmarked response
    SHARE = "share"                           # User shared the response


class PreferenceCategory(Enum):
    """Categories of user preferences."""
    
    RESPONSE_STYLE = "response_style"          # Formal, casual, creative, technical
    DETAIL_LEVEL = "detail_level"             # Brief, moderate, detailed, comprehensive
    CREATIVITY_LEVEL = "creativity_level"     # Conservative, balanced, creative, highly_creative
    COST_SENSITIVITY = "cost_sensitivity"     # Budget-conscious to cost-no-object
    SPEED_PREFERENCE = "speed_preference"     # Speed vs quality trade-off
    PROVIDER_PREFERENCE = "provider_preference"  # Preferred AI providers
    TASK_APPROACH = "task_approach"           # How user prefers different tasks handled


@dataclass
class UserFeedback:
    """Individual piece of user feedback."""
    
    feedback_id: str = field(default_factory=lambda: str(uuid4()))
    user_id: str = ""
    session_id: str = ""
    task_type: TTRPGTaskType = TTRPGTaskType.RULE_LOOKUP
    provider_type: ProviderType = ProviderType.ANTHROPIC
    model_id: str = ""
    
    feedback_type: FeedbackType = FeedbackType.IMPLICIT_USAGE
    rating: Optional[float] = None  # 0.0 - 1.0 normalized score
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Context information
    request_text: str = ""
    response_text: str = ""
    response_time: float = 0.0  # milliseconds
    cost: float = 0.0          # USD
    
    # Behavioral data
    time_to_feedback: float = 0.0     # Time between response and feedback
    interaction_duration: float = 0.0  # How long user spent with response
    follow_up_questions: int = 0       # Number of follow-up questions
    
    # Metadata
    campaign_context: Dict[str, Any] = field(default_factory=dict)
    additional_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UserPreferenceProfile:
    """Comprehensive user preference profile."""
    
    user_id: str
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    
    # Preference scores (0.0 - 1.0)
    response_style_preferences: Dict[str, float] = field(default_factory=lambda: {
        "formal": 0.5, "casual": 0.5, "creative": 0.5, "technical": 0.5
    })
    
    detail_level_preference: float = 0.5  # 0.0 = brief, 1.0 = comprehensive
    creativity_preference: float = 0.5    # 0.0 = conservative, 1.0 = highly creative
    cost_sensitivity: float = 0.5         # 0.0 = cost no object, 1.0 = budget conscious
    speed_vs_quality: float = 0.5         # 0.0 = prioritize speed, 1.0 = prioritize quality
    
    # Provider preferences
    provider_preferences: Dict[ProviderType, float] = field(default_factory=dict)
    model_preferences: Dict[str, float] = field(default_factory=dict)
    
    # Task-specific preferences
    task_preferences: Dict[TTRPGTaskType, Dict[str, float]] = field(default_factory=dict)
    
    # Campaign-specific preferences
    campaign_preferences: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Statistical data
    total_interactions: int = 0
    feedback_count: int = 0
    average_satisfaction: float = 0.5
    confidence_score: float = 0.0  # Based on amount of data
    
    # Behavioral patterns
    typical_session_length: float = 0.0   # minutes
    preferred_interaction_times: List[int] = field(default_factory=list)  # hours of day
    common_task_sequences: Dict[str, int] = field(default_factory=dict)


class PreferenceLearner:
    """System for learning user preferences from feedback and behavior."""
    
    def __init__(self, learning_rate: float = 0.1, decay_factor: float = 0.95):
        self.learning_rate = learning_rate
        self.decay_factor = decay_factor  # For temporal decay of old preferences
        
        # Storage
        self.user_profiles: Dict[str, UserPreferenceProfile] = {}
        self.feedback_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.session_data: Dict[str, Dict[str, Any]] = {}
        
        # Learning parameters
        self.feedback_weights = {
            FeedbackType.EXPLICIT_RATING: 1.0,
            FeedbackType.REGENERATION_REQUEST: -0.8,  # Negative feedback
            FeedbackType.ACCEPTANCE: 0.6,
            FeedbackType.MODIFICATION: -0.3,
            FeedbackType.BOOKMARK: 0.9,
            FeedbackType.SHARE: 0.8,
            FeedbackType.IMPLICIT_USAGE: 0.3
        }
        
        # Confidence thresholds
        self.min_interactions_for_confidence = 10
        self.high_confidence_threshold = 50
        
        # Pattern detection
        self.pattern_cache: Dict[str, Tuple[datetime, Dict[str, Any]]] = {}
        self.task_transition_patterns: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    
    async def record_feedback(self, feedback: UserFeedback) -> None:
        """Record user feedback and update preferences."""
        user_id = feedback.user_id
        
        # Store feedback
        self.feedback_history[user_id].append(feedback)
        
        # Initialize user profile if needed
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = UserPreferenceProfile(user_id=user_id)
        
        profile = self.user_profiles[user_id]
        
        # Update interaction statistics
        profile.total_interactions += 1
        profile.feedback_count += 1
        profile.last_updated = datetime.now()
        
        # Process the feedback
        await self._process_feedback(profile, feedback)
        
        # Update confidence score
        profile.confidence_score = min(1.0, profile.feedback_count / self.high_confidence_threshold)
        
        # Detect patterns
        await self._detect_patterns(profile, feedback)
        
        logger.info(
            "Recorded user feedback",
            user_id=user_id,
            feedback_type=feedback.feedback_type.value,
            task_type=feedback.task_type.value,
            confidence=profile.confidence_score
        )
    
    async def _process_feedback(self, profile: UserPreferenceProfile, feedback: UserFeedback) -> None:
        """Process individual feedback and update preferences."""
        feedback_weight = self.feedback_weights.get(feedback.feedback_type, 0.5)
        
        # Adjust weight based on feedback rating if available
        if feedback.rating is not None:
            feedback_weight = feedback.rating * 2 - 1  # Convert 0-1 to -1 to 1
        
        # Update provider preferences
        if feedback.provider_type not in profile.provider_preferences:
            profile.provider_preferences[feedback.provider_type] = 0.5
        
        current_pref = profile.provider_preferences[feedback.provider_type]
        new_pref = current_pref + self.learning_rate * feedback_weight * (1 - current_pref if feedback_weight > 0 else current_pref)
        profile.provider_preferences[feedback.provider_type] = max(0.0, min(1.0, new_pref))
        
        # Update model preferences
        model_key = f"{feedback.provider_type.value}:{feedback.model_id}"
        if model_key not in profile.model_preferences:
            profile.model_preferences[model_key] = 0.5
        
        current_model_pref = profile.model_preferences[model_key]
        new_model_pref = current_model_pref + self.learning_rate * feedback_weight * (1 - current_model_pref if feedback_weight > 0 else current_model_pref)
        profile.model_preferences[model_key] = max(0.0, min(1.0, new_model_pref))
        
        # Update task-specific preferences
        if feedback.task_type not in profile.task_preferences:
            profile.task_preferences[feedback.task_type] = {
                "preferred_detail": 0.5,
                "preferred_creativity": 0.5,
                "speed_importance": 0.5,
                "cost_sensitivity": 0.5
            }
        
        task_prefs = profile.task_preferences[feedback.task_type]
        
        # Infer preferences from response characteristics
        if feedback.response_text:
            words = feedback.response_text.lower().split()
            response_length = len(words)
            creativity_indicators = len([word for word in words
                                       if word in ["creative", "imaginative", "unique", "innovative", "original"]])
            
            # Update detail preference based on response length and feedback
            if response_length > 200:  # Long response
                detail_signal = feedback_weight * 0.1
                task_prefs["preferred_detail"] += detail_signal
            elif response_length < 50:  # Short response  
                detail_signal = feedback_weight * 0.1
                task_prefs["preferred_detail"] -= detail_signal
            
            # Update creativity preference
            if creativity_indicators > 0:
                creativity_signal = feedback_weight * 0.1
                task_prefs["preferred_creativity"] += creativity_signal
        
        # Update speed vs quality preference based on response time and feedback
        if feedback.response_time > 0:
            if feedback.response_time > 5000 and feedback_weight > 0:  # Slow but good response
                task_prefs["speed_importance"] -= 0.05
            elif feedback.response_time < 1000 and feedback_weight > 0:  # Fast and good response
                task_prefs["speed_importance"] += 0.05
        
        # Update cost sensitivity based on cost and feedback
        if feedback.cost > 0:
            if feedback.cost > 0.1 and feedback_weight < 0:  # Expensive and bad
                task_prefs["cost_sensitivity"] += 0.1
            elif feedback.cost < 0.01 and feedback_weight > 0:  # Cheap and good
                task_prefs["cost_sensitivity"] += 0.05
        
        # Normalize task preferences
        for key in task_prefs:
            task_prefs[key] = max(0.0, min(1.0, task_prefs[key]))
        
        # Update overall satisfaction
        if feedback.rating is not None:
            profile.average_satisfaction = (
                profile.average_satisfaction * 0.9 + feedback.rating * 0.1
            )
        elif feedback_weight != 0:
            # Convert feedback weight to 0-1 satisfaction score
            satisfaction = max(0, min(1, (feedback_weight + 1) / 2))
            profile.average_satisfaction = (
                profile.average_satisfaction * 0.9 + satisfaction * 0.1
            )
    
    async def _detect_patterns(self, profile: UserPreferenceProfile, feedback: UserFeedback) -> None:
        """Detect behavioral patterns from user feedback."""
        user_id = profile.user_id
        
        # Update interaction time patterns
        current_hour = feedback.timestamp.hour
        if current_hour not in profile.preferred_interaction_times:
            profile.preferred_interaction_times.append(current_hour)
        
        # Keep only the most common hours (limit to top 6 hours)
        if len(profile.preferred_interaction_times) > 6:
            hour_counts = defaultdict(int)
            for hour in profile.preferred_interaction_times:
                hour_counts[hour] += 1
            
            top_hours = sorted(hour_counts.items(), key=lambda x: x[1], reverse=True)[:6]
            profile.preferred_interaction_times = [hour for hour, count in top_hours]
        
        # Track task transition patterns
        if user_id in self.session_data:
            session_data = self.session_data[user_id]
            last_task = session_data.get("last_task")
            
            if last_task and last_task != feedback.task_type.value:
                transition_key = f"{last_task}->{feedback.task_type.value}"
                self.task_transition_patterns[user_id][transition_key] += 1
        else:
            self.session_data[user_id] = {}
        
        self.session_data[user_id]["last_task"] = feedback.task_type.value
        self.session_data[user_id]["last_update"] = datetime.now()
        
        # Update session length tracking
        if feedback.interaction_duration > 0:
            if profile.typical_session_length == 0:
                profile.typical_session_length = feedback.interaction_duration / 60  # Convert to minutes
            else:
                # Exponential moving average
                profile.typical_session_length = (
                    profile.typical_session_length * 0.8 + 
                    (feedback.interaction_duration / 60) * 0.2
                )
    
    async def get_user_preferences(
        self,
        user_id: str,
        task_type: Optional[TTRPGTaskType] = None,
        campaign_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get comprehensive user preferences for model selection."""
        if user_id not in self.user_profiles:
            # Return default preferences for new user
            return self._get_default_preferences()
        
        profile = self.user_profiles[user_id]
        preferences = {
            "confidence_score": profile.confidence_score,
            "total_interactions": profile.total_interactions,
            "average_satisfaction": profile.average_satisfaction,
            
            # General preferences
            "detail_level_preference": profile.detail_level_preference,
            "creativity_preference": profile.creativity_preference,
            "cost_sensitivity": profile.cost_sensitivity,
            "speed_vs_quality": profile.speed_vs_quality,
            
            # Provider preferences
            "provider_preferences": {
                provider.value: score 
                for provider, score in profile.provider_preferences.items()
            },
            "model_preferences": dict(profile.model_preferences),
            
            # Behavioral patterns
            "typical_session_length": profile.typical_session_length,
            "preferred_interaction_times": profile.preferred_interaction_times,
            "common_task_sequences": dict(profile.common_task_sequences)
        }
        
        # Add task-specific preferences if requested
        if task_type and task_type in profile.task_preferences:
            preferences["task_specific"] = profile.task_preferences[task_type]
        
        # Add campaign-specific preferences if requested
        if campaign_id and campaign_id in profile.campaign_preferences:
            preferences["campaign_specific"] = profile.campaign_preferences[campaign_id]
        
        return preferences
    
    def _get_default_preferences(self) -> Dict[str, Any]:
        """Get default preferences for new users."""
        return {
            "confidence_score": 0.0,
            "total_interactions": 0,
            "average_satisfaction": 0.5,
            "detail_level_preference": 0.5,
            "creativity_preference": 0.5,
            "cost_sensitivity": 0.5,
            "speed_vs_quality": 0.5,
            "provider_preferences": {},
            "model_preferences": {},
            "typical_session_length": 0.0,
            "preferred_interaction_times": [],
            "common_task_sequences": {}
        }
    
    async def predict_user_satisfaction(
        self,
        user_id: str,
        task_type: TTRPGTaskType,
        provider_type: ProviderType,
        model_id: str,
        expected_response_time: float,
        expected_cost: float,
        expected_creativity_level: float = 0.5
    ) -> float:
        """Predict user satisfaction for a given model choice."""
        if user_id not in self.user_profiles:
            return 0.5  # Neutral prediction for new users
        
        profile = self.user_profiles[user_id]
        satisfaction_score = 0.5  # Base score
        
        # Provider preference
        provider_pref = profile.provider_preferences.get(provider_type, 0.5)
        satisfaction_score += (provider_pref - 0.5) * 0.3
        
        # Model preference
        model_key = f"{provider_type.value}:{model_id}"
        model_pref = profile.model_preferences.get(model_key, 0.5)
        satisfaction_score += (model_pref - 0.5) * 0.2
        
        # Task-specific preferences
        if task_type in profile.task_preferences:
            task_prefs = profile.task_preferences[task_type]
            
            # Speed preference
            speed_pref = task_prefs.get("speed_importance", 0.5)
            if expected_response_time > 5000:  # Slow response
                satisfaction_score -= (1 - speed_pref) * 0.2
            elif expected_response_time < 1000:  # Fast response
                satisfaction_score += speed_pref * 0.1
            
            # Cost preference
            cost_pref = task_prefs.get("cost_sensitivity", 0.5)
            if expected_cost > 0.1:  # Expensive
                satisfaction_score -= cost_pref * 0.2
            elif expected_cost < 0.01:  # Cheap
                satisfaction_score += cost_pref * 0.1
            
            # Creativity preference
            creativity_pref = task_prefs.get("preferred_creativity", 0.5)
            creativity_match = 1 - abs(creativity_pref - expected_creativity_level)
            satisfaction_score += creativity_match * 0.1
        
        # Apply confidence weighting
        satisfaction_score = (
            satisfaction_score * profile.confidence_score + 
            0.5 * (1 - profile.confidence_score)
        )
        
        return max(0.0, min(1.0, satisfaction_score))
    
    async def recommend_improvements(self, user_id: str) -> List[Dict[str, Any]]:
        """Recommend improvements based on user preference patterns."""
        if user_id not in self.user_profiles:
            return []
        
        profile = self.user_profiles[user_id]
        recommendations = []
        
        # Check if user has strong preferences we should leverage
        if profile.confidence_score > 0.5:
            # Provider recommendations
            best_provider = max(profile.provider_preferences.items(), key=lambda x: x[1], default=None)
            if best_provider and best_provider[1] > 0.7:
                recommendations.append({
                    "type": "provider_preference",
                    "message": f"You seem to prefer {best_provider[0].value} - we'll prioritize this provider",
                    "confidence": best_provider[1],
                    "action": "prioritize_provider",
                    "provider": best_provider[0].value
                })
            
            # Speed vs quality recommendations
            if profile.speed_vs_quality > 0.7:
                recommendations.append({
                    "type": "performance_preference", 
                    "message": "You prefer high-quality responses - we'll use more capable models",
                    "confidence": profile.speed_vs_quality,
                    "action": "prioritize_quality"
                })
            elif profile.speed_vs_quality < 0.3:
                recommendations.append({
                    "type": "performance_preference",
                    "message": "You prefer fast responses - we'll use quicker models", 
                    "confidence": 1 - profile.speed_vs_quality,
                    "action": "prioritize_speed"
                })
            
            # Cost sensitivity recommendations
            if profile.cost_sensitivity > 0.7:
                recommendations.append({
                    "type": "cost_preference",
                    "message": "You're budget-conscious - we'll prioritize cost-effective models",
                    "confidence": profile.cost_sensitivity,
                    "action": "prioritize_cost_efficiency"
                })
        
        # Behavioral pattern recommendations
        if profile.typical_session_length > 30:  # Long sessions
            recommendations.append({
                "type": "usage_pattern",
                "message": "You have long gaming sessions - we'll optimize for sustained performance",
                "confidence": 0.8,
                "action": "optimize_for_long_sessions"
            })
        
        # Low satisfaction recommendations
        if profile.average_satisfaction < 0.4 and profile.feedback_count > 5:
            recommendations.append({
                "type": "satisfaction_improvement",
                "message": "We notice you've been less satisfied - let's try different models",
                "confidence": 1 - profile.average_satisfaction,
                "action": "diversify_model_selection"
            })
        
        return recommendations
    
    async def analyze_preference_trends(
        self, 
        user_id: str,
        time_range_days: int = 30
    ) -> Dict[str, Any]:
        """Analyze trends in user preferences over time."""
        if user_id not in self.feedback_history:
            return {"trends": [], "insights": []}
        
        cutoff_time = datetime.now() - timedelta(days=time_range_days)
        recent_feedback = [
            f for f in self.feedback_history[user_id]
            if f.timestamp > cutoff_time
        ]
        
        if len(recent_feedback) < 5:
            return {"trends": [], "insights": ["Not enough recent data for trend analysis"]}
        
        trends = {}
        insights = []
        
        # Analyze satisfaction trend
        satisfaction_scores = []
        timestamps = []
        
        for feedback in recent_feedback:
            if feedback.rating is not None:
                satisfaction_scores.append(feedback.rating)
                timestamps.append(feedback.timestamp)
            elif feedback.feedback_type in self.feedback_weights:
                # Convert feedback to satisfaction score
                weight = self.feedback_weights[feedback.feedback_type]
                satisfaction = max(0, min(1, (weight + 1) / 2))
                satisfaction_scores.append(satisfaction)
                timestamps.append(feedback.timestamp)
        
        if len(satisfaction_scores) >= 3:
            # Simple trend analysis
            recent_avg = statistics.mean(satisfaction_scores[-len(satisfaction_scores)//3:])
            earlier_avg = statistics.mean(satisfaction_scores[:len(satisfaction_scores)//3])
            
            trend = (recent_avg - earlier_avg) / earlier_avg if earlier_avg > 0 else 0
            
            trends["satisfaction_trend"] = {
                "direction": "improving" if trend > 0.1 else "declining" if trend < -0.1 else "stable",
                "change": trend,
                "recent_average": recent_avg,
                "earlier_average": earlier_avg
            }
            
            if trend > 0.1:
                insights.append("Your satisfaction with responses has been improving")
            elif trend < -0.1:
                insights.append("Your satisfaction has been declining - we'll adjust our approach")
        
        # Analyze provider usage trends
        provider_usage = defaultdict(int)
        for feedback in recent_feedback:
            provider_usage[feedback.provider_type.value] += 1
        
        if provider_usage:
            most_used = max(provider_usage.items(), key=lambda x: x[1])
            trends["provider_usage"] = dict(provider_usage)
            insights.append(f"You've been using {most_used[0]} most frequently ({most_used[1]} times)")
        
        # Analyze task type patterns
        task_usage = defaultdict(int)
        for feedback in recent_feedback:
            task_usage[feedback.task_type.value] += 1
        
        if task_usage:
            most_common_task = max(task_usage.items(), key=lambda x: x[1])
            trends["task_patterns"] = dict(task_usage)
            insights.append(f"Your most common task type is {most_common_task[0]}")
        
        return {"trends": trends, "insights": insights}
    
    async def export_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """Export user preferences for backup or analysis."""
        if user_id not in self.user_profiles:
            return {}
        
        profile = self.user_profiles[user_id]
        
        return {
            "user_id": user_id,
            "profile": {
                "created_at": profile.created_at.isoformat(),
                "last_updated": profile.last_updated.isoformat(),
                "response_style_preferences": profile.response_style_preferences,
                "detail_level_preference": profile.detail_level_preference,
                "creativity_preference": profile.creativity_preference,
                "cost_sensitivity": profile.cost_sensitivity,
                "speed_vs_quality": profile.speed_vs_quality,
                "provider_preferences": {k.value: v for k, v in profile.provider_preferences.items()},
                "model_preferences": profile.model_preferences,
                "task_preferences": {
                    k.value: v for k, v in profile.task_preferences.items()
                },
                "campaign_preferences": profile.campaign_preferences,
                "total_interactions": profile.total_interactions,
                "feedback_count": profile.feedback_count,
                "average_satisfaction": profile.average_satisfaction,
                "confidence_score": profile.confidence_score,
                "typical_session_length": profile.typical_session_length,
                "preferred_interaction_times": profile.preferred_interaction_times
            },
            "recent_feedback": [
                {
                    "feedback_id": f.feedback_id,
                    "timestamp": f.timestamp.isoformat(),
                    "task_type": f.task_type.value,
                    "provider_type": f.provider_type.value,
                    "model_id": f.model_id,
                    "feedback_type": f.feedback_type.value,
                    "rating": f.rating
                }
                for f in list(self.feedback_history[user_id])[-20:]  # Last 20 feedback items
            ]
        }
    
    async def cleanup_old_data(self, retention_days: int = 90) -> int:
        """Clean up old preference data."""
        cutoff_time = datetime.now() - timedelta(days=retention_days)
        cleaned_count = 0
        
        # Clean up feedback history
        for user_id in list(self.feedback_history.keys()):
            original_length = len(self.feedback_history[user_id])
            
            # Filter out old feedback
            filtered_feedback = deque(
                (f for f in self.feedback_history[user_id] if f.timestamp > cutoff_time),
                maxlen=1000
            )
            
            self.feedback_history[user_id] = filtered_feedback
            cleaned_count += original_length - len(filtered_feedback)
            
            # Remove user profiles with no recent activity
            if len(filtered_feedback) == 0 and user_id in self.user_profiles:
                del self.user_profiles[user_id]
        
        # Clean up session data
        for user_id in list(self.session_data.keys()):
            last_update = self.session_data[user_id].get("last_update")
            if last_update and last_update < cutoff_time:
                del self.session_data[user_id]
        
        # Clear pattern cache
        self.pattern_cache.clear()
        
        logger.info(
            "Preference data cleanup completed",
            cleaned_feedback=cleaned_count,
            active_users=len(self.user_profiles),
            retention_days=retention_days
        )
        
        return cleaned_count