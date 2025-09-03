"""Per-user usage tracking with persistent storage in JSON and ChromaDB."""

import json
import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path
from uuid import uuid4
from collections import defaultdict
import hashlib
from structlog import get_logger

from .models import ProviderType, UsageRecord, AIRequest, AIResponse
from ..core.database import ChromaDBManager, get_db_manager

logger = get_logger(__name__)


@dataclass
class UserProfile:
    """User profile with usage preferences and limits."""
    
    user_id: str
    username: str
    email: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    is_active: bool = True
    user_tier: str = "free"  # free, premium, enterprise
    preferences: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UserUsageAggregation:
    """Aggregated usage statistics for a user."""
    
    user_id: str
    date: str  # YYYY-MM-DD format
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost: float = 0.0
    avg_latency_ms: float = 0.0
    providers_used: Dict[str, int] = field(default_factory=dict)  # provider -> request count
    models_used: Dict[str, int] = field(default_factory=dict)  # model -> request count
    session_count: int = 0
    unique_sessions: set = field(default_factory=set)
    
    def __post_init__(self):
        """Convert set to list for JSON serialization."""
        if isinstance(self.unique_sessions, set):
            self.unique_sessions = list(self.unique_sessions)


@dataclass
class UserSpendingLimits:
    """Spending limits configuration for a user."""
    
    user_id: str
    daily_limit: Optional[float] = None  # USD
    weekly_limit: Optional[float] = None  # USD
    monthly_limit: Optional[float] = None  # USD
    per_request_limit: Optional[float] = None  # USD
    per_session_limit: Optional[float] = None  # USD
    provider_limits: Dict[str, float] = field(default_factory=dict)  # provider -> limit
    model_limits: Dict[str, float] = field(default_factory=dict)  # model -> limit
    alert_thresholds: List[float] = field(default_factory=lambda: [0.5, 0.8, 0.95])
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class UserUsageAlert:
    """Usage alert for a user."""
    
    alert_id: str = field(default_factory=lambda: str(uuid4()))
    user_id: str = ""
    alert_type: str = ""  # daily_limit, monthly_limit, etc.
    threshold: float = 0.0
    current_usage: float = 0.0
    limit: float = 0.0
    percentage: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False
    resolved: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class UserUsageTracker:
    """Per-user usage tracking with persistent storage."""
    
    def __init__(self, storage_path: Optional[str] = None, use_chromadb: bool = True):
        self.storage_path = Path(storage_path) if storage_path else Path("./data/user_usage")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.use_chromadb = use_chromadb
        if use_chromadb:
            self.db_manager = get_db_manager()
            self._initialize_chromadb_collections()
        
        # In-memory caches for performance
        self.user_profiles: Dict[str, UserProfile] = {}
        self.user_limits: Dict[str, UserSpendingLimits] = {}
        self.daily_usage: Dict[str, Dict[str, UserUsageAggregation]] = {}  # user_id -> date -> aggregation
        self.monthly_usage: Dict[str, Dict[str, UserUsageAggregation]] = {}  # user_id -> month -> aggregation
        self.active_sessions: Dict[str, Dict[str, float]] = {}  # user_id -> session_id -> cost
        self.recent_alerts: Dict[str, List[UserUsageAlert]] = {}  # user_id -> alerts
        
        # Performance tracking
        self._lock = asyncio.Lock()
        self._batch_size = 100
        self._pending_records: List[UsageRecord] = []
        
        # Load initial data
        self._load_user_data()
        
        logger.info("User usage tracker initialized", storage_path=str(self.storage_path))
    
    def _initialize_chromadb_collections(self) -> None:
        """Initialize ChromaDB collections for user data."""
        try:
            # User profiles collection
            if "user_profiles" not in self.db_manager.collections:
                self.db_manager.client.create_collection(
                    name=f"{self.db_manager.client._settings.chroma_collection_prefix}user_profiles",
                    embedding_function=self.db_manager.embedding_function,
                    metadata={"description": "User profiles and preferences"}
                )
                self.db_manager.collections["user_profiles"] = self.db_manager.client.get_collection("user_profiles")
            
            # Usage records collection
            if "user_usage" not in self.db_manager.collections:
                self.db_manager.client.create_collection(
                    name=f"{self.db_manager.client._settings.chroma_collection_prefix}user_usage",
                    embedding_function=self.db_manager.embedding_function,
                    metadata={"description": "User usage records and aggregations"}
                )
                self.db_manager.collections["user_usage"] = self.db_manager.client.get_collection("user_usage")
            
        except Exception as e:
            logger.warning(f"Failed to initialize ChromaDB collections: {e}")
            self.use_chromadb = False
    
    def _load_user_data(self) -> None:
        """Load user data from JSON files."""
        try:
            # Load user profiles
            profiles_file = self.storage_path / "user_profiles.json"
            if profiles_file.exists():
                with open(profiles_file, 'r') as f:
                    profiles_data = json.load(f)
                    for user_data in profiles_data.get("users", []):
                        profile = UserProfile(
                            user_id=user_data["user_id"],
                            username=user_data["username"],
                            email=user_data.get("email"),
                            created_at=datetime.fromisoformat(user_data["created_at"]),
                            updated_at=datetime.fromisoformat(user_data["updated_at"]),
                            is_active=user_data.get("is_active", True),
                            user_tier=user_data.get("user_tier", "free"),
                            preferences=user_data.get("preferences", {}),
                            metadata=user_data.get("metadata", {})
                        )
                        self.user_profiles[profile.user_id] = profile
            
            # Load spending limits
            limits_file = self.storage_path / "spending_limits.json"
            if limits_file.exists():
                with open(limits_file, 'r') as f:
                    limits_data = json.load(f)
                    for limit_data in limits_data.get("limits", []):
                        limits = UserSpendingLimits(
                            user_id=limit_data["user_id"],
                            daily_limit=limit_data.get("daily_limit"),
                            weekly_limit=limit_data.get("weekly_limit"),
                            monthly_limit=limit_data.get("monthly_limit"),
                            per_request_limit=limit_data.get("per_request_limit"),
                            per_session_limit=limit_data.get("per_session_limit"),
                            provider_limits=limit_data.get("provider_limits", {}),
                            model_limits=limit_data.get("model_limits", {}),
                            alert_thresholds=limit_data.get("alert_thresholds", [0.5, 0.8, 0.95]),
                            enabled=limit_data.get("enabled", True),
                            created_at=datetime.fromisoformat(limit_data["created_at"]),
                            updated_at=datetime.fromisoformat(limit_data["updated_at"])
                        )
                        self.user_limits[limits.user_id] = limits
            
            # Load recent usage aggregations
            self._load_usage_aggregations()
            
            logger.info("User data loaded", 
                       profiles=len(self.user_profiles), 
                       limits=len(self.user_limits))
            
        except Exception as e:
            logger.error("Failed to load user data", error=str(e))
    
    def _load_usage_aggregations(self) -> None:
        """Load usage aggregations from JSON files."""
        try:
            # Load daily usage
            daily_file = self.storage_path / "daily_usage.json"
            if daily_file.exists():
                with open(daily_file, 'r') as f:
                    daily_data = json.load(f)
                    for user_id, user_daily in daily_data.items():
                        if user_id not in self.daily_usage:
                            self.daily_usage[user_id] = {}
                        for date, agg_data in user_daily.items():
                            agg = UserUsageAggregation(
                                user_id=user_id,
                                date=date,
                                total_requests=agg_data.get("total_requests", 0),
                                successful_requests=agg_data.get("successful_requests", 0),
                                failed_requests=agg_data.get("failed_requests", 0),
                                total_input_tokens=agg_data.get("total_input_tokens", 0),
                                total_output_tokens=agg_data.get("total_output_tokens", 0),
                                total_cost=agg_data.get("total_cost", 0.0),
                                avg_latency_ms=agg_data.get("avg_latency_ms", 0.0),
                                providers_used=agg_data.get("providers_used", {}),
                                models_used=agg_data.get("models_used", {}),
                                session_count=agg_data.get("session_count", 0),
                                unique_sessions=set(agg_data.get("unique_sessions", []))
                            )
                            self.daily_usage[user_id][date] = agg
            
            # Load monthly usage (similar structure)
            monthly_file = self.storage_path / "monthly_usage.json"
            if monthly_file.exists():
                with open(monthly_file, 'r') as f:
                    monthly_data = json.load(f)
                    for user_id, user_monthly in monthly_data.items():
                        if user_id not in self.monthly_usage:
                            self.monthly_usage[user_id] = {}
                        for month, agg_data in user_monthly.items():
                            agg = UserUsageAggregation(
                                user_id=user_id,
                                date=month,
                                total_requests=agg_data.get("total_requests", 0),
                                successful_requests=agg_data.get("successful_requests", 0),
                                failed_requests=agg_data.get("failed_requests", 0),
                                total_input_tokens=agg_data.get("total_input_tokens", 0),
                                total_output_tokens=agg_data.get("total_output_tokens", 0),
                                total_cost=agg_data.get("total_cost", 0.0),
                                avg_latency_ms=agg_data.get("avg_latency_ms", 0.0),
                                providers_used=agg_data.get("providers_used", {}),
                                models_used=agg_data.get("models_used", {}),
                                session_count=agg_data.get("session_count", 0),
                                unique_sessions=set(agg_data.get("unique_sessions", []))
                            )
                            self.monthly_usage[user_id][month] = agg
                            
        except Exception as e:
            logger.error("Failed to load usage aggregations", error=str(e))
    
    async def create_user_profile(self, user_profile: UserProfile) -> UserProfile:
        """Create a new user profile."""
        async with self._lock:
            # Generate user ID if not provided
            if not user_profile.user_id:
                user_profile.user_id = self._generate_user_id(user_profile.username)
            
            user_profile.created_at = datetime.now()
            user_profile.updated_at = datetime.now()
            
            self.user_profiles[user_profile.user_id] = user_profile
            
            # Create default spending limits
            default_limits = UserSpendingLimits(
                user_id=user_profile.user_id,
                daily_limit=10.0 if user_profile.user_tier == "free" else 100.0,
                monthly_limit=100.0 if user_profile.user_tier == "free" else 1000.0,
                per_request_limit=1.0 if user_profile.user_tier == "free" else 10.0,
            )
            self.user_limits[user_profile.user_id] = default_limits
            
            # Initialize usage tracking
            self.daily_usage[user_profile.user_id] = {}
            self.monthly_usage[user_profile.user_id] = {}
            self.active_sessions[user_profile.user_id] = {}
            self.recent_alerts[user_profile.user_id] = []
            
            # Save to persistent storage
            await self._save_user_profiles()
            await self._save_spending_limits()
            
            if self.use_chromadb:
                await self._save_profile_to_chromadb(user_profile)
            
            logger.info("User profile created", user_id=user_profile.user_id, username=user_profile.username)
            
            return user_profile
    
    def _generate_user_id(self, username: str) -> str:
        """Generate a unique user ID."""
        # Create a hash-based user ID
        hash_input = f"{username}:{datetime.now().isoformat()}"
        user_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:16]
        return f"user_{user_hash}"
    
    async def record_user_usage(
        self,
        user_id: str,
        usage_record: UsageRecord,
        request: Optional[AIRequest] = None,
        response: Optional[AIResponse] = None
    ) -> None:
        """Record usage for a specific user."""
        async with self._lock:
            # Ensure user exists
            if user_id not in self.user_profiles:
                # Create a minimal user profile
                await self.create_user_profile(UserProfile(
                    user_id=user_id,
                    username=f"user_{user_id[:8]}"
                ))
            
            # Update usage record with user information
            usage_record.metadata["user_id"] = user_id
            
            # Update daily aggregation
            today = datetime.now().date().isoformat()
            if user_id not in self.daily_usage:
                self.daily_usage[user_id] = {}
            if today not in self.daily_usage[user_id]:
                self.daily_usage[user_id][today] = UserUsageAggregation(user_id=user_id, date=today)
            
            daily_agg = self.daily_usage[user_id][today]
            daily_agg.total_requests += 1
            if usage_record.success:
                daily_agg.successful_requests += 1
            else:
                daily_agg.failed_requests += 1
            
            daily_agg.total_input_tokens += usage_record.input_tokens
            daily_agg.total_output_tokens += usage_record.output_tokens
            daily_agg.total_cost += usage_record.cost
            
            # Update average latency
            total_latency = daily_agg.avg_latency_ms * (daily_agg.total_requests - 1) + usage_record.latency_ms
            daily_agg.avg_latency_ms = total_latency / daily_agg.total_requests
            
            # Update provider and model usage
            provider_key = usage_record.provider_type.value
            daily_agg.providers_used[provider_key] = daily_agg.providers_used.get(provider_key, 0) + 1
            daily_agg.models_used[usage_record.model] = daily_agg.models_used.get(usage_record.model, 0) + 1
            
            # Track unique sessions
            if usage_record.session_id:
                if usage_record.session_id not in daily_agg.unique_sessions:
                    daily_agg.unique_sessions.add(usage_record.session_id)
                    daily_agg.session_count = len(daily_agg.unique_sessions)
                
                # Update active sessions
                if user_id not in self.active_sessions:
                    self.active_sessions[user_id] = {}
                self.active_sessions[user_id][usage_record.session_id] = (
                    self.active_sessions[user_id].get(usage_record.session_id, 0.0) + usage_record.cost
                )
            
            # Update monthly aggregation
            current_month = datetime.now().strftime("%Y-%m")
            if user_id not in self.monthly_usage:
                self.monthly_usage[user_id] = {}
            if current_month not in self.monthly_usage[user_id]:
                self.monthly_usage[user_id][current_month] = UserUsageAggregation(user_id=user_id, date=current_month)
            
            monthly_agg = self.monthly_usage[user_id][current_month]
            monthly_agg.total_requests += 1
            if usage_record.success:
                monthly_agg.successful_requests += 1
            else:
                monthly_agg.failed_requests += 1
            
            monthly_agg.total_input_tokens += usage_record.input_tokens
            monthly_agg.total_output_tokens += usage_record.output_tokens
            monthly_agg.total_cost += usage_record.cost
            
            # Update monthly averages and counts
            total_latency = monthly_agg.avg_latency_ms * (monthly_agg.total_requests - 1) + usage_record.latency_ms
            monthly_agg.avg_latency_ms = total_latency / monthly_agg.total_requests
            
            monthly_agg.providers_used[provider_key] = monthly_agg.providers_used.get(provider_key, 0) + 1
            monthly_agg.models_used[usage_record.model] = monthly_agg.models_used.get(usage_record.model, 0) + 1
            
            if usage_record.session_id:
                if usage_record.session_id not in monthly_agg.unique_sessions:
                    monthly_agg.unique_sessions.add(usage_record.session_id)
                    monthly_agg.session_count = len(monthly_agg.unique_sessions)
            
            # Check spending limits and generate alerts
            await self._check_spending_limits(user_id)
            
            # Add to batch for persistent storage
            self._pending_records.append(usage_record)
            
            # Flush batch if it's large enough
            if len(self._pending_records) >= self._batch_size:
                await self._flush_pending_records()
            
            logger.debug("User usage recorded", user_id=user_id, cost=usage_record.cost)
    
    async def _check_spending_limits(self, user_id: str) -> None:
        """Check spending limits and generate alerts."""
        if user_id not in self.user_limits:
            return
        
        limits = self.user_limits[user_id]
        if not limits.enabled:
            return
        
        alerts_to_add = []
        
        # Check daily limits
        if limits.daily_limit:
            today = datetime.now().date().isoformat()
            daily_usage = self.get_user_daily_usage(user_id, today)
            usage_percentage = (daily_usage / limits.daily_limit) * 100
            
            for threshold in limits.alert_thresholds:
                if usage_percentage >= threshold * 100:
                    alert = UserUsageAlert(
                        user_id=user_id,
                        alert_type="daily_limit",
                        threshold=threshold,
                        current_usage=daily_usage,
                        limit=limits.daily_limit,
                        percentage=usage_percentage
                    )
                    alerts_to_add.append(alert)
                    break
        
        # Check monthly limits
        if limits.monthly_limit:
            current_month = datetime.now().strftime("%Y-%m")
            monthly_usage = self.get_user_monthly_usage(user_id, current_month)
            usage_percentage = (monthly_usage / limits.monthly_limit) * 100
            
            for threshold in limits.alert_thresholds:
                if usage_percentage >= threshold * 100:
                    alert = UserUsageAlert(
                        user_id=user_id,
                        alert_type="monthly_limit",
                        threshold=threshold,
                        current_usage=monthly_usage,
                        limit=limits.monthly_limit,
                        percentage=usage_percentage
                    )
                    alerts_to_add.append(alert)
                    break
        
        # Add new alerts
        if alerts_to_add:
            if user_id not in self.recent_alerts:
                self.recent_alerts[user_id] = []
            self.recent_alerts[user_id].extend(alerts_to_add)
            
            # Limit alert history
            if len(self.recent_alerts[user_id]) > 100:
                self.recent_alerts[user_id] = self.recent_alerts[user_id][-50:]
            
            logger.info("Spending limit alerts generated", user_id=user_id, count=len(alerts_to_add))
    
    def get_user_daily_usage(self, user_id: str, date: Optional[str] = None) -> float:
        """Get total daily usage cost for a user."""
        if date is None:
            date = datetime.now().date().isoformat()
        
        if user_id in self.daily_usage and date in self.daily_usage[user_id]:
            return self.daily_usage[user_id][date].total_cost
        return 0.0
    
    def get_user_monthly_usage(self, user_id: str, month: Optional[str] = None) -> float:
        """Get total monthly usage cost for a user."""
        if month is None:
            month = datetime.now().strftime("%Y-%m")
        
        if user_id in self.monthly_usage and month in self.monthly_usage[user_id]:
            return self.monthly_usage[user_id][month].total_cost
        return 0.0
    
    def get_user_session_usage(self, user_id: str, session_id: str) -> float:
        """Get usage cost for a specific session."""
        if user_id in self.active_sessions and session_id in self.active_sessions[user_id]:
            return self.active_sessions[user_id][session_id]
        return 0.0
    
    async def _save_user_profiles(self) -> None:
        """Save user profiles to JSON file."""
        try:
            profiles_data = {
                "last_updated": datetime.now().isoformat(),
                "users": []
            }
            
            for profile in self.user_profiles.values():
                user_data = {
                    "user_id": profile.user_id,
                    "username": profile.username,
                    "email": profile.email,
                    "created_at": profile.created_at.isoformat(),
                    "updated_at": profile.updated_at.isoformat(),
                    "is_active": profile.is_active,
                    "user_tier": profile.user_tier,
                    "preferences": profile.preferences,
                    "metadata": profile.metadata
                }
                profiles_data["users"].append(user_data)
            
            profiles_file = self.storage_path / "user_profiles.json"
            with open(profiles_file, 'w') as f:
                json.dump(profiles_data, f, indent=2)
                
        except Exception as e:
            logger.error("Failed to save user profiles", error=str(e))
    
    async def _save_spending_limits(self) -> None:
        """Save spending limits to JSON file."""
        try:
            limits_data = {
                "last_updated": datetime.now().isoformat(),
                "limits": []
            }
            
            for limits in self.user_limits.values():
                limit_data = {
                    "user_id": limits.user_id,
                    "daily_limit": limits.daily_limit,
                    "weekly_limit": limits.weekly_limit,
                    "monthly_limit": limits.monthly_limit,
                    "per_request_limit": limits.per_request_limit,
                    "per_session_limit": limits.per_session_limit,
                    "provider_limits": limits.provider_limits,
                    "model_limits": limits.model_limits,
                    "alert_thresholds": limits.alert_thresholds,
                    "enabled": limits.enabled,
                    "created_at": limits.created_at.isoformat(),
                    "updated_at": limits.updated_at.isoformat()
                }
                limits_data["limits"].append(limit_data)
            
            limits_file = self.storage_path / "spending_limits.json"
            with open(limits_file, 'w') as f:
                json.dump(limits_data, f, indent=2)
                
        except Exception as e:
            logger.error("Failed to save spending limits", error=str(e))
    
    async def _flush_pending_records(self) -> None:
        """Flush pending usage records to persistent storage."""
        if not self._pending_records:
            return
        
        try:
            # Save to JSON
            await self._save_usage_aggregations()
            
            # Save to ChromaDB if enabled
            if self.use_chromadb:
                await self._save_records_to_chromadb(self._pending_records)
            
            # Clear pending records
            self._pending_records.clear()
            
        except Exception as e:
            logger.error("Failed to flush pending records", error=str(e))
    
    async def _save_usage_aggregations(self) -> None:
        """Save usage aggregations to JSON files."""
        try:
            # Save daily usage
            daily_data = {}
            for user_id, user_daily in self.daily_usage.items():
                daily_data[user_id] = {}
                for date, agg in user_daily.items():
                    daily_data[user_id][date] = {
                        "total_requests": agg.total_requests,
                        "successful_requests": agg.successful_requests,
                        "failed_requests": agg.failed_requests,
                        "total_input_tokens": agg.total_input_tokens,
                        "total_output_tokens": agg.total_output_tokens,
                        "total_cost": agg.total_cost,
                        "avg_latency_ms": agg.avg_latency_ms,
                        "providers_used": agg.providers_used,
                        "models_used": agg.models_used,
                        "session_count": agg.session_count,
                        "unique_sessions": list(agg.unique_sessions) if isinstance(agg.unique_sessions, set) else agg.unique_sessions
                    }
            
            daily_file = self.storage_path / "daily_usage.json"
            with open(daily_file, 'w') as f:
                json.dump(daily_data, f, indent=2)
            
            # Save monthly usage (similar structure)
            monthly_data = {}
            for user_id, user_monthly in self.monthly_usage.items():
                monthly_data[user_id] = {}
                for month, agg in user_monthly.items():
                    monthly_data[user_id][month] = {
                        "total_requests": agg.total_requests,
                        "successful_requests": agg.successful_requests,
                        "failed_requests": agg.failed_requests,
                        "total_input_tokens": agg.total_input_tokens,
                        "total_output_tokens": agg.total_output_tokens,
                        "total_cost": agg.total_cost,
                        "avg_latency_ms": agg.avg_latency_ms,
                        "providers_used": agg.providers_used,
                        "models_used": agg.models_used,
                        "session_count": agg.session_count,
                        "unique_sessions": list(agg.unique_sessions) if isinstance(agg.unique_sessions, set) else agg.unique_sessions
                    }
            
            monthly_file = self.storage_path / "monthly_usage.json"
            with open(monthly_file, 'w') as f:
                json.dump(monthly_data, f, indent=2)
                
        except Exception as e:
            logger.error("Failed to save usage aggregations", error=str(e))
    
    async def _save_profile_to_chromadb(self, profile: UserProfile) -> None:
        """Save user profile to ChromaDB."""
        try:
            if "user_profiles" not in self.db_manager.collections:
                return
            
            collection = self.db_manager.collections["user_profiles"]
            
            # Create document content for embedding
            content = f"User: {profile.username} ({profile.user_tier}) - {profile.email or 'No email'}"
            
            metadata = {
                "user_id": profile.user_id,
                "username": profile.username,
                "email": profile.email or "",
                "user_tier": profile.user_tier,
                "is_active": profile.is_active,
                "created_at": profile.created_at.isoformat(),
                "updated_at": profile.updated_at.isoformat(),
            }
            
            collection.upsert(
                ids=[profile.user_id],
                documents=[content],
                metadatas=[metadata]
            )
            
        except Exception as e:
            logger.error("Failed to save profile to ChromaDB", user_id=profile.user_id, error=str(e))
    
    async def _save_records_to_chromadb(self, records: List[UsageRecord]) -> None:
        """Save usage records to ChromaDB."""
        try:
            if "user_usage" not in self.db_manager.collections or not records:
                return
            
            collection = self.db_manager.collections["user_usage"]
            
            ids = []
            documents = []
            metadatas = []
            
            for record in records:
                # Create document content
                content = (
                    f"Usage by {record.metadata.get('user_id', 'unknown')} "
                    f"using {record.provider_type.value} {record.model} "
                    f"- {record.input_tokens + record.output_tokens} tokens "
                    f"at ${record.cost:.4f}"
                )
                
                metadata = {
                    "user_id": record.metadata.get("user_id", ""),
                    "request_id": record.request_id,
                    "session_id": record.session_id or "",
                    "provider_type": record.provider_type.value,
                    "model": record.model,
                    "input_tokens": record.input_tokens,
                    "output_tokens": record.output_tokens,
                    "cost": record.cost,
                    "success": record.success,
                    "timestamp": record.timestamp.isoformat(),
                }
                
                ids.append(f"usage_{record.request_id}")
                documents.append(content)
                metadatas.append(metadata)
            
            # Batch insert
            collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
            
        except Exception as e:
            logger.error("Failed to save records to ChromaDB", count=len(records), error=str(e))
    
    def get_user_usage_summary(self, user_id: str, days: int = 30) -> Dict[str, Any]:
        """Get comprehensive usage summary for a user."""
        summary = {
            "user_id": user_id,
            "profile": None,
            "limits": None,
            "daily_usage": {},
            "monthly_usage": {},
            "recent_alerts": [],
            "active_sessions": {},
            "statistics": {
                "total_requests": 0,
                "total_cost": 0.0,
                "avg_daily_cost": 0.0,
                "most_used_provider": None,
                "most_used_model": None,
            }
        }
        
        # Get user profile
        if user_id in self.user_profiles:
            summary["profile"] = asdict(self.user_profiles[user_id])
        
        # Get spending limits
        if user_id in self.user_limits:
            summary["limits"] = asdict(self.user_limits[user_id])
        
        # Get daily usage for the last N days
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days)
        
        total_requests = 0
        total_cost = 0.0
        provider_counts = defaultdict(int)
        model_counts = defaultdict(int)
        
        if user_id in self.daily_usage:
            for date_str, agg in self.daily_usage[user_id].items():
                date = datetime.fromisoformat(date_str).date()
                if start_date <= date <= end_date:
                    summary["daily_usage"][date_str] = asdict(agg)
                    total_requests += agg.total_requests
                    total_cost += agg.total_cost
                    
                    for provider, count in agg.providers_used.items():
                        provider_counts[provider] += count
                    
                    for model, count in agg.models_used.items():
                        model_counts[model] += count
        
        # Get monthly usage
        if user_id in self.monthly_usage:
            summary["monthly_usage"] = {
                month: asdict(agg) for month, agg in self.monthly_usage[user_id].items()
            }
        
        # Get recent alerts
        if user_id in self.recent_alerts:
            summary["recent_alerts"] = [asdict(alert) for alert in self.recent_alerts[user_id][-10:]]
        
        # Get active sessions
        if user_id in self.active_sessions:
            summary["active_sessions"] = dict(self.active_sessions[user_id])
        
        # Calculate statistics
        summary["statistics"]["total_requests"] = total_requests
        summary["statistics"]["total_cost"] = total_cost
        summary["statistics"]["avg_daily_cost"] = total_cost / days if days > 0 else 0.0
        summary["statistics"]["most_used_provider"] = max(provider_counts.items(), key=lambda x: x[1])[0] if provider_counts else None
        summary["statistics"]["most_used_model"] = max(model_counts.items(), key=lambda x: x[1])[0] if model_counts else None
        
        return summary
    
    async def cleanup(self) -> None:
        """Clean up resources and flush pending data."""
        async with self._lock:
            await self._flush_pending_records()
            await self._save_user_profiles()
            await self._save_spending_limits()
        
        logger.info("User usage tracker cleanup completed")