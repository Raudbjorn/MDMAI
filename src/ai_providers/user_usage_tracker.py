"""Scalable per-user usage tracking with partitioned storage and streaming processing."""

import json
import asyncio
import sqlite3
import aiosqlite
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union, Set, AsyncIterator
from dataclasses import dataclass, field, asdict
from pathlib import Path
from uuid import uuid4
from collections import defaultdict
import hashlib
import aiofiles
from structlog import get_logger
import os
from threading import RLock
from functools import lru_cache

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
        
        # Scalable storage paths - partitioned by user and time period
        self.partitioned_storage = self.storage_path / "partitioned"
        self.partitioned_storage.mkdir(parents=True, exist_ok=True)
        
        # SQLite database for aggregations and fast querying
        self.db_path = self.storage_path / "usage_aggregations.db"
        self._db_initialized = False
        
        # Limited in-memory caches with LRU eviction
        self._cache_size = 1000  # Maximum items in cache
        self.user_profiles: Dict[str, UserProfile] = {}
        self.user_limits: Dict[str, UserSpendingLimits] = {}
        
        # Time-bounded caches (only keep recent data in memory)
        self._cache_retention_days = 7
        self._daily_usage_cache: Dict[str, Dict[str, UserUsageAggregation]] = {}
        self._monthly_usage_cache: Dict[str, Dict[str, UserUsageAggregation]] = {}
        self.active_sessions: Dict[str, Dict[str, float]] = {}  # user_id -> session_id -> cost
        self.recent_alerts: Dict[str, List[UserUsageAlert]] = {}  # user_id -> alerts
        
        # Cache management
        self._last_cache_cleanup = datetime.now()
        self._cache_cleanup_interval = timedelta(hours=1)
        
        # Thread safety for cache operations
        self._cache_lock = RLock()
        
        # Performance tracking
        self._lock = asyncio.Lock()
        self._batch_size = 100
        self._pending_records: List[UsageRecord] = []
        
        # Initialize database and load initial data
        asyncio.create_task(self._initialize_database())
        asyncio.create_task(self._load_user_data())
        
        logger.info("User usage tracker initialized", storage_path=str(self.storage_path))
    
    async def _initialize_database(self) -> None:
        """Initialize SQLite database for efficient aggregation storage and querying."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Create tables for efficient storage and querying
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS daily_aggregations (
                        user_id TEXT NOT NULL,
                        date TEXT NOT NULL,
                        total_requests INTEGER DEFAULT 0,
                        successful_requests INTEGER DEFAULT 0,
                        failed_requests INTEGER DEFAULT 0,
                        total_input_tokens INTEGER DEFAULT 0,
                        total_output_tokens INTEGER DEFAULT 0,
                        total_cost REAL DEFAULT 0.0,
                        avg_latency_ms REAL DEFAULT 0.0,
                        session_count INTEGER DEFAULT 0,
                        providers_used TEXT DEFAULT '{}',
                        models_used TEXT DEFAULT '{}',
                        unique_sessions TEXT DEFAULT '[]',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (user_id, date)
                    )
                """)
                
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS monthly_aggregations (
                        user_id TEXT NOT NULL,
                        month TEXT NOT NULL,
                        total_requests INTEGER DEFAULT 0,
                        successful_requests INTEGER DEFAULT 0,
                        failed_requests INTEGER DEFAULT 0,
                        total_input_tokens INTEGER DEFAULT 0,
                        total_output_tokens INTEGER DEFAULT 0,
                        total_cost REAL DEFAULT 0.0,
                        avg_latency_ms REAL DEFAULT 0.0,
                        session_count INTEGER DEFAULT 0,
                        providers_used TEXT DEFAULT '{}',
                        models_used TEXT DEFAULT '{}',
                        unique_sessions TEXT DEFAULT '[]',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (user_id, month)
                    )
                """)
                
                # Create indexes for efficient querying
                await db.execute("CREATE INDEX IF NOT EXISTS idx_daily_user_date ON daily_aggregations(user_id, date)")
                await db.execute("CREATE INDEX IF NOT EXISTS idx_daily_date ON daily_aggregations(date)")
                await db.execute("CREATE INDEX IF NOT EXISTS idx_daily_cost ON daily_aggregations(total_cost)")
                await db.execute("CREATE INDEX IF NOT EXISTS idx_monthly_user_month ON monthly_aggregations(user_id, month)")
                await db.execute("CREATE INDEX IF NOT EXISTS idx_monthly_month ON monthly_aggregations(month)")
                await db.execute("CREATE INDEX IF NOT EXISTS idx_monthly_cost ON monthly_aggregations(total_cost)")
                
                await db.commit()
            
            self._db_initialized = True
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            self._db_initialized = False
    
    def _get_user_partition_path(self, user_id: str) -> Path:
        """Get partitioned storage path for a user."""
        # Create a hash-based partition to distribute users evenly
        user_hash = hashlib.md5(user_id.encode()).hexdigest()[:2]
        partition_path = self.partitioned_storage / user_hash / user_id
        partition_path.mkdir(parents=True, exist_ok=True)
        return partition_path
    
    def _get_time_partition_path(self, user_id: str, date: str) -> Path:
        """Get time-partitioned storage path for user data."""
        user_path = self._get_user_partition_path(user_id)
        year_month = date[:7]  # Extract YYYY-MM from YYYY-MM-DD
        time_path = user_path / year_month
        time_path.mkdir(parents=True, exist_ok=True)
        return time_path
    
    async def _cleanup_cache(self) -> None:
        """Clean up old cached data to prevent memory leaks."""
        if datetime.now() - self._last_cache_cleanup < self._cache_cleanup_interval:
            return
        
        with self._cache_lock:
            cutoff_date = (datetime.now() - timedelta(days=self._cache_retention_days)).date()
            cutoff_str = cutoff_date.isoformat()
            
            # Clean up daily usage cache
            for user_id in list(self._daily_usage_cache.keys()):
                user_daily = self._daily_usage_cache[user_id]
                old_dates = [date for date in user_daily.keys() if date < cutoff_str]
                for old_date in old_dates:
                    del user_daily[old_date]
                
                # Remove user from cache if no recent data
                if not user_daily:
                    del self._daily_usage_cache[user_id]
            
            # Clean up monthly usage cache (keep last 12 months)
            cutoff_month = (datetime.now() - timedelta(days=365)).strftime('%Y-%m')
            for user_id in list(self._monthly_usage_cache.keys()):
                user_monthly = self._monthly_usage_cache[user_id]
                old_months = [month for month in user_monthly.keys() if month < cutoff_month]
                for old_month in old_months:
                    del user_monthly[old_month]
                
                if not user_monthly:
                    del self._monthly_usage_cache[user_id]
            
            self._last_cache_cleanup = datetime.now()
        
        logger.debug("Cache cleanup completed")
    
    async def _load_daily_aggregation(self, user_id: str, date: str) -> Optional[UserUsageAggregation]:
        """Load daily aggregation from database or partitioned files."""
        try:
            # First try database
            if self._db_initialized:
                async with aiosqlite.connect(self.db_path) as db:
                    async with db.execute(
                        "SELECT * FROM daily_aggregations WHERE user_id = ? AND date = ?",
                        (user_id, date)
                    ) as cursor:
                        row = await cursor.fetchone()
                        if row:
                            return UserUsageAggregation(
                                user_id=row[0],
                                date=row[1],
                                total_requests=row[2],
                                successful_requests=row[3],
                                failed_requests=row[4],
                                total_input_tokens=row[5],
                                total_output_tokens=row[6],
                                total_cost=row[7],
                                avg_latency_ms=row[8],
                                session_count=row[9],
                                providers_used=json.loads(row[10]),
                                models_used=json.loads(row[11]),
                                unique_sessions=set(json.loads(row[12]))
                            )
            
            # Fallback to partitioned file
            partition_path = self._get_time_partition_path(user_id, date)
            daily_file = partition_path / f"daily_{date}.json"
            
            if daily_file.exists():
                async with aiofiles.open(daily_file, 'r') as f:
                    data = json.loads(await f.read())
                    return UserUsageAggregation(
                        user_id=data["user_id"],
                        date=data["date"],
                        total_requests=data.get("total_requests", 0),
                        successful_requests=data.get("successful_requests", 0),
                        failed_requests=data.get("failed_requests", 0),
                        total_input_tokens=data.get("total_input_tokens", 0),
                        total_output_tokens=data.get("total_output_tokens", 0),
                        total_cost=data.get("total_cost", 0.0),
                        avg_latency_ms=data.get("avg_latency_ms", 0.0),
                        session_count=data.get("session_count", 0),
                        providers_used=data.get("providers_used", {}),
                        models_used=data.get("models_used", {}),
                        unique_sessions=set(data.get("unique_sessions", []))
                    )
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to load daily aggregation for {user_id} on {date}: {e}")
            return None
    
    async def _load_monthly_aggregation(self, user_id: str, month: str) -> Optional[UserUsageAggregation]:
        """Load monthly aggregation from database or partitioned files."""
        try:
            # First try database
            if self._db_initialized:
                async with aiosqlite.connect(self.db_path) as db:
                    async with db.execute(
                        "SELECT * FROM monthly_aggregations WHERE user_id = ? AND month = ?",
                        (user_id, month)
                    ) as cursor:
                        row = await cursor.fetchone()
                        if row:
                            return UserUsageAggregation(
                                user_id=row[0],
                                date=row[1],
                                total_requests=row[2],
                                successful_requests=row[3],
                                failed_requests=row[4],
                                total_input_tokens=row[5],
                                total_output_tokens=row[6],
                                total_cost=row[7],
                                avg_latency_ms=row[8],
                                session_count=row[9],
                                providers_used=json.loads(row[10]),
                                models_used=json.loads(row[11]),
                                unique_sessions=set(json.loads(row[12]))
                            )
            
            # Fallback to partitioned file
            user_path = self._get_user_partition_path(user_id)
            monthly_file = user_path / f"monthly_{month}.json"
            
            if monthly_file.exists():
                async with aiofiles.open(monthly_file, 'r') as f:
                    data = json.loads(await f.read())
                    return UserUsageAggregation(
                        user_id=data["user_id"],
                        date=data["month"],
                        total_requests=data.get("total_requests", 0),
                        successful_requests=data.get("successful_requests", 0),
                        failed_requests=data.get("failed_requests", 0),
                        total_input_tokens=data.get("total_input_tokens", 0),
                        total_output_tokens=data.get("total_output_tokens", 0),
                        total_cost=data.get("total_cost", 0.0),
                        avg_latency_ms=data.get("avg_latency_ms", 0.0),
                        session_count=data.get("session_count", 0),
                        providers_used=data.get("providers_used", {}),
                        models_used=data.get("models_used", {}),
                        unique_sessions=set(data.get("unique_sessions", []))
                    )
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to load monthly aggregation for {user_id} on {month}: {e}")
            return None
    
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
    
    async def _load_user_data(self) -> None:
        """Load user data from JSON files."""
        try:
            # Load user profiles - support both .json and .jsonl formats
            profiles_file = self.storage_path / "user_profiles.json"
            profiles_jsonl_file = self.storage_path / "user_profiles.jsonl"
            
            # Try loading from .jsonl first (newer format), then fall back to .json
            if profiles_jsonl_file.exists():
                await self._load_profiles_from_jsonl(profiles_jsonl_file)
            elif profiles_file.exists():
                profiles_data = await asyncio.to_thread(
                    self._read_json_file,
                    profiles_file
                )
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
            
            # Load spending limits - support both partitioned and legacy format
            limits_dir = self.storage_path / "spending_limits"
            legacy_limits_file = self.storage_path / "spending_limits.json"
            
            # First try loading from partitioned storage (scalable format)
            if limits_dir.exists() and limits_dir.is_dir():
                # Load index if it exists for faster lookups
                index_file = limits_dir / "index.json"
                user_files = []
                
                if index_file.exists():
                    try:
                        index_data = await asyncio.to_thread(
                            self._read_json_file,
                            index_file
                        )
                        # Load specific user files from index
                        import hashlib
                        for user_id, user_hash in index_data.get("user_index", {}).items():
                            user_file = limits_dir / f"user_{user_hash}.json"
                            if user_file.exists():
                                user_files.append(user_file)
                    except Exception as e:
                        logger.warning(f"Failed to load index file, scanning directory: {e}")
                        # Fall back to scanning directory
                        user_files = list(limits_dir.glob("user_*.json"))
                else:
                    # No index, scan directory
                    user_files = list(limits_dir.glob("user_*.json"))
                
                # Load each user's limits
                for user_file in user_files:
                    try:
                        limit_data = await asyncio.to_thread(
                            self._read_json_file,
                            user_file
                        )
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
                        self.user_limits[limit_data["user_id"]] = limits
                    except Exception as e:
                        logger.warning(f"Failed to load user limits from {user_file}: {e}")
                        
            # Fall back to legacy single file if partitioned storage doesn't exist
            elif legacy_limits_file.exists():
                limits_data = await asyncio.to_thread(
                    self._read_json_file,
                    legacy_limits_file
                )
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
                
                # Migrate to partitioned storage on next save
                logger.info("Loaded spending limits from legacy format, will migrate on next save")
            
            # Load recent usage aggregations
            await self._load_usage_aggregations()
            
            logger.info("User data loaded", 
                       profiles=len(self.user_profiles), 
                       limits=len(self.user_limits))
            
        except Exception as e:
            logger.error("Failed to load user data", error=str(e))
    
    async def _load_profiles_from_jsonl(self, profiles_file: Path) -> None:
        """Load user profiles from JSON Lines format with deduplication."""
        try:
            profiles_data = {}  # user_id -> (profile, updated_at)
            
            async with aiofiles.open(profiles_file, 'r') as f:
                async for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        user_data = json.loads(line)
                        user_id = user_data.get("user_id")
                        
                        if not user_id:
                            continue
                        
                        updated_at = datetime.fromisoformat(user_data["updated_at"])
                        
                        # Keep the most recent profile for each user
                        if user_id not in profiles_data or updated_at > profiles_data[user_id][1]:
                            profile = UserProfile(
                                user_id=user_id,
                                username=user_data["username"],
                                email=user_data.get("email"),
                                created_at=datetime.fromisoformat(user_data["created_at"]),
                                updated_at=updated_at,
                                is_active=user_data.get("is_active", True),
                                user_tier=user_data.get("user_tier", "free"),
                                preferences=user_data.get("preferences", {}),
                                metadata=user_data.get("metadata", {})
                            )
                            profiles_data[user_id] = (profile, updated_at)
                    
                    except (json.JSONDecodeError, KeyError, ValueError) as parse_error:
                        logger.warning("Failed to parse user profile line", 
                                     error=str(parse_error), line=line[:100])
                        continue
            
            # Store the deduplicated profiles
            for user_id, (profile, _) in profiles_data.items():
                self.user_profiles[user_id] = profile
        
        except Exception as e:
            logger.error("Failed to load profiles from JSONL", error=str(e))
    
    async def _load_usage_aggregations(self) -> None:
        """Load usage aggregations from database and legacy JSON files for backward compatibility."""
        try:
            # Load recent data from database if available
            if self._db_initialized:
                await self._load_recent_aggregations_from_db()
            
            # Load legacy data from monolithic JSON files for migration
            await self._migrate_legacy_aggregations()
                            
        except Exception as e:
            logger.error("Failed to load usage aggregations", error=str(e))
    
    async def _load_recent_aggregations_from_db(self) -> None:
        """Load recent aggregations from database into cache."""
        try:
            cutoff_date = (datetime.now() - timedelta(days=self._cache_retention_days)).date().isoformat()
            
            async with aiosqlite.connect(self.db_path) as db:
                # Load recent daily aggregations
                async with db.execute(
                    "SELECT * FROM daily_aggregations WHERE date >= ? ORDER BY date DESC LIMIT 1000",
                    (cutoff_date,)
                ) as cursor:
                    async for row in cursor:
                        user_id = row[0]
                        date = row[1]
                        
                        if user_id not in self._daily_usage_cache:
                            self._daily_usage_cache[user_id] = {}
                        
                        self._daily_usage_cache[user_id][date] = UserUsageAggregation(
                            user_id=user_id,
                            date=date,
                            total_requests=row[2],
                            successful_requests=row[3],
                            failed_requests=row[4],
                            total_input_tokens=row[5],
                            total_output_tokens=row[6],
                            total_cost=row[7],
                            avg_latency_ms=row[8],
                            session_count=row[9],
                            providers_used=json.loads(row[10]),
                            models_used=json.loads(row[11]),
                            unique_sessions=set(json.loads(row[12]))
                        )
                
                # Load recent monthly aggregations
                cutoff_month = (datetime.now() - timedelta(days=365)).strftime('%Y-%m')
                async with db.execute(
                    "SELECT * FROM monthly_aggregations WHERE month >= ? ORDER BY month DESC LIMIT 200",
                    (cutoff_month,)
                ) as cursor:
                    async for row in cursor:
                        user_id = row[0]
                        month = row[1]
                        
                        if user_id not in self._monthly_usage_cache:
                            self._monthly_usage_cache[user_id] = {}
                        
                        self._monthly_usage_cache[user_id][month] = UserUsageAggregation(
                            user_id=user_id,
                            date=month,
                            total_requests=row[2],
                            successful_requests=row[3],
                            failed_requests=row[4],
                            total_input_tokens=row[5],
                            total_output_tokens=row[6],
                            total_cost=row[7],
                            avg_latency_ms=row[8],
                            session_count=row[9],
                            providers_used=json.loads(row[10]),
                            models_used=json.loads(row[11]),
                            unique_sessions=set(json.loads(row[12]))
                        )
        
        except Exception as e:
            logger.warning(f"Failed to load recent aggregations from database: {e}")
    
    async def _migrate_legacy_aggregations(self) -> None:
        """Migrate data from legacy monolithic JSON files to partitioned storage."""
        try:
            # Migrate daily usage
            daily_file = self.storage_path / "daily_usage.json"
            if daily_file.exists():
                logger.info("Migrating legacy daily usage data to scalable format")
                # Check file size and use streaming for large files
                file_size = daily_file.stat().st_size
                if file_size > 50 * 1024 * 1024:  # 50MB threshold
                    logger.info(f"Large legacy file detected ({file_size / (1024*1024):.1f}MB), using streaming migration")
                    await self._migrate_large_json_file(daily_file, "daily")
                    return
                
                daily_data = await asyncio.to_thread(self._read_json_file, daily_file)
                
                for user_id, user_daily in daily_data.items():
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
                        
                        # Save to new partitioned format
                        if user_id not in self._daily_usage_cache:
                            self._daily_usage_cache[user_id] = {}
                        self._daily_usage_cache[user_id][date] = agg
                
                # Backup the legacy file and remove it
                backup_file = self.storage_path / "daily_usage.json.legacy_backup"
                await asyncio.to_thread(daily_file.rename, backup_file)
                logger.info("Legacy daily usage file backed up and migrated")
            
            # Migrate monthly usage
            monthly_file = self.storage_path / "monthly_usage.json"
            if monthly_file.exists():
                logger.info("Migrating legacy monthly usage data to scalable format")
                # Check file size and use streaming for large files
                file_size = monthly_file.stat().st_size
                if file_size > 50 * 1024 * 1024:  # 50MB threshold
                    logger.info(f"Large legacy file detected ({file_size / (1024*1024):.1f}MB), using streaming migration")
                    await self._migrate_large_json_file(monthly_file, "monthly")
                    return
                
                monthly_data = await asyncio.to_thread(self._read_json_file, monthly_file)
                
                for user_id, user_monthly in monthly_data.items():
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
                        
                        # Save to new partitioned format
                        if user_id not in self._monthly_usage_cache:
                            self._monthly_usage_cache[user_id] = {}
                        self._monthly_usage_cache[user_id][month] = agg
                
                # Backup the legacy file and remove it
                backup_file = self.storage_path / "monthly_usage.json.legacy_backup"
                await asyncio.to_thread(monthly_file.rename, backup_file)
                logger.info("Legacy monthly usage file backed up and migrated")
                
        except Exception as e:
            logger.warning(f"Failed to migrate legacy aggregations: {e}")
    
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
            self._daily_usage_cache[user_profile.user_id] = {}
            self._monthly_usage_cache[user_profile.user_id] = {}
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
            
            # Update daily aggregation in cache
            today = datetime.now().date().isoformat()
            if user_id not in self._daily_usage_cache:
                self._daily_usage_cache[user_id] = {}
            if today not in self._daily_usage_cache[user_id]:
                # Try to load from database first
                daily_agg = await self._load_daily_aggregation(user_id, today)
                if daily_agg is None:
                    daily_agg = UserUsageAggregation(user_id=user_id, date=today)
                self._daily_usage_cache[user_id][today] = daily_agg
            
            daily_agg = self._daily_usage_cache[user_id][today]
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
            
            # Update monthly aggregation in cache
            current_month = datetime.now().strftime("%Y-%m")
            if user_id not in self._monthly_usage_cache:
                self._monthly_usage_cache[user_id] = {}
            if current_month not in self._monthly_usage_cache[user_id]:
                # Try to load from database first
                monthly_agg = await self._load_monthly_aggregation(user_id, current_month)
                if monthly_agg is None:
                    monthly_agg = UserUsageAggregation(user_id=user_id, date=current_month)
                self._monthly_usage_cache[user_id][current_month] = monthly_agg
            
            monthly_agg = self._monthly_usage_cache[user_id][current_month]
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
            daily_usage = await self.get_user_daily_usage(user_id, today)
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
            monthly_usage = await self.get_user_monthly_usage(user_id, current_month)
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
    
    async def get_user_daily_usage(self, user_id: str, date: Optional[str] = None) -> float:
        """Get total daily usage cost for a user."""
        if date is None:
            date = datetime.now().date().isoformat()
        
        # Check cache first
        if user_id in self._daily_usage_cache and date in self._daily_usage_cache[user_id]:
            return self._daily_usage_cache[user_id][date].total_cost
        
        # Load from database/partitioned storage
        daily_agg = await self._load_daily_aggregation(user_id, date)
        if daily_agg:
            # Cache the result
            if user_id not in self._daily_usage_cache:
                self._daily_usage_cache[user_id] = {}
            self._daily_usage_cache[user_id][date] = daily_agg
            return daily_agg.total_cost
            
        return 0.0
    
    async def get_user_monthly_usage(self, user_id: str, month: Optional[str] = None) -> float:
        """Get total monthly usage cost for a user."""
        if month is None:
            month = datetime.now().strftime("%Y-%m")
        
        # Check cache first
        if user_id in self._monthly_usage_cache and month in self._monthly_usage_cache[user_id]:
            return self._monthly_usage_cache[user_id][month].total_cost
        
        # Load from database/partitioned storage
        monthly_agg = await self._load_monthly_aggregation(user_id, month)
        if monthly_agg:
            # Cache the result
            if user_id not in self._monthly_usage_cache:
                self._monthly_usage_cache[user_id] = {}
            self._monthly_usage_cache[user_id][month] = monthly_agg
            return monthly_agg.total_cost
            
        return 0.0
    
    def get_user_session_usage(self, user_id: str, session_id: str) -> float:
        """Get usage cost for a specific session."""
        if user_id in self.active_sessions and session_id in self.active_sessions[user_id]:
            return self.active_sessions[user_id][session_id]
        return 0.0
    
    async def reserve_budget(self, user_id: str, amount: float) -> bool:
        """Reserve budget atomically to prevent race conditions in budget enforcement.
        
        This method atomically checks and reserves budget to ensure that concurrent
        requests don't exceed limits due to race conditions.
        
        Returns:
            True if budget was successfully reserved, False if it would exceed limits
        """
        async with self._lock:
            # Check current usage
            current_daily = await self.get_user_daily_usage(user_id)
            current_monthly = await self.get_user_monthly_usage(user_id)
            
            # Get user limits
            limits = self.user_limits.get(user_id)
            if not limits or not limits.enabled:
                return True  # No limits set, allow the reservation
            
            # Check if reservation would exceed limits
            if limits.daily_limit and (current_daily + amount) > limits.daily_limit:
                return False
            
            if limits.monthly_limit and (current_monthly + amount) > limits.monthly_limit:
                return False
            
            if limits.per_request_limit and amount > limits.per_request_limit:
                return False
            
            # Reserve the budget by updating aggregations immediately
            today = datetime.now().date().isoformat()
            current_month = datetime.now().strftime("%Y-%m")
            
            # Update daily usage in cache
            if user_id not in self._daily_usage_cache:
                self._daily_usage_cache[user_id] = {}
            if today not in self._daily_usage_cache[user_id]:
                # Load existing data if available
                daily_agg = await self._load_daily_aggregation(user_id, today)
                if daily_agg is None:
                    daily_agg = UserUsageAggregation(user_id=user_id, date=today)
                self._daily_usage_cache[user_id][today] = daily_agg
            
            self._daily_usage_cache[user_id][today].total_cost += amount
            
            # Update monthly usage in cache
            if user_id not in self._monthly_usage_cache:
                self._monthly_usage_cache[user_id] = {}
            if current_month not in self._monthly_usage_cache[user_id]:
                # Load existing data if available
                monthly_agg = await self._load_monthly_aggregation(user_id, current_month)
                if monthly_agg is None:
                    monthly_agg = UserUsageAggregation(user_id=user_id, date=current_month)
                self._monthly_usage_cache[user_id][current_month] = monthly_agg
            
            self._monthly_usage_cache[user_id][current_month].total_cost += amount
            
            return True
    
    async def _save_user_profiles(self) -> None:
        """Save user profiles to JSON file with proper persistence strategy."""
        try:
            profiles_file = self.storage_path / "user_profiles.jsonl"
            temp_file = self.storage_path / "user_profiles.jsonl.tmp"
            
            # Write to temporary file first to ensure atomicity
            async with aiofiles.open(temp_file, 'w') as f:
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
                    await f.write(json.dumps(user_data) + "\n")
            
            # Atomically replace the original file
            await asyncio.to_thread(temp_file.replace, profiles_file)
            
            # Clean up old .json file if it exists (migration support)
            old_json_file = self.storage_path / "user_profiles.json"
            if old_json_file.exists():
                backup_file = self.storage_path / "user_profiles.json.backup"
                await asyncio.to_thread(old_json_file.rename, backup_file)
                logger.info("Migrated user_profiles.json to .jsonl format, created backup")
                
        except Exception as e:
            logger.error("Failed to save user profiles", error=str(e))
            # Clean up temporary file on error
            temp_file = self.storage_path / "user_profiles.jsonl.tmp"
            if temp_file.exists():
                await asyncio.to_thread(temp_file.unlink)
    
    async def _save_spending_limits(self) -> None:
        """Save spending limits using partitioned storage for scalability."""
        try:
            # Create spending_limits directory if it doesn't exist
            limits_dir = self.storage_path / "spending_limits"
            limits_dir.mkdir(exist_ok=True)
            
            # Keep track of which user files exist for cleanup
            existing_files = set()
            
            # Save each user's limits to a separate file (partitioned by user_id)
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
                    "updated_at": limits.updated_at.isoformat(),
                    "last_updated": datetime.now().isoformat()
                }
                
                # Use user ID hash for consistent filename (avoid special chars in filenames)
                import hashlib
                user_hash = hashlib.sha256(limits.user_id.encode()).hexdigest()[:16]
                user_file = limits_dir / f"user_{user_hash}.json"
                existing_files.add(user_file.name)
                
                # Use atomic write with temp file
                temp_file = user_file.with_suffix('.tmp')
                await asyncio.to_thread(
                    self._write_json_file,
                    temp_file,
                    limit_data
                )
                # Atomic rename
                temp_file.replace(user_file)
            
            # Create index file for quick lookups (small file with user_id -> hash mapping)
            index_data = {
                "last_updated": datetime.now().isoformat(),
                "user_index": {
                    limits.user_id: hashlib.sha256(limits.user_id.encode()).hexdigest()[:16]
                    for limits in self.user_limits.values()
                }
            }
            
            index_file = limits_dir / "index.json"
            temp_index = index_file.with_suffix('.tmp')
            await asyncio.to_thread(
                self._write_json_file,
                temp_index,
                index_data
            )
            temp_index.replace(index_file)
            
            # Optional: Clean up old user files that are no longer in use
            # This prevents accumulation of orphaned files
            for existing_file in limits_dir.glob("user_*.json"):
                if existing_file.name not in existing_files:
                    try:
                        existing_file.unlink()
                        logger.debug(f"Removed orphaned limits file: {existing_file.name}")
                    except Exception as e:
                        logger.warning(f"Failed to remove orphaned file {existing_file.name}: {e}")
                
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
        """Save usage aggregations using scalable partitioned storage and database."""
        try:
            # Save to database for efficient querying
            if self._db_initialized:
                await self._save_aggregations_to_database()
            
            # Save to partitioned files for backup and compatibility
            await self._save_aggregations_to_partitioned_files()
            
        except Exception as e:
            logger.error("Failed to save usage aggregations", error=str(e))
    
    async def _save_aggregations_to_database(self) -> None:
        """Save aggregations to SQLite database efficiently."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Save daily aggregations
                for user_id, user_daily in self._daily_usage_cache.items():
                    for date, agg in user_daily.items():
                        await db.execute("""
                            INSERT OR REPLACE INTO daily_aggregations (
                                user_id, date, total_requests, successful_requests, failed_requests,
                                total_input_tokens, total_output_tokens, total_cost, avg_latency_ms,
                                session_count, providers_used, models_used, unique_sessions, updated_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                        """, (
                            user_id, date, agg.total_requests, agg.successful_requests, 
                            agg.failed_requests, agg.total_input_tokens, agg.total_output_tokens,
                            agg.total_cost, agg.avg_latency_ms, agg.session_count,
                            json.dumps(agg.providers_used), json.dumps(agg.models_used),
                            json.dumps(list(agg.unique_sessions) if isinstance(agg.unique_sessions, set) else agg.unique_sessions)
                        ))
                
                # Save monthly aggregations
                for user_id, user_monthly in self._monthly_usage_cache.items():
                    for month, agg in user_monthly.items():
                        await db.execute("""
                            INSERT OR REPLACE INTO monthly_aggregations (
                                user_id, month, total_requests, successful_requests, failed_requests,
                                total_input_tokens, total_output_tokens, total_cost, avg_latency_ms,
                                session_count, providers_used, models_used, unique_sessions, updated_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                        """, (
                            user_id, month, agg.total_requests, agg.successful_requests,
                            agg.failed_requests, agg.total_input_tokens, agg.total_output_tokens,
                            agg.total_cost, agg.avg_latency_ms, agg.session_count,
                            json.dumps(agg.providers_used), json.dumps(agg.models_used),
                            json.dumps(list(agg.unique_sessions) if isinstance(agg.unique_sessions, set) else agg.unique_sessions)
                        ))
                
                await db.commit()
                
        except Exception as e:
            logger.error("Failed to save aggregations to database", error=str(e))
    
    async def _save_aggregations_to_partitioned_files(self) -> None:
        """Save aggregations to partitioned JSON files for backup."""
        try:
            # Save daily aggregations in partitioned structure
            for user_id, user_daily in self._daily_usage_cache.items():
                for date, agg in user_daily.items():
                    partition_path = self._get_time_partition_path(user_id, date)
                    daily_file = partition_path / f"daily_{date}.json"
                    
                    agg_data = {
                        "user_id": user_id,
                        "date": date,
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
                        "unique_sessions": list(agg.unique_sessions) if isinstance(agg.unique_sessions, set) else agg.unique_sessions,
                        "updated_at": datetime.now().isoformat()
                    }
                    
                    # Use atomic write to prevent data corruption
                    temp_file = f"{daily_file}.tmp"
                    async with aiofiles.open(temp_file, 'w') as f:
                        await f.write(json.dumps(agg_data, indent=2))
                    
                    # Atomic replace
                    await aiofiles.os.replace(temp_file, daily_file)
            
            # Save monthly aggregations in partitioned structure
            for user_id, user_monthly in self._monthly_usage_cache.items():
                for month, agg in user_monthly.items():
                    user_path = self._get_user_partition_path(user_id)
                    monthly_file = user_path / f"monthly_{month}.json"
                    
                    agg_data = {
                        "user_id": user_id,
                        "month": month,
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
                        "unique_sessions": list(agg.unique_sessions) if isinstance(agg.unique_sessions, set) else agg.unique_sessions,
                        "updated_at": datetime.now().isoformat()
                    }
                    
                    # Use atomic write to prevent data corruption
                    temp_file = f"{monthly_file}.tmp"
                    async with aiofiles.open(temp_file, 'w') as f:
                        await f.write(json.dumps(agg_data, indent=2))
                    
                    # Atomic replace
                    await aiofiles.os.replace(temp_file, monthly_file)
                        
        except Exception as e:
            logger.error("Failed to save aggregations to partitioned files", error=str(e))
    
    async def _migrate_large_json_file(self, file_path: Path, data_type: str) -> None:
        """Migrate large JSON files using streaming to prevent memory issues."""
        try:
            import ijson  # Streaming JSON parser
            
            # Process file in chunks to avoid loading entire file into memory
            batch_size = 0
            processed_users = 0
            
            async with aiofiles.open(file_path, 'rb') as file:
                parser = ijson.parse(file)
                current_user = None
                current_data = {}
                
                async for prefix, event, value in parser:
                    if prefix == '' and event == 'start_map':
                        # Start of root object
                        continue
                    elif prefix.count('.') == 0 and event == 'map_key':
                        # User ID key
                        current_user = value
                        current_data = {}
                    elif prefix.count('.') == 1 and event == 'start_map':
                        # Start of user's data object
                        continue
                    elif prefix.count('.') == 1 and event == 'map_key':
                        # Date/month key for this user
                        current_date = value
                    elif prefix.count('.') == 2 and event in ('start_map', 'end_map'):
                        # Skip nested object markers
                        continue
                    elif prefix.count('.') == 2 and event == 'end_map' and current_user:
                        # End of a date's data - process it
                        agg_data = current_data.get(current_date, {})
                        if agg_data:
                            agg = UserUsageAggregation(
                                user_id=current_user,
                                date=current_date,
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
                            
                            # Save to appropriate cache
                            cache = (self._daily_usage_cache if data_type == "daily" 
                                   else self._monthly_usage_cache)
                            if current_user not in cache:
                                cache[current_user] = {}
                            cache[current_user][current_date] = agg
                            
                            batch_size += 1
                            
                            # Periodically save batches to prevent memory buildup
                            if batch_size >= 100:
                                await self._save_aggregations_to_partitioned_files()
                                batch_size = 0
                                logger.debug(f"Processed batch for {processed_users} users")
                    
                    elif prefix.count('.') == 1 and event == 'end_map' and current_user:
                        # End of user's data
                        processed_users += 1
                        if processed_users % 10 == 0:
                            logger.info(f"Migrated {processed_users} users from legacy {data_type} file")
            
            # Save any remaining data
            if batch_size > 0:
                await self._save_aggregations_to_partitioned_files()
                
            # Backup and remove the legacy file
            backup_file = file_path.with_suffix(f"{file_path.suffix}.legacy_backup")
            await asyncio.to_thread(file_path.rename, backup_file)
            logger.info(f"Legacy {data_type} file backed up and migrated ({processed_users} users)")
            
        except ImportError:
            logger.warning("ijson not available for streaming JSON parsing, falling back to memory-based migration")
            # Fallback to original approach for smaller files
            data = await asyncio.to_thread(self._read_json_file, file_path)
            await self._process_legacy_data(data, data_type)
            
        except Exception as e:
            logger.error(f"Failed to migrate large {data_type} file: {e}")
            raise
    
    async def _process_legacy_data(self, data: dict, data_type: str) -> None:
        """Process legacy data from in-memory dictionary."""
        for user_id, user_data in data.items():
            for date_key, agg_data in user_data.items():
                agg = UserUsageAggregation(
                    user_id=user_id,
                    date=date_key,
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
                
                # Save to appropriate cache
                cache = (self._daily_usage_cache if data_type == "daily" 
                       else self._monthly_usage_cache)
                if user_id not in cache:
                    cache[user_id] = {}
                cache[user_id][date_key] = agg

    def _write_json_file(self, file_path, data):
        """Synchronous helper method for writing JSON files atomically."""
        temp_file = f"{file_path}.tmp"
        with open(temp_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        # Atomic replace to prevent data corruption
        import os
        os.replace(temp_file, file_path)
    
    def _read_json_file(self, file_path):
        """Synchronous helper method for reading JSON files."""
        with open(file_path, 'r') as f:
            return json.load(f)
    
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
    
    async def get_user_usage_summary(self, user_id: str, days: int = 30) -> Dict[str, Any]:
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
        
        # Get daily usage from cache first, then try to load missing data
        if user_id in self._daily_usage_cache:
            for date_str, agg in self._daily_usage_cache[user_id].items():
                date = datetime.fromisoformat(date_str).date()
                if start_date <= date <= end_date:
                    summary["daily_usage"][date_str] = asdict(agg)
                    total_requests += agg.total_requests
                    total_cost += agg.total_cost
                    
                    for provider, count in agg.providers_used.items():
                        provider_counts[provider] += count
                    
                    for model, count in agg.models_used.items():
                        model_counts[model] += count
        
        # Load additional data from database if needed for the requested time range
        # This ensures we get complete data even if not all is cached
        for i in range(days):
            check_date = (start_date + timedelta(days=i)).isoformat()
            if check_date not in summary["daily_usage"]:
                daily_agg = await self._load_daily_aggregation(user_id, check_date)
                if daily_agg:
                    summary["daily_usage"][check_date] = asdict(daily_agg)
                    total_requests += daily_agg.total_requests
                    total_cost += daily_agg.total_cost
                    
                    for provider, count in daily_agg.providers_used.items():
                        provider_counts[provider] += count
                    
                    for model, count in daily_agg.models_used.items():
                        model_counts[model] += count
        
        # Get monthly usage from cache
        if user_id in self._monthly_usage_cache:
            summary["monthly_usage"] = {
                month: asdict(agg) for month, agg in self._monthly_usage_cache[user_id].items()
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