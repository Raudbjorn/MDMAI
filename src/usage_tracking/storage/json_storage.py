"""
JSON file-based storage implementation for usage tracking.

This module provides:
- Atomic file operations for consistency
- Efficient user profile and budget management
- Transaction logging with rotation
- Backup and recovery mechanisms
- Concurrent access handling
"""

import json
import logging
import os
import shutil
import threading
import time
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import fcntl
import gzip
from contextlib import contextmanager

from .models import (
    UserProfile, SpendingLimit, UsageRecord, UsageMetrics,
    JSONStorageConfig, TimeAggregation
)

logger = logging.getLogger(__name__)


class DecimalEncoder(json.JSONEncoder):
    """JSON encoder that handles Decimal objects."""
    
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


class AtomicFileWriter:
    """Atomic file writer with backup and rollback capabilities."""
    
    def __init__(self, file_path: Path, create_backup: bool = True, compress: bool = False):
        self.file_path = file_path
        self.temp_path = file_path.with_suffix(f"{file_path.suffix}.tmp.{os.getpid()}")
        self.backup_path = file_path.with_suffix(f"{file_path.suffix}.backup")
        self.create_backup = create_backup
        self.compress = compress
        
        # Ensure parent directory exists
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
    
    @contextmanager
    def write(self):
        """Context manager for atomic file writing."""
        try:
            # Create backup if file exists
            if self.create_backup and self.file_path.exists():
                shutil.copy2(self.file_path, self.backup_path)
            
            # Open temporary file for writing
            if self.compress:
                file_obj = gzip.open(self.temp_path, 'wt', encoding='utf-8')
            else:
                file_obj = open(self.temp_path, 'w', encoding='utf-8')
            
            try:
                # Acquire exclusive lock
                fcntl.flock(file_obj.fileno(), fcntl.LOCK_EX)
                yield file_obj
                
                # Force write to disk
                file_obj.flush()
                os.fsync(file_obj.fileno())
                
            finally:
                file_obj.close()
            
            # Atomic move from temp to final location
            os.rename(self.temp_path, self.file_path)
            
        except Exception as e:
            # Clean up temp file on error
            if self.temp_path.exists():
                self.temp_path.unlink()
            
            # Restore from backup if available
            if self.create_backup and self.backup_path.exists():
                logger.warning(f"Restoring from backup due to write error: {e}")
                shutil.copy2(self.backup_path, self.file_path)
            
            raise
    
    def read_with_fallback(self) -> Dict[str, Any]:
        """Read file with fallback to backup if corrupted."""
        for file_path in [self.file_path, self.backup_path]:
            if not file_path.exists():
                continue
            
            try:
                if self.compress:
                    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                        return json.load(f)
                else:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to read {file_path}: {e}")
                continue
        
        return {}


class TransactionLog:
    """Transaction log for tracking all usage records chronologically."""
    
    def __init__(self, base_path: Path, max_file_size_mb: int = 100):
        self.base_path = base_path
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
        self.current_file_path = None
        self.current_file = None
        self.lock = threading.RLock()
        
        # Ensure log directory exists
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize current log file
        self._initialize_current_file()
    
    def _initialize_current_file(self) -> None:
        """Initialize or rotate to current log file."""
        with self.lock:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            self.current_file_path = self.base_path / f"transactions_{timestamp}.jsonl"
            
            # Close previous file if open
            if self.current_file:
                self.current_file.close()
            
            # Open new file in append mode
            self.current_file = open(self.current_file_path, 'a', encoding='utf-8')
    
    def _should_rotate(self) -> bool:
        """Check if current log file should be rotated."""
        if not self.current_file_path or not self.current_file_path.exists():
            return True
        
        return self.current_file_path.stat().st_size >= self.max_file_size_bytes
    
    def append_record(self, record: UsageRecord) -> None:
        """Append usage record to transaction log."""
        with self.lock:
            try:
                # Check if rotation is needed
                if self._should_rotate():
                    self._rotate_log_file()
                
                # Prepare log entry
                log_entry = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "record_id": record.record_id,
                    "user_id": record.user_id,
                    "event_type": record.event_type.value,
                    "provider": record.provider.value,
                    "cost_usd": float(record.cost_usd),
                    "token_count": record.token_count,
                    "success": record.success,
                    "session_id": record.session_id,
                    "context_id": record.context_id,
                    "operation": record.operation,
                }
                
                # Write to log file
                json.dump(log_entry, self.current_file, cls=DecimalEncoder)
                self.current_file.write('\n')
                self.current_file.flush()
                
            except Exception as e:
                logger.error(f"Failed to append to transaction log: {e}")
                raise
    
    def _rotate_log_file(self) -> None:
        """Rotate to a new log file."""
        # Compress old log file
        if self.current_file_path and self.current_file_path.exists():
            compressed_path = self.current_file_path.with_suffix('.jsonl.gz')
            with open(self.current_file_path, 'rb') as f_in:
                with gzip.open(compressed_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            # Remove uncompressed file
            self.current_file_path.unlink()
            
            logger.info(f"Rotated log file: {compressed_path}")
        
        # Initialize new file
        self._initialize_current_file()
    
    def read_records_range(
        self,
        start_time: datetime,
        end_time: datetime,
        user_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Read transaction records within time range."""
        records = []
        
        # Find relevant log files
        log_files = []
        for file_path in self.base_path.iterdir():
            if file_path.suffix in ['.jsonl', '.gz']:
                log_files.append(file_path)
        
        # Sort by creation time
        log_files.sort(key=lambda p: p.stat().st_mtime)
        
        # Read records from all files
        for file_path in log_files:
            try:
                if file_path.suffix == '.gz':
                    file_obj = gzip.open(file_path, 'rt', encoding='utf-8')
                else:
                    file_obj = open(file_path, 'r', encoding='utf-8')
                
                with file_obj:
                    for line in file_obj:
                        line = line.strip()
                        if not line:
                            continue
                        
                        try:
                            entry = json.loads(line)
                            entry_time = datetime.fromisoformat(entry['timestamp'])
                            
                            # Filter by time range
                            if entry_time < start_time or entry_time > end_time:
                                continue
                            
                            # Filter by user if specified
                            if user_id and entry.get('user_id') != user_id:
                                continue
                            
                            records.append(entry)
                            
                        except (json.JSONDecodeError, ValueError) as e:
                            logger.warning(f"Skipping invalid log entry: {e}")
                            continue
            
            except Exception as e:
                logger.error(f"Error reading log file {file_path}: {e}")
                continue
        
        return records
    
    def cleanup_old_logs(self, retention_days: int) -> int:
        """Clean up old log files beyond retention period."""
        cutoff_time = time.time() - (retention_days * 24 * 3600)
        cleaned_files = 0
        
        for file_path in self.base_path.iterdir():
            if file_path.suffix in ['.jsonl', '.gz']:
                try:
                    if file_path.stat().st_mtime < cutoff_time:
                        file_path.unlink()
                        cleaned_files += 1
                        logger.info(f"Cleaned up old log file: {file_path}")
                except Exception as e:
                    logger.error(f"Error cleaning up {file_path}: {e}")
        
        return cleaned_files
    
    def close(self) -> None:
        """Close transaction log."""
        with self.lock:
            if self.current_file:
                self.current_file.close()
                self.current_file = None


class JSONUsageStorage:
    """JSON file-based storage for usage tracking data."""
    
    def __init__(self, config: JSONStorageConfig):
        self.config = config
        self.base_path = Path(config.base_path)
        
        # Initialize storage directories
        self.profiles_path = self.base_path / config.profiles_path
        self.transactions_path = self.base_path / config.transactions_path
        self.analytics_path = self.base_path / config.analytics_path
        self.backups_path = self.base_path / config.backups_path
        
        for path in [self.profiles_path, self.transactions_path, 
                     self.analytics_path, self.backups_path]:
            path.mkdir(parents=True, exist_ok=True)
        
        # Initialize transaction log
        self.transaction_log = TransactionLog(
            self.transactions_path,
            config.max_file_size_mb
        )
        
        # Cache for user profiles
        self._profile_cache = {}
        self._cache_lock = threading.RLock()
        
        # Performance tracking
        self._operation_stats = {}
        
        logger.info(
            "JSON storage initialized",
            base_path=str(self.base_path),
            compression=config.compression,
            atomic_writes=config.use_atomic_writes
        )
    
    def _get_profile_path(self, user_id: str) -> Path:
        """Get file path for user profile."""
        # Use first 2 chars of user_id for directory sharding
        shard = user_id[:2] if len(user_id) >= 2 else "00"
        shard_dir = self.profiles_path / shard
        shard_dir.mkdir(exist_ok=True)
        
        return shard_dir / f"{user_id}.json"
    
    async def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Get user profile from storage."""
        start_time = time.time()
        
        try:
            # Check cache first
            with self._cache_lock:
                if user_id in self._profile_cache:
                    profile_data = self._profile_cache[user_id].copy()
                    return UserProfile(**profile_data)
            
            # Load from file
            profile_path = self._get_profile_path(user_id)
            atomic_writer = AtomicFileWriter(
                profile_path,
                create_backup=self.config.create_backups,
                compress=self.config.compression
            )
            
            profile_data = atomic_writer.read_with_fallback()
            
            if not profile_data:
                return None
            
            # Update cache
            with self._cache_lock:
                self._profile_cache[user_id] = profile_data.copy()
            
            # Convert spending limits
            if 'spending_limits' in profile_data:
                spending_limits = {}
                for limit_type, limit_data in profile_data['spending_limits'].items():
                    spending_limits[limit_type] = SpendingLimit(**limit_data)
                profile_data['spending_limits'] = spending_limits
            
            profile = UserProfile(**profile_data)
            
            # Track performance
            execution_time = time.time() - start_time
            self._track_operation_performance("get_user_profile", execution_time)
            
            return profile
            
        except Exception as e:
            logger.error(f"Failed to get user profile {user_id}: {e}")
            return None
    
    async def save_user_profile(self, profile: UserProfile) -> bool:
        """Save user profile to storage."""
        start_time = time.time()
        
        try:
            profile_path = self._get_profile_path(profile.user_id)
            atomic_writer = AtomicFileWriter(
                profile_path,
                create_backup=self.config.create_backups,
                compress=self.config.compression
            )
            
            # Update timestamps
            profile.updated_at = datetime.utcnow()
            
            # Convert to dict for JSON serialization
            profile_data = profile.dict()
            
            # Write atomically
            with atomic_writer.write() as f:
                json.dump(profile_data, f, cls=DecimalEncoder, indent=2)
            
            # Update cache
            with self._cache_lock:
                self._profile_cache[profile.user_id] = profile_data.copy()
            
            # Track performance
            execution_time = time.time() - start_time
            self._track_operation_performance("save_user_profile", execution_time)
            
            logger.debug(
                "User profile saved",
                user_id=profile.user_id,
                execution_time=execution_time
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save user profile {profile.user_id}: {e}")
            return False
    
    async def update_spending_limit(
        self,
        user_id: str,
        limit_type: str,
        limit: SpendingLimit
    ) -> bool:
        """Update a specific spending limit for a user."""
        try:
            profile = await self.get_user_profile(user_id)
            if not profile:
                # Create new profile
                profile = UserProfile(user_id=user_id)
            
            profile.spending_limits[limit_type] = limit
            return await self.save_user_profile(profile)
            
        except Exception as e:
            logger.error(f"Failed to update spending limit: {e}")
            return False
    
    async def add_usage_to_limits(
        self,
        user_id: str,
        cost: Decimal,
        timestamp: Optional[datetime] = None
    ) -> Dict[str, bool]:
        """Add usage cost to user's spending limits and check if exceeded."""
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        try:
            profile = await self.get_user_profile(user_id)
            if not profile:
                return {}
            
            results = {}
            updated = False
            
            for limit_type, spending_limit in profile.spending_limits.items():
                # Check if limit needs reset based on time
                if self._should_reset_limit(spending_limit, timestamp):
                    spending_limit.current_spent = Decimal("0.00")
                    spending_limit.reset_date = self._calculate_next_reset(limit_type, timestamp)
                    updated = True
                
                # Add current cost
                spending_limit.current_spent += cost
                updated = True
                
                # Check if limit is exceeded
                results[limit_type] = not spending_limit.is_exceeded
            
            # Save updated profile if changed
            if updated:
                await self.save_user_profile(profile)
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to add usage to limits: {e}")
            return {}
    
    def _should_reset_limit(self, limit: SpendingLimit, current_time: datetime) -> bool:
        """Check if spending limit should be reset based on time."""
        if not limit.reset_date:
            return True
        
        return current_time >= limit.reset_date
    
    def _calculate_next_reset(self, limit_type: str, current_time: datetime) -> datetime:
        """Calculate next reset time for spending limit."""
        if limit_type == "daily":
            return current_time.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        elif limit_type == "weekly":
            days_until_monday = (7 - current_time.weekday()) % 7
            if days_until_monday == 0:
                days_until_monday = 7
            return current_time.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=days_until_monday)
        elif limit_type == "monthly":
            if current_time.month == 12:
                return datetime(current_time.year + 1, 1, 1)
            else:
                return datetime(current_time.year, current_time.month + 1, 1)
        else:
            # Default to daily
            return current_time.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
    
    async def store_usage_record(self, record: UsageRecord) -> str:
        """Store usage record in transaction log."""
        start_time = time.time()
        
        try:
            # Add to transaction log
            self.transaction_log.append_record(record)
            
            # Update user spending limits
            if record.cost_usd > 0:
                await self.add_usage_to_limits(record.user_id, record.cost_usd, record.timestamp)
            
            # Track performance
            execution_time = time.time() - start_time
            self._track_operation_performance("store_usage_record", execution_time)
            
            return record.record_id
            
        except Exception as e:
            logger.error(f"Failed to store usage record: {e}")
            raise
    
    async def get_usage_records(
        self,
        user_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """Get usage records from transaction log."""
        start = start_time or (datetime.utcnow() - timedelta(days=7))
        end = end_time or datetime.utcnow()
        
        try:
            records = self.transaction_log.read_records_range(start, end, user_id)
            
            # Sort by timestamp and limit results
            records.sort(key=lambda r: r.get('timestamp', ''), reverse=True)
            return records[:limit]
            
        except Exception as e:
            logger.error(f"Failed to get usage records: {e}")
            return []
    
    async def store_usage_metrics(self, metrics: UsageMetrics) -> str:
        """Store aggregated usage metrics."""
        start_time = time.time()
        
        try:
            # Determine file path based on aggregation type and time
            date_str = metrics.period_start.strftime("%Y%m%d")
            filename = f"metrics_{metrics.aggregation_type.value}_{date_str}_{metrics.metric_id}.json"
            
            metrics_path = self.analytics_path / filename
            atomic_writer = AtomicFileWriter(
                metrics_path,
                create_backup=self.config.create_backups,
                compress=self.config.compression
            )
            
            # Convert to dict and write
            metrics_data = metrics.dict()
            with atomic_writer.write() as f:
                json.dump(metrics_data, f, cls=DecimalEncoder, indent=2)
            
            # Track performance
            execution_time = time.time() - start_time
            self._track_operation_performance("store_usage_metrics", execution_time)
            
            logger.debug(
                "Usage metrics stored",
                metric_id=metrics.metric_id,
                aggregation_type=metrics.aggregation_type.value,
                execution_time=execution_time
            )
            
            return metrics.metric_id
            
        except Exception as e:
            logger.error(f"Failed to store usage metrics: {e}")
            raise
    
    async def get_usage_metrics(
        self,
        aggregation_type: TimeAggregation,
        start_date: datetime,
        end_date: datetime,
        user_id: Optional[str] = None
    ) -> List[UsageMetrics]:
        """Get stored usage metrics."""
        metrics_list = []
        
        try:
            # Find relevant metrics files
            pattern = f"metrics_{aggregation_type.value}_*"
            for metrics_file in self.analytics_path.glob(pattern):
                try:
                    atomic_writer = AtomicFileWriter(
                        metrics_file,
                        create_backup=False,
                        compress=self.config.compression
                    )
                    
                    metrics_data = atomic_writer.read_with_fallback()
                    if not metrics_data:
                        continue
                    
                    # Check if metrics match filters
                    metrics_start = datetime.fromisoformat(metrics_data['period_start'])
                    metrics_end = datetime.fromisoformat(metrics_data['period_end'])
                    
                    # Check time range overlap
                    if metrics_end < start_date or metrics_start > end_date:
                        continue
                    
                    # Check user filter
                    if user_id and metrics_data.get('user_id') != user_id:
                        continue
                    
                    # Convert spending limits back to Decimal
                    metrics = UsageMetrics(**metrics_data)
                    metrics_list.append(metrics)
                    
                except Exception as e:
                    logger.warning(f"Error loading metrics from {metrics_file}: {e}")
                    continue
            
            # Sort by period start time
            metrics_list.sort(key=lambda m: m.period_start)
            
        except Exception as e:
            logger.error(f"Failed to get usage metrics: {e}")
        
        return metrics_list
    
    def _track_operation_performance(self, operation: str, execution_time: float) -> None:
        """Track operation performance metrics."""
        if operation not in self._operation_stats:
            self._operation_stats[operation] = {
                "count": 0,
                "total_time": 0.0,
                "avg_time": 0.0,
                "max_time": 0.0,
                "min_time": float('inf')
            }
        
        stats = self._operation_stats[operation]
        stats["count"] += 1
        stats["total_time"] += execution_time
        stats["avg_time"] = stats["total_time"] / stats["count"]
        stats["max_time"] = max(stats["max_time"], execution_time)
        stats["min_time"] = min(stats["min_time"], execution_time)
    
    async def get_storage_stats(self) -> Dict[str, Any]:
        """Get comprehensive storage statistics."""
        stats = {
            "operation_performance": self._operation_stats,
            "cache_size": len(self._profile_cache),
            "storage_paths": {
                "profiles": str(self.profiles_path),
                "transactions": str(self.transactions_path),
                "analytics": str(self.analytics_path),
                "backups": str(self.backups_path)
            }
        }
        
        # Get directory sizes
        for name, path_str in stats["storage_paths"].items():
            path = Path(path_str)
            if path.exists():
                total_size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
                file_count = sum(1 for f in path.rglob('*') if f.is_file())
                stats[f"{name}_size_bytes"] = total_size
                stats[f"{name}_file_count"] = file_count
        
        return stats
    
    async def cleanup_old_data(self, retention_days: int = 90) -> Dict[str, int]:
        """Clean up old data beyond retention period."""
        cleanup_results = {}
        
        # Clean up transaction logs
        cleanup_results["transaction_logs"] = self.transaction_log.cleanup_old_logs(retention_days)
        
        # Clean up old analytics files
        cutoff_time = time.time() - (retention_days * 24 * 3600)
        cleaned_analytics = 0
        
        for file_path in self.analytics_path.rglob('*'):
            if file_path.is_file():
                try:
                    if file_path.stat().st_mtime < cutoff_time:
                        file_path.unlink()
                        cleaned_analytics += 1
                except Exception as e:
                    logger.error(f"Error cleaning up {file_path}: {e}")
        
        cleanup_results["analytics_files"] = cleaned_analytics
        
        # Clean up old backups
        cleaned_backups = 0
        for file_path in self.backups_path.rglob('*'):
            if file_path.is_file():
                try:
                    if file_path.stat().st_mtime < cutoff_time:
                        file_path.unlink()
                        cleaned_backups += 1
                except Exception as e:
                    logger.error(f"Error cleaning up backup {file_path}: {e}")
        
        cleanup_results["backup_files"] = cleaned_backups
        
        logger.info("Data cleanup completed", **cleanup_results)
        return cleanup_results
    
    async def create_backup(self) -> str:
        """Create a backup of all storage data."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        backup_name = f"usage_storage_backup_{timestamp}"
        backup_path = self.backups_path / backup_name
        
        try:
            # Create backup directory
            backup_path.mkdir(exist_ok=True)
            
            # Backup user profiles
            profiles_backup = backup_path / "profiles"
            if self.profiles_path.exists():
                shutil.copytree(self.profiles_path, profiles_backup)
            
            # Backup analytics
            analytics_backup = backup_path / "analytics"
            if self.analytics_path.exists():
                shutil.copytree(self.analytics_path, analytics_backup)
            
            # Compress backup
            compressed_backup = f"{backup_path}.tar.gz"
            shutil.make_archive(str(backup_path), 'gztar', str(backup_path))
            
            # Remove uncompressed backup
            shutil.rmtree(backup_path)
            
            logger.info(f"Backup created: {compressed_backup}")
            return compressed_backup
            
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            # Clean up partial backup
            if backup_path.exists():
                shutil.rmtree(backup_path)
            raise
    
    async def close(self) -> None:
        """Close JSON storage and clean up resources."""
        try:
            # Close transaction log
            self.transaction_log.close()
            
            # Clear cache
            with self._cache_lock:
                self._profile_cache.clear()
            
            logger.info("JSON storage closed successfully")
            
        except Exception as e:
            logger.error(f"Error closing JSON storage: {e}")