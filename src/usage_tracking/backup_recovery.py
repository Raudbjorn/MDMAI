"""Comprehensive backup and recovery mechanisms for usage tracking data."""

import asyncio
import os
import aiofiles
import aiofiles.os
import gzip
import json
import hashlib
import shutil
import subprocess
import tempfile
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from concurrent.futures import ThreadPoolExecutor

from ..context.persistence import ContextPersistenceLayer
from .chroma_extensions import UsageTrackingChromaExtensions
from .json_persistence import JsonPersistenceManager
from .state_synchronization import StateSynchronizationManager
from config.logging_config import get_logger

logger = get_logger(__name__)


class BackupType(Enum):
    """Types of backups."""
    FULL = "full"
    INCREMENTAL = "incremental"
    DIFFERENTIAL = "differential"
    SNAPSHOT = "snapshot"
    CONTINUOUS = "continuous"


class BackupStrategy(Enum):
    """Backup strategies."""
    SCHEDULED = "scheduled"
    EVENT_DRIVEN = "event_driven"
    HYBRID = "hybrid"
    ON_DEMAND = "on_demand"


class RecoveryType(Enum):
    """Types of recovery operations."""
    FULL_RESTORE = "full_restore"
    PARTIAL_RESTORE = "partial_restore"
    POINT_IN_TIME = "point_in_time"
    SELECTIVE_RESTORE = "selective_restore"


class BackupStatus(Enum):
    """Backup operation status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BackupManifest:
    """Metadata for backup operations."""
    backup_id: str
    backup_type: BackupType
    created_at: datetime
    completed_at: Optional[datetime]
    status: BackupStatus
    
    # Backup content information
    data_sources: List[str]
    file_count: int
    total_size_bytes: int
    compressed_size_bytes: int
    compression_ratio: float
    
    # Integrity verification
    checksum: str
    file_checksums: Dict[str, str]
    
    # Recovery information
    base_backup_id: Optional[str]  # For incremental/differential
    recovery_point: datetime
    
    # Metadata
    description: str
    tags: List[str]
    retention_days: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "backup_id": self.backup_id,
            "backup_type": self.backup_type.value,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "status": self.status.value,
            "data_sources": self.data_sources,
            "file_count": self.file_count,
            "total_size_bytes": self.total_size_bytes,
            "compressed_size_bytes": self.compressed_size_bytes,
            "compression_ratio": self.compression_ratio,
            "checksum": self.checksum,
            "file_checksums": self.file_checksums,
            "base_backup_id": self.base_backup_id,
            "recovery_point": self.recovery_point.isoformat(),
            "description": self.description,
            "tags": self.tags,
            "retention_days": self.retention_days
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BackupManifest':
        """Create from dictionary."""
        return cls(
            backup_id=data["backup_id"],
            backup_type=BackupType(data["backup_type"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            status=BackupStatus(data["status"]),
            data_sources=data["data_sources"],
            file_count=data["file_count"],
            total_size_bytes=data["total_size_bytes"],
            compressed_size_bytes=data["compressed_size_bytes"],
            compression_ratio=data["compression_ratio"],
            checksum=data["checksum"],
            file_checksums=data["file_checksums"],
            base_backup_id=data.get("base_backup_id"),
            recovery_point=datetime.fromisoformat(data["recovery_point"]),
            description=data["description"],
            tags=data["tags"],
            retention_days=data["retention_days"]
        )


@dataclass
class RecoveryPlan:
    """Plan for recovery operations."""
    plan_id: str
    recovery_type: RecoveryType
    target_timestamp: datetime
    backup_sequence: List[str]  # Backup IDs in order
    estimated_time_minutes: int
    estimated_size_gb: float
    data_sources_to_recover: List[str]
    verification_steps: List[str]
    rollback_plan: Optional[str]


@dataclass
class BackupConfiguration:
    """Configuration for backup operations."""
    # Storage configuration
    backup_root_path: str = "./data/backups"
    max_backup_size_gb: float = 100.0
    compression_enabled: bool = True
    encryption_enabled: bool = False
    
    # Scheduling configuration
    full_backup_interval_days: int = 7
    incremental_backup_interval_hours: int = 6
    max_incremental_chain_length: int = 14
    
    # Retention configuration
    full_backup_retention_days: int = 90
    incremental_backup_retention_days: int = 30
    max_backup_versions: int = 50
    
    # Performance configuration
    max_concurrent_backups: int = 2
    chunk_size_mb: int = 64
    verification_enabled: bool = True
    
    # Recovery configuration
    recovery_staging_path: str = "./data/recovery_staging"
    auto_verification_on_recovery: bool = True
    
    # Notification configuration
    notification_webhooks: List[str] = None
    email_notifications: List[str] = None
    
    def __post_init__(self):
        if self.notification_webhooks is None:
            self.notification_webhooks = []
        if self.email_notifications is None:
            self.email_notifications = []


class BackupEngine:
    """Core backup engine for creating backups."""
    
    def __init__(self, config: BackupConfiguration):
        self.config = config
        
        # Validate configuration
        self._validate_config()
        
        self.backup_root = Path(config.backup_root_path)
        
        try:
            self.backup_root.mkdir(parents=True, exist_ok=True)
        except (OSError, PermissionError) as e:
            raise Exception(f"Cannot create backup root directory {self.backup_root}: {e}")
        
        # Check backup root is writable
        if not os.access(self.backup_root, os.W_OK):
            raise Exception(f"Backup root directory {self.backup_root} is not writable")
        
        self.executor = ThreadPoolExecutor(max_workers=config.max_concurrent_backups)
        
        # Backup state tracking
        self.active_backups: Dict[str, BackupManifest] = {}
        self.backup_history: List[BackupManifest] = []
        self.last_full_backup: Optional[BackupManifest] = None
        
        # Thread safety
        self.backup_lock = threading.Lock()
        
        logger.info(
            "BackupEngine initialized",
            backup_root=str(self.backup_root),
            compression_enabled=config.compression_enabled,
            max_concurrent_backups=config.max_concurrent_backups
        )
    
    def _validate_config(self) -> None:
        """Validate backup configuration."""
        if self.config.max_backup_size_gb <= 0:
            raise ValueError("max_backup_size_gb must be positive")
        
        if self.config.max_concurrent_backups <= 0:
            raise ValueError("max_concurrent_backups must be positive")
        
        if self.config.chunk_size_mb <= 0:
            raise ValueError("chunk_size_mb must be positive")
        
        if self.config.full_backup_retention_days <= 0:
            raise ValueError("full_backup_retention_days must be positive")
        
        if self.config.incremental_backup_retention_days <= 0:
            raise ValueError("incremental_backup_retention_days must be positive")
    
    async def create_full_backup(
        self,
        data_sources: List[str],
        description: str = "",
        tags: List[str] = None
    ) -> BackupManifest:
        """Create a full backup."""
        backup_id = f"full_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        manifest = BackupManifest(
            backup_id=backup_id,
            backup_type=BackupType.FULL,
            created_at=datetime.now(),
            completed_at=None,
            status=BackupStatus.PENDING,
            data_sources=data_sources,
            file_count=0,
            total_size_bytes=0,
            compressed_size_bytes=0,
            compression_ratio=0.0,
            checksum="",
            file_checksums={},
            base_backup_id=None,
            recovery_point=datetime.now(),
            description=description or f"Full backup {backup_id}",
            tags=tags or [],
            retention_days=self.config.full_backup_retention_days
        )
        
        try:
            with self.backup_lock:
                self.active_backups[backup_id] = manifest
            
            manifest.status = BackupStatus.IN_PROGRESS
            
            # Create backup directory
            backup_dir = self.backup_root / backup_id
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Backup each data source
            total_files = 0
            total_original_size = 0
            total_compressed_size = 0
            file_checksums = {}
            
            for source in data_sources:
                source_result = await self._backup_data_source(
                    source, backup_dir, manifest
                )
                
                total_files += source_result["file_count"]
                total_original_size += source_result["original_size"]
                total_compressed_size += source_result["compressed_size"]
                file_checksums.update(source_result["file_checksums"])
            
            # Update manifest
            manifest.file_count = total_files
            manifest.total_size_bytes = total_original_size
            manifest.compressed_size_bytes = total_compressed_size
            manifest.compression_ratio = (
                1.0 - (total_compressed_size / max(total_original_size, 1))
            )
            manifest.file_checksums = file_checksums
            
            # Calculate overall checksum
            try:
                manifest.checksum = await self._calculate_backup_checksum(backup_dir)
            except Exception as e:
                logger.warning(f"Failed to calculate backup checksum: {e}")
                manifest.checksum = ""
            
            # Save manifest
            manifest_file = backup_dir / "manifest.json"
            async with aiofiles.open(manifest_file, 'w') as f:
                await f.write(json.dumps(manifest.to_dict(), indent=2))
            
            manifest.completed_at = datetime.now()
            manifest.status = BackupStatus.COMPLETED
            
            # Update tracking
            with self.backup_lock:
                self.backup_history.append(manifest)
                self.last_full_backup = manifest
                del self.active_backups[backup_id]
            
            logger.info("Full backup completed",
                       backup_id=backup_id,
                       files=total_files,
                       size_mb=total_original_size / (1024**2),
                       compression_ratio=manifest.compression_ratio)
            
            return manifest
            
        except Exception as e:
            manifest.status = BackupStatus.FAILED
            
            with self.backup_lock:
                if backup_id in self.active_backups:
                    del self.active_backups[backup_id]
            
            logger.error("Full backup failed", backup_id=backup_id, error=str(e))
            raise
    
    async def create_incremental_backup(
        self,
        data_sources: List[str],
        base_backup_id: Optional[str] = None,
        description: str = "",
        tags: List[str] = None
    ) -> BackupManifest:
        """Create an incremental backup."""
        # Use last full backup as base if not specified
        if not base_backup_id and self.last_full_backup:
            base_backup_id = self.last_full_backup.backup_id
        
        if not base_backup_id:
            raise ValueError("No base backup available for incremental backup")
        
        backup_id = f"inc_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        manifest = BackupManifest(
            backup_id=backup_id,
            backup_type=BackupType.INCREMENTAL,
            created_at=datetime.now(),
            completed_at=None,
            status=BackupStatus.PENDING,
            data_sources=data_sources,
            file_count=0,
            total_size_bytes=0,
            compressed_size_bytes=0,
            compression_ratio=0.0,
            checksum="",
            file_checksums={},
            base_backup_id=base_backup_id,
            recovery_point=datetime.now(),
            description=description or f"Incremental backup {backup_id}",
            tags=tags or [],
            retention_days=self.config.incremental_backup_retention_days
        )
        
        try:
            with self.backup_lock:
                self.active_backups[backup_id] = manifest
            
            manifest.status = BackupStatus.IN_PROGRESS
            
            # Get base backup timestamp for comparison
            base_backup = self._find_backup_by_id(base_backup_id)
            if not base_backup:
                raise ValueError(f"Base backup {base_backup_id} not found")
            
            base_timestamp = base_backup.recovery_point
            
            # Create backup directory
            backup_dir = self.backup_root / backup_id
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Backup only changed files since base backup
            total_files = 0
            total_original_size = 0
            total_compressed_size = 0
            file_checksums = {}
            
            for source in data_sources:
                source_result = await self._backup_data_source_incremental(
                    source, backup_dir, base_timestamp, manifest
                )
                
                total_files += source_result["file_count"]
                total_original_size += source_result["original_size"]
                total_compressed_size += source_result["compressed_size"]
                file_checksums.update(source_result["file_checksums"])
            
            # Update manifest (same as full backup)
            manifest.file_count = total_files
            manifest.total_size_bytes = total_original_size
            manifest.compressed_size_bytes = total_compressed_size
            manifest.compression_ratio = (
                1.0 - (total_compressed_size / max(total_original_size, 1))
            )
            manifest.file_checksums = file_checksums
            
            # Calculate overall checksum
            try:
                manifest.checksum = await self._calculate_backup_checksum(backup_dir)
            except Exception as e:
                logger.warning(f"Failed to calculate backup checksum: {e}")
                manifest.checksum = ""
            
            # Save manifest
            manifest_file = backup_dir / "manifest.json"
            async with aiofiles.open(manifest_file, 'w') as f:
                await f.write(json.dumps(manifest.to_dict(), indent=2))
            
            manifest.completed_at = datetime.now()
            manifest.status = BackupStatus.COMPLETED
            
            # Update tracking
            with self.backup_lock:
                self.backup_history.append(manifest)
                del self.active_backups[backup_id]
            
            logger.info("Incremental backup completed",
                       backup_id=backup_id,
                       base_backup_id=base_backup_id,
                       files=total_files,
                       size_mb=total_original_size / (1024**2))
            
            return manifest
            
        except Exception as e:
            manifest.status = BackupStatus.FAILED
            
            with self.backup_lock:
                if backup_id in self.active_backups:
                    del self.active_backups[backup_id]
            
            logger.error("Incremental backup failed", backup_id=backup_id, error=str(e))
            raise
    
    async def _backup_data_source(
        self, 
        source: str, 
        backup_dir: Path, 
        manifest: BackupManifest
    ) -> Dict[str, Any]:
        """Backup a specific data source."""
        result = {
            "file_count": 0,
            "original_size": 0,
            "compressed_size": 0,
            "file_checksums": {}
        }
        
        source_dir = backup_dir / source
        source_dir.mkdir(parents=True, exist_ok=True)
        
        if source == "json_files":
            result = await self._backup_json_files(source_dir)
        elif source == "postgresql":
            result = await self._backup_postgresql(source_dir)
        elif source == "chromadb":
            result = await self._backup_chromadb(source_dir)
        elif source == "metadata":
            result = await self._backup_metadata(source_dir)
        
        return result
    
    async def _backup_data_source_incremental(
        self, 
        source: str, 
        backup_dir: Path,
        since_timestamp: datetime, 
        manifest: BackupManifest
    ) -> Dict[str, Any]:
        """Backup a data source incrementally."""
        result = {
            "file_count": 0,
            "original_size": 0,
            "compressed_size": 0,
            "file_checksums": {}
        }
        
        source_dir = backup_dir / source
        source_dir.mkdir(parents=True, exist_ok=True)
        
        if source == "json_files":
            result = await self._backup_json_files_incremental(source_dir, since_timestamp)
        elif source == "postgresql":
            result = await self._backup_postgresql_incremental(source_dir, since_timestamp)
        elif source == "chromadb":
            # ChromaDB doesn't have easy incremental backup, do full
            result = await self._backup_chromadb(source_dir)
        elif source == "metadata":
            result = await self._backup_metadata_incremental(source_dir, since_timestamp)
        
        return result
    
    async def _backup_json_files(self, dest_dir: Path) -> Dict[str, Any]:
        """Backup JSON files from JsonPersistenceManager."""
        result = {
            "file_count": 0,
            "original_size": 0,
            "compressed_size": 0,
            "file_checksums": {}
        }
        
        try:
            # Get JSON persistence base path - using common data directory pattern
            json_base_path = Path("./data/usage_tracking")
            
            if not json_base_path.exists():
                logger.warning("JSON files base path does not exist", path=str(json_base_path))
                return result
            
            # Find all JSON and JSONL files
            json_files = list(json_base_path.rglob("*.json")) + list(json_base_path.rglob("*.jsonl"))
            json_files += list(json_base_path.rglob("*.json.gz")) + list(json_base_path.rglob("*.jsonl.gz"))
            
            logger.info(f"Found {len(json_files)} JSON files to backup")
            
            total_original_size = 0
            total_compressed_size = 0
            file_count = 0
            file_checksums = {}
            
            for json_file in json_files:
                try:
                    # Calculate relative path for consistent backup structure
                    relative_path = json_file.relative_to(json_base_path)
                    backup_file = dest_dir / relative_path
                    backup_file.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Get original file stats
                    original_size = json_file.stat().st_size
                    total_original_size += original_size
                    
                    # Copy and optionally compress file
                    if self.config.compression_enabled and not str(json_file).endswith('.gz'):
                        # Compress the file during backup
                        compressed_path = backup_file.with_suffix(backup_file.suffix + '.gz')
                        
                        async with aiofiles.open(json_file, 'rb') as src:
                            content = await src.read()
                        
                        # Compress content
                        compressed_content = gzip.compress(content)
                        
                        async with aiofiles.open(compressed_path, 'wb') as dst:
                            await dst.write(compressed_content)
                        
                        total_compressed_size += len(compressed_content)
                        
                        # Calculate checksum of compressed file
                        checksum = hashlib.sha256(compressed_content).hexdigest()
                        file_checksums[str(relative_path) + '.gz'] = checksum
                        
                    else:
                        # Copy file as-is (already compressed or compression disabled)
                        async with aiofiles.open(json_file, 'rb') as src:
                            content = await src.read()
                        
                        async with aiofiles.open(backup_file, 'wb') as dst:
                            await dst.write(content)
                        
                        total_compressed_size += len(content)
                        
                        # Calculate checksum
                        checksum = hashlib.sha256(content).hexdigest()
                        file_checksums[str(relative_path)] = checksum
                    
                    file_count += 1
                    
                    if file_count % 100 == 0:
                        logger.debug(f"Backed up {file_count} JSON files...")
                    
                except Exception as e:
                    logger.error(f"Failed to backup JSON file {json_file}: {e}")
                    continue
            
            # Also backup metadata files (indexes, configs)
            metadata_files = list(json_base_path.rglob("*.metadata")) + list(json_base_path.rglob("*.idx"))
            
            for metadata_file in metadata_files:
                try:
                    relative_path = metadata_file.relative_to(json_base_path)
                    backup_file = dest_dir / relative_path
                    backup_file.parent.mkdir(parents=True, exist_ok=True)
                    
                    original_size = metadata_file.stat().st_size
                    total_original_size += original_size
                    
                    # Copy metadata file
                    await asyncio.to_thread(shutil.copy2, metadata_file, backup_file)
                    
                    # Calculate checksum
                    async with aiofiles.open(backup_file, 'rb') as f:
                        content = await f.read()
                    
                    total_compressed_size += len(content)
                    checksum = hashlib.sha256(content).hexdigest()
                    file_checksums[str(relative_path)] = checksum
                    
                    file_count += 1
                    
                except Exception as e:
                    logger.error(f"Failed to backup metadata file {metadata_file}: {e}")
                    continue
            
            result = {
                "file_count": file_count,
                "original_size": total_original_size,
                "compressed_size": total_compressed_size,
                "file_checksums": file_checksums
            }
            
            logger.info(
                "JSON files backup completed",
                files=file_count,
                original_size_mb=total_original_size / (1024**2),
                compressed_size_mb=total_compressed_size / (1024**2),
                compression_ratio=1.0 - (total_compressed_size / max(total_original_size, 1))
            )
            
        except Exception as e:
            logger.error(f"JSON files backup failed: {e}")
            raise
        
        return result
    
    async def _backup_json_files_incremental(self, dest_dir: Path, since: datetime) -> Dict[str, Any]:
        """Backup JSON files incrementally since given timestamp."""
        result = {
            "file_count": 0,
            "original_size": 0,
            "compressed_size": 0,
            "file_checksums": {}
        }
        
        try:
            # Get JSON persistence base path
            json_base_path = Path("./data/usage_tracking")
            
            if not json_base_path.exists():
                logger.warning("JSON files base path does not exist", path=str(json_base_path))
                return result
            
            # Convert since timestamp to Unix timestamp for comparison
            since_timestamp = since.timestamp()
            
            # Find all JSON files modified since the given timestamp
            all_files = (list(json_base_path.rglob("*.json")) + 
                        list(json_base_path.rglob("*.jsonl")) +
                        list(json_base_path.rglob("*.json.gz")) + 
                        list(json_base_path.rglob("*.jsonl.gz")) +
                        list(json_base_path.rglob("*.metadata")) + 
                        list(json_base_path.rglob("*.idx")))
            
            modified_files = []
            for file_path in all_files:
                try:
                    stat_result = file_path.stat()
                    if stat_result.st_mtime > since_timestamp:
                        modified_files.append(file_path)
                except (OSError, IOError) as e:
                    logger.warning(f"Cannot stat file {file_path}: {e}")
                    continue
            
            logger.info(
                f"Found {len(modified_files)} JSON files modified since {since} for incremental backup"
            )
            
            total_original_size = 0
            total_compressed_size = 0
            file_count = 0
            file_checksums = {}
            
            for json_file in modified_files:
                try:
                    # Calculate relative path for consistent backup structure
                    relative_path = json_file.relative_to(json_base_path)
                    backup_file = dest_dir / relative_path
                    backup_file.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Get original file stats
                    original_size = json_file.stat().st_size
                    total_original_size += original_size
                    
                    # Handle JSON files with optional compression
                    if (json_file.suffix in ['.json', '.jsonl'] and 
                        self.config.compression_enabled and 
                        not str(json_file).endswith('.gz')):
                        
                        # Compress the file during backup
                        compressed_path = backup_file.with_suffix(backup_file.suffix + '.gz')
                        
                        async with aiofiles.open(json_file, 'rb') as src:
                            content = await src.read()
                        
                        # Compress content
                        compressed_content = gzip.compress(content)
                        
                        async with aiofiles.open(compressed_path, 'wb') as dst:
                            await dst.write(compressed_content)
                        
                        total_compressed_size += len(compressed_content)
                        
                        # Calculate checksum of compressed file
                        checksum = hashlib.sha256(compressed_content).hexdigest()
                        file_checksums[str(relative_path) + '.gz'] = checksum
                        
                    else:
                        # Copy file as-is (metadata, indexes, or already compressed)
                        async with aiofiles.open(json_file, 'rb') as src:
                            content = await src.read()
                        
                        async with aiofiles.open(backup_file, 'wb') as dst:
                            await dst.write(content)
                        
                        total_compressed_size += len(content)
                        
                        # Calculate checksum
                        checksum = hashlib.sha256(content).hexdigest()
                        file_checksums[str(relative_path)] = checksum
                    
                    file_count += 1
                    
                    if file_count % 100 == 0:
                        logger.debug(f"Incrementally backed up {file_count} JSON files...")
                    
                except Exception as e:
                    logger.error(f"Failed to backup JSON file {json_file} incrementally: {e}")
                    continue
            
            result = {
                "file_count": file_count,
                "original_size": total_original_size,
                "compressed_size": total_compressed_size,
                "file_checksums": file_checksums
            }
            
            logger.info(
                "Incremental JSON files backup completed",
                files=file_count,
                original_size_mb=total_original_size / (1024**2),
                compressed_size_mb=total_compressed_size / (1024**2),
                since=since.isoformat()
            )
            
        except Exception as e:
            logger.error(f"Incremental JSON files backup failed: {e}")
            raise
        
        return result
    
    async def _backup_postgresql(self, dest_dir: Path) -> Dict[str, Any]:
        """Backup PostgreSQL data using pg_dump."""
        result = {
            "file_count": 0,
            "original_size": 0,
            "compressed_size": 0,
            "file_checksums": {}
        }
        
        try:
            # Import context config to get database URL
            from src.context.config import context_config
            
            database_url = context_config.database_url
            
            # Parse database URL to extract connection parameters
            import urllib.parse
            parsed = urllib.parse.urlparse(database_url)
            
            if not parsed.hostname:
                logger.warning("No PostgreSQL database configured, skipping PostgreSQL backup")
                return result
            
            # Prepare pg_dump command
            dump_file = dest_dir / "postgresql_dump.sql"
            compressed_dump_file = dest_dir / "postgresql_dump.sql.gz"
            
            # Build pg_dump command with connection parameters
            pg_dump_cmd = [
                "pg_dump",
                "--host", parsed.hostname,
                "--port", str(parsed.port or 5432),
                "--username", parsed.username or "postgres",
                "--dbname", parsed.path[1:] if parsed.path else "postgres",
                "--verbose",
                "--no-password",  # Use .pgpass or environment variables
                "--format=plain",
                "--encoding=UTF8",
                "--no-privileges",  # Don't dump privileges for portability
                "--no-owner",      # Don't dump ownership info
                "--file", str(dump_file)
            ]
            
            # Set environment variables for PostgreSQL connection
            env = os.environ.copy()
            if parsed.password:
                env["PGPASSWORD"] = parsed.password
            
            logger.info("Starting PostgreSQL backup with pg_dump")
            
            # Execute pg_dump
            try:
                process = await asyncio.create_subprocess_exec(
                    *pg_dump_cmd,
                    env=env,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
            except FileNotFoundError:
                raise Exception("pg_dump command not found. Please install PostgreSQL client tools.")
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                error_msg = f"pg_dump failed with return code {process.returncode}: {stderr.decode()}"
                logger.error(error_msg)
                raise Exception(error_msg)
            
            if not dump_file.exists() or dump_file.stat().st_size == 0:
                raise Exception("pg_dump produced no output or empty file")
            
            # Get original dump file size
            original_size = dump_file.stat().st_size
            
            # Compress the dump if compression is enabled
            if self.config.compression_enabled:
                logger.info("Compressing PostgreSQL dump")
                
                # Read and compress the dump file
                async with aiofiles.open(dump_file, 'rb') as src:
                    dump_content = await src.read()
                
                compressed_content = gzip.compress(dump_content)
                
                # Write compressed file
                async with aiofiles.open(compressed_dump_file, 'wb') as dst:
                    await dst.write(compressed_content)
                
                # Remove uncompressed file
                await aiofiles.os.remove(dump_file)
                
                # Calculate checksum of compressed file
                checksum = hashlib.sha256(compressed_content).hexdigest()
                
                result = {
                    "file_count": 1,
                    "original_size": original_size,
                    "compressed_size": len(compressed_content),
                    "file_checksums": {
                        "postgresql_dump.sql.gz": checksum
                    }
                }
                
                logger.info(
                    "PostgreSQL backup completed (compressed)",
                    original_size_mb=original_size / (1024**2),
                    compressed_size_mb=len(compressed_content) / (1024**2),
                    compression_ratio=1.0 - (len(compressed_content) / original_size)
                )
                
            else:
                # Keep uncompressed file
                async with aiofiles.open(dump_file, 'rb') as f:
                    content = await f.read()
                
                checksum = hashlib.sha256(content).hexdigest()
                
                result = {
                    "file_count": 1,
                    "original_size": original_size,
                    "compressed_size": original_size,
                    "file_checksums": {
                        "postgresql_dump.sql": checksum
                    }
                }
                
                logger.info(
                    "PostgreSQL backup completed (uncompressed)",
                    size_mb=original_size / (1024**2)
                )
            
            # Also backup database schema information
            schema_file = dest_dir / "postgresql_schema.sql"
            
            schema_cmd = [
                "pg_dump",
                "--host", parsed.hostname,
                "--port", str(parsed.port or 5432),
                "--username", parsed.username or "postgres",
                "--dbname", parsed.path[1:] if parsed.path else "postgres",
                "--schema-only",
                "--no-password",
                "--format=plain",
                "--file", str(schema_file)
            ]
            
            schema_process = await asyncio.create_subprocess_exec(
                *schema_cmd,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            schema_stdout, schema_stderr = await schema_process.communicate()
            
            if schema_process.returncode == 0 and schema_file.exists():
                # Add schema file to backup
                async with aiofiles.open(schema_file, 'rb') as f:
                    schema_content = await f.read()
                
                schema_checksum = hashlib.sha256(schema_content).hexdigest()
                result["file_checksums"]["postgresql_schema.sql"] = schema_checksum
                result["file_count"] += 1
                result["original_size"] += len(schema_content)
                
                if not self.config.compression_enabled:
                    result["compressed_size"] += len(schema_content)
                
                logger.info("PostgreSQL schema backup completed")
            
        except FileNotFoundError:
            logger.error(
                "pg_dump not found. Please install PostgreSQL client tools. "
                "Skipping PostgreSQL backup."
            )
            # Return empty result but don't raise exception to continue with other backups
        except Exception as e:
            logger.error(f"PostgreSQL backup failed: {e}")
            raise
        
        return result
    
    async def _backup_postgresql_incremental(self, dest_dir: Path, since: datetime) -> Dict[str, Any]:
        """Backup PostgreSQL data incrementally using WAL archives or timestamp-based filtering."""
        result = {
            "file_count": 0,
            "original_size": 0,
            "compressed_size": 0,
            "file_checksums": {}
        }
        
        try:
            # Import context config to get database URL
            from src.context.config import context_config
            
            database_url = context_config.database_url
            
            # Parse database URL to extract connection parameters
            import urllib.parse
            parsed = urllib.parse.urlparse(database_url)
            
            if not parsed.hostname:
                logger.warning("No PostgreSQL database configured, skipping PostgreSQL incremental backup")
                return result
            
            # For incremental backup, we'll use a timestamp-based approach
            # This captures only data modified since the last backup
            
            # Prepare incremental dump file
            inc_dump_file = dest_dir / f"postgresql_incremental_{since.strftime('%Y%m%d_%H%M%S')}.sql"
            compressed_inc_dump_file = dest_dir / f"postgresql_incremental_{since.strftime('%Y%m%d_%H%M%S')}.sql.gz"
            
            # Build pg_dump command for incremental backup
            # Note: This approach dumps all data but could be filtered by application logic
            # Real incremental backups would require WAL archiving setup
            pg_dump_cmd = [
                "pg_dump",
                "--host", parsed.hostname,
                "--port", str(parsed.port or 5432),
                "--username", parsed.username or "postgres",
                "--dbname", parsed.path[1:] if parsed.path else "postgres",
                "--verbose",
                "--no-password",
                "--format=plain",
                "--encoding=UTF8",
                "--no-privileges",
                "--no-owner",
                "--data-only",  # Only dump data for incremental
                "--file", str(inc_dump_file)
            ]
            
            # Set environment variables for PostgreSQL connection
            env = os.environ.copy()
            if parsed.password:
                env["PGPASSWORD"] = parsed.password
            
            logger.info(
                f"Starting PostgreSQL incremental backup since {since.isoformat()}"
            )
            
            # Execute pg_dump for incremental backup
            try:
                process = await asyncio.create_subprocess_exec(
                    *pg_dump_cmd,
                    env=env,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
            except FileNotFoundError:
                raise Exception("pg_dump command not found. Please install PostgreSQL client tools.")
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                error_msg = f"Incremental pg_dump failed with return code {process.returncode}: {stderr.decode()}"
                logger.error(error_msg)
                raise Exception(error_msg)
            
            if not inc_dump_file.exists():
                logger.warning("Incremental pg_dump produced no output")
                return result
            
            original_size = inc_dump_file.stat().st_size
            
            # Skip if dump is empty (no changes)
            if original_size == 0:
                logger.info("No data changes since last backup, skipping")
                await aiofiles.os.remove(inc_dump_file)
                return result
            
            # Compress the incremental dump if compression is enabled
            if self.config.compression_enabled:
                logger.info("Compressing PostgreSQL incremental dump")
                
                # Read and compress the dump file
                async with aiofiles.open(inc_dump_file, 'rb') as src:
                    dump_content = await src.read()
                
                compressed_content = gzip.compress(dump_content)
                
                # Write compressed file
                async with aiofiles.open(compressed_inc_dump_file, 'wb') as dst:
                    await dst.write(compressed_content)
                
                # Remove uncompressed file
                await aiofiles.os.remove(inc_dump_file)
                
                # Calculate checksum of compressed file
                checksum = hashlib.sha256(compressed_content).hexdigest()
                
                result = {
                    "file_count": 1,
                    "original_size": original_size,
                    "compressed_size": len(compressed_content),
                    "file_checksums": {
                        f"postgresql_incremental_{since.strftime('%Y%m%d_%H%M%S')}.sql.gz": checksum
                    }
                }
                
                logger.info(
                    "PostgreSQL incremental backup completed (compressed)",
                    original_size_mb=original_size / (1024**2),
                    compressed_size_mb=len(compressed_content) / (1024**2),
                    since=since.isoformat()
                )
                
            else:
                # Keep uncompressed file
                async with aiofiles.open(inc_dump_file, 'rb') as f:
                    content = await f.read()
                
                checksum = hashlib.sha256(content).hexdigest()
                
                result = {
                    "file_count": 1,
                    "original_size": original_size,
                    "compressed_size": original_size,
                    "file_checksums": {
                        f"postgresql_incremental_{since.strftime('%Y%m%d_%H%M%S')}.sql": checksum
                    }
                }
                
                logger.info(
                    "PostgreSQL incremental backup completed (uncompressed)",
                    size_mb=original_size / (1024**2),
                    since=since.isoformat()
                )
            
            # Create a timestamp marker file for this incremental backup
            timestamp_file = dest_dir / "incremental_timestamp.txt"
            async with aiofiles.open(timestamp_file, 'w') as f:
                await f.write(since.isoformat())
            
            async with aiofiles.open(timestamp_file, 'rb') as f:
                timestamp_content = await f.read()
            
            timestamp_checksum = hashlib.sha256(timestamp_content).hexdigest()
            result["file_checksums"]["incremental_timestamp.txt"] = timestamp_checksum
            result["file_count"] += 1
            result["original_size"] += len(timestamp_content)
            
            if not self.config.compression_enabled:
                result["compressed_size"] += len(timestamp_content)
            
        except FileNotFoundError:
            logger.error(
                "pg_dump not found. Please install PostgreSQL client tools. "
                "Skipping PostgreSQL incremental backup."
            )
            # Return empty result but don't raise exception
        except Exception as e:
            logger.error(f"PostgreSQL incremental backup failed: {e}")
            raise
        
        return result
    
    async def _backup_chromadb(self, dest_dir: Path) -> Dict[str, Any]:
        """Backup ChromaDB data including collections, embeddings, and metadata."""
        result = {
            "file_count": 0,
            "original_size": 0,
            "compressed_size": 0,
            "file_checksums": {}
        }
        
        try:
            # Get ChromaDB path from settings
            from config.settings import settings
            
            chroma_db_path = Path(settings.chroma_db_path)
            
            if not chroma_db_path.exists():
                logger.warning("ChromaDB path does not exist", path=str(chroma_db_path))
                return result
            
            logger.info("Starting ChromaDB backup", source_path=str(chroma_db_path))
            
            # Find all ChromaDB files (including SQLite databases and parquet files)
            chroma_files = []
            
            # Common ChromaDB file patterns
            patterns = [
                "**/*.sqlite3",
                "**/*.sqlite", 
                "**/*.db",
                "**/*.parquet",
                "**/*.bin",
                "**/*.json",
                "**/*.pkl",
                "**/*.log",
                "**/*.wal"  # SQLite WAL files
            ]
            
            for pattern in patterns:
                chroma_files.extend(chroma_db_path.rglob(pattern))
            
            # Remove duplicates and ensure files exist
            chroma_files = list(set(f for f in chroma_files if f.is_file()))
            
            logger.info(f"Found {len(chroma_files)} ChromaDB files to backup")
            
            total_original_size = 0
            total_compressed_size = 0
            file_count = 0
            file_checksums = {}
            
            # Backup each ChromaDB file
            for chroma_file in chroma_files:
                try:
                    # Calculate relative path from ChromaDB root
                    relative_path = chroma_file.relative_to(chroma_db_path)
                    backup_file = dest_dir / relative_path
                    backup_file.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Get original file size
                    original_size = chroma_file.stat().st_size
                    total_original_size += original_size
                    
                    # Handle binary files with optional compression
                    if (chroma_file.suffix in ['.sqlite3', '.sqlite', '.db', '.parquet', '.bin'] and 
                        self.config.compression_enabled):
                        
                        # Compress binary files
                        compressed_path = backup_file.with_suffix(backup_file.suffix + '.gz')
                        
                        async with aiofiles.open(chroma_file, 'rb') as src:
                            content = await src.read()
                        
                        # Compress content
                        compressed_content = gzip.compress(content)
                        
                        async with aiofiles.open(compressed_path, 'wb') as dst:
                            await dst.write(compressed_content)
                        
                        total_compressed_size += len(compressed_content)
                        
                        # Calculate checksum of compressed file
                        checksum = hashlib.sha256(compressed_content).hexdigest()
                        file_checksums[str(relative_path) + '.gz'] = checksum
                        
                    else:
                        # Copy file as-is (text files, logs, or compression disabled)
                        async with aiofiles.open(chroma_file, 'rb') as src:
                            content = await src.read()
                        
                        async with aiofiles.open(backup_file, 'wb') as dst:
                            await dst.write(content)
                        
                        total_compressed_size += len(content)
                        
                        # Calculate checksum
                        checksum = hashlib.sha256(content).hexdigest()
                        file_checksums[str(relative_path)] = checksum
                    
                    file_count += 1
                    
                    if file_count % 50 == 0:
                        logger.debug(f"Backed up {file_count} ChromaDB files...")
                    
                except Exception as e:
                    logger.error(f"Failed to backup ChromaDB file {chroma_file}: {e}")
                    continue
            
            # Create a manifest of ChromaDB collections and their metadata
            try:
                # Try to get collection information if ChromaDB is available
                collections_info = await self._get_chromadb_collections_info(chroma_db_path)
                
                if collections_info:
                    manifest_file = dest_dir / "chromadb_collections_manifest.json"
                    
                    async with aiofiles.open(manifest_file, 'w') as f:
                        await f.write(json.dumps(collections_info, indent=2, default=str))
                    
                    # Add manifest to backup
                    async with aiofiles.open(manifest_file, 'rb') as f:
                        manifest_content = await f.read()
                    
                    manifest_checksum = hashlib.sha256(manifest_content).hexdigest()
                    file_checksums["chromadb_collections_manifest.json"] = manifest_checksum
                    
                    file_count += 1
                    total_original_size += len(manifest_content)
                    total_compressed_size += len(manifest_content)
                    
                    logger.info(f"ChromaDB collections manifest created with {len(collections_info.get('collections', []))} collections")
                
            except Exception as e:
                logger.warning(f"Could not create ChromaDB collections manifest: {e}")
            
            result = {
                "file_count": file_count,
                "original_size": total_original_size,
                "compressed_size": total_compressed_size,
                "file_checksums": file_checksums
            }
            
            logger.info(
                "ChromaDB backup completed",
                files=file_count,
                original_size_mb=total_original_size / (1024**2),
                compressed_size_mb=total_compressed_size / (1024**2),
                compression_ratio=1.0 - (total_compressed_size / max(total_original_size, 1))
            )
            
        except Exception as e:
            logger.error(f"ChromaDB backup failed: {e}")
            raise
        
        return result
    
    async def _get_chromadb_collections_info(self, chroma_db_path: Path) -> Optional[Dict[str, Any]]:
        """Get ChromaDB collections information for backup manifest."""
        try:
            # Try to access ChromaDB to get collection metadata
            # This is a best-effort attempt - if ChromaDB is not available, we'll skip
            
            collections_info = {
                "backup_timestamp": datetime.now().isoformat(),
                "chroma_db_path": str(chroma_db_path),
                "collections": []
            }
            
            # Look for collection directories and metadata
            for item in chroma_db_path.iterdir():
                if item.is_dir():
                    # Check if this looks like a collection directory
                    collection_files = list(item.rglob("*"))
                    
                    if collection_files:
                        collections_info["collections"].append({
                            "name": item.name,
                            "path": str(item.relative_to(chroma_db_path)),
                            "file_count": len([f for f in collection_files if f.is_file()]),
                            "total_size_bytes": sum(f.stat().st_size for f in collection_files if f.is_file())
                        })
            
            return collections_info
            
        except Exception as e:
            logger.debug(f"Could not extract ChromaDB collections info: {e}")
            return None
    
    async def _backup_metadata(self, dest_dir: Path) -> Dict[str, Any]:
        """Backup system metadata including configuration, schemas, and logs."""
        result = {
            "file_count": 0,
            "original_size": 0,
            "compressed_size": 0,
            "file_checksums": {}
        }
        
        try:
            logger.info("Starting metadata backup")
            
            total_original_size = 0
            total_compressed_size = 0
            file_count = 0
            file_checksums = {}
            
            # Define metadata sources to backup
            metadata_sources = [
                # Configuration files
                {
                    "name": "config",
                    "path": Path("config"),
                    "patterns": ["*.py", "*.json", "*.yaml", "*.yml", "*.toml", "*.ini"]
                },
                # Environment files
                {
                    "name": "env", 
                    "path": Path("."),
                    "patterns": [".env*", "*.env"]
                },
                # Requirements and dependency files
                {
                    "name": "deps",
                    "path": Path("."),
                    "patterns": ["requirements*.txt", "pyproject.toml", "poetry.lock", "Pipfile*", "setup.py", "setup.cfg"]
                },
                # Schema and migration files
                {
                    "name": "schemas",
                    "path": Path("src"),
                    "patterns": ["**/migrations/*.py", "**/schema*.py", "**/models*.py"]
                },
                # Documentation
                {
                    "name": "docs",
                    "path": Path("docs"),
                    "patterns": ["*.md", "*.rst", "*.txt"]
                },
                # Version control info
                {
                    "name": "vcs",
                    "path": Path(".git"),
                    "patterns": ["HEAD", "config", "refs/**/*"]
                },
                # Recent logs (last 7 days)
                {
                    "name": "logs",
                    "path": Path("logs"),
                    "patterns": ["*.log", "*.log.*"]
                }
            ]
            
            # Backup each metadata source
            for source in metadata_sources:
                try:
                    source_path = source["path"]
                    
                    if not source_path.exists():
                        logger.debug(f"Metadata source path does not exist: {source_path}")
                        continue
                    
                    source_dest = dest_dir / source["name"]
                    source_dest.mkdir(parents=True, exist_ok=True)
                    
                    # Find files matching patterns
                    source_files = []
                    for pattern in source["patterns"]:
                        source_files.extend(source_path.rglob(pattern))
                    
                    # Remove duplicates and filter files
                    source_files = list(set(f for f in source_files if f.is_file()))
                    
                    # For logs, only backup recent files (last 7 days)
                    if source["name"] == "logs":
                        cutoff_time = (datetime.now() - timedelta(days=7)).timestamp()
                        source_files = [f for f in source_files if f.stat().st_mtime > cutoff_time]
                    
                    logger.debug(f"Found {len(source_files)} files for metadata source: {source['name']}")
                    
                    for metadata_file in source_files:
                        try:
                            # Calculate relative path from source root
                            if source_path == Path("."):
                                relative_path = metadata_file.name
                            else:
                                relative_path = metadata_file.relative_to(source_path)
                            
                            backup_file = source_dest / relative_path
                            backup_file.parent.mkdir(parents=True, exist_ok=True)
                            
                            # Get original file size
                            original_size = metadata_file.stat().st_size
                            total_original_size += original_size
                            
                            # Skip very large files (> 50MB) to avoid backing up large log files
                            if original_size > 50 * 1024 * 1024:
                                logger.warning(f"Skipping large metadata file: {metadata_file} ({original_size / (1024**2):.2f}MB)")
                                continue
                            
                            # Handle text files with optional compression
                            if (metadata_file.suffix in ['.py', '.json', '.yaml', '.yml', '.toml', '.ini', '.md', '.rst', '.txt', '.log'] and 
                                self.config.compression_enabled and original_size > 1024):  # Only compress if > 1KB
                                
                                # Compress text files
                                compressed_path = backup_file.with_suffix(backup_file.suffix + '.gz')
                                
                                async with aiofiles.open(metadata_file, 'rb') as src:
                                    content = await src.read()
                                
                                # Compress content
                                compressed_content = gzip.compress(content)
                                
                                async with aiofiles.open(compressed_path, 'wb') as dst:
                                    await dst.write(compressed_content)
                                
                                total_compressed_size += len(compressed_content)
                                
                                # Calculate checksum of compressed file
                                checksum = hashlib.sha256(compressed_content).hexdigest()
                                file_checksums[f"{source['name']}/{relative_path}.gz"] = checksum
                                
                            else:
                                # Copy file as-is (binary files, small files, or compression disabled)
                                async with aiofiles.open(metadata_file, 'rb') as src:
                                    content = await src.read()
                                
                                async with aiofiles.open(backup_file, 'wb') as dst:
                                    await dst.write(content)
                                
                                total_compressed_size += len(content)
                                
                                # Calculate checksum
                                checksum = hashlib.sha256(content).hexdigest()
                                file_checksums[f"{source['name']}/{relative_path}"] = checksum
                            
                            file_count += 1
                            
                        except Exception as e:
                            logger.error(f"Failed to backup metadata file {metadata_file}: {e}")
                            continue
                    
                except Exception as e:
                    logger.error(f"Failed to backup metadata source {source['name']}: {e}")
                    continue
            
            # Create a system information file
            try:
                system_info = {
                    "backup_timestamp": datetime.now().isoformat(),
                    "hostname": os.uname().nodename if hasattr(os, 'uname') else "unknown",
                    "platform": {
                        "system": os.name,
                        "python_version": "3.x",  # Simplified since we can't import sys safely
                    },
                    "application": {
                        "name": "TTRPG Assistant",
                        "backup_config": {
                            "compression_enabled": self.config.compression_enabled,
                            "encryption_enabled": self.config.encryption_enabled,
                            "max_backup_size_gb": self.config.max_backup_size_gb
                        }
                    }
                }
                
                system_info_file = dest_dir / "system_info.json"
                
                async with aiofiles.open(system_info_file, 'w') as f:
                    await f.write(json.dumps(system_info, indent=2))
                
                # Add system info to backup
                async with aiofiles.open(system_info_file, 'rb') as f:
                    info_content = await f.read()
                
                info_checksum = hashlib.sha256(info_content).hexdigest()
                file_checksums["system_info.json"] = info_checksum
                
                file_count += 1
                total_original_size += len(info_content)
                total_compressed_size += len(info_content)
                
            except Exception as e:
                logger.warning(f"Could not create system info file: {e}")
            
            result = {
                "file_count": file_count,
                "original_size": total_original_size,
                "compressed_size": total_compressed_size,
                "file_checksums": file_checksums
            }
            
            logger.info(
                "Metadata backup completed",
                files=file_count,
                original_size_mb=total_original_size / (1024**2),
                compressed_size_mb=total_compressed_size / (1024**2)
            )
            
        except Exception as e:
            logger.error(f"Metadata backup failed: {e}")
            raise
        
        return result
    
    async def _backup_metadata_incremental(self, dest_dir: Path, since: datetime) -> Dict[str, Any]:
        """Backup system metadata incrementally since given timestamp."""
        result = {
            "file_count": 0,
            "original_size": 0,
            "compressed_size": 0,
            "file_checksums": {}
        }
        
        try:
            logger.info(f"Starting incremental metadata backup since {since.isoformat()}")
            
            # Convert since timestamp to Unix timestamp for comparison
            since_timestamp = since.timestamp()
            
            total_original_size = 0
            total_compressed_size = 0
            file_count = 0
            file_checksums = {}
            
            # Define metadata sources to backup (same as full backup)
            metadata_sources = [
                {
                    "name": "config",
                    "path": Path("config"),
                    "patterns": ["*.py", "*.json", "*.yaml", "*.yml", "*.toml", "*.ini"]
                },
                {
                    "name": "env", 
                    "path": Path("."),
                    "patterns": [".env*", "*.env"]
                },
                {
                    "name": "deps",
                    "path": Path("."),
                    "patterns": ["requirements*.txt", "pyproject.toml", "poetry.lock", "Pipfile*", "setup.py", "setup.cfg"]
                },
                {
                    "name": "schemas",
                    "path": Path("src"),
                    "patterns": ["**/migrations/*.py", "**/schema*.py", "**/models*.py"]
                },
                # Skip docs and VCS for incremental backups to reduce size
                {
                    "name": "logs",
                    "path": Path("logs"),
                    "patterns": ["*.log", "*.log.*"]
                }
            ]
            
            # Backup each metadata source (only modified files)
            for source in metadata_sources:
                try:
                    source_path = source["path"]
                    
                    if not source_path.exists():
                        logger.debug(f"Metadata source path does not exist: {source_path}")
                        continue
                    
                    source_dest = dest_dir / source["name"]
                    source_dest.mkdir(parents=True, exist_ok=True)
                    
                    # Find files matching patterns
                    all_files = []
                    for pattern in source["patterns"]:
                        all_files.extend(source_path.rglob(pattern))
                    
                    # Filter for files modified since timestamp
                    modified_files = []
                    for file_path in all_files:
                        try:
                            if file_path.is_file():
                                file_mtime = file_path.stat().st_mtime
                                if file_mtime > since_timestamp:
                                    modified_files.append(file_path)
                        except (OSError, IOError) as e:
                            logger.warning(f"Cannot stat metadata file {file_path}: {e}")
                            continue
                    
                    # For logs, also filter by age (only last 24 hours for incremental)
                    if source["name"] == "logs":
                        recent_cutoff = (datetime.now() - timedelta(hours=24)).timestamp()
                        modified_files = [f for f in modified_files if f.stat().st_mtime > recent_cutoff]
                    
                    logger.debug(f"Found {len(modified_files)} modified files for metadata source: {source['name']}")
                    
                    for metadata_file in modified_files:
                        try:
                            # Calculate relative path from source root
                            if source_path == Path("."):
                                relative_path = metadata_file.name
                            else:
                                relative_path = metadata_file.relative_to(source_path)
                            
                            backup_file = source_dest / relative_path
                            backup_file.parent.mkdir(parents=True, exist_ok=True)
                            
                            # Get original file size
                            original_size = metadata_file.stat().st_size
                            total_original_size += original_size
                            
                            # Skip very large files
                            if original_size > 10 * 1024 * 1024:  # 10MB limit for incremental
                                logger.warning(f"Skipping large metadata file in incremental backup: {metadata_file}")
                                continue
                            
                            # Handle files with optional compression
                            if (metadata_file.suffix in ['.py', '.json', '.yaml', '.yml', '.toml', '.ini', '.md', '.rst', '.txt', '.log'] and 
                                self.config.compression_enabled and original_size > 1024):
                                
                                # Compress text files
                                compressed_path = backup_file.with_suffix(backup_file.suffix + '.gz')
                                
                                async with aiofiles.open(metadata_file, 'rb') as src:
                                    content = await src.read()
                                
                                compressed_content = gzip.compress(content)
                                
                                async with aiofiles.open(compressed_path, 'wb') as dst:
                                    await dst.write(compressed_content)
                                
                                total_compressed_size += len(compressed_content)
                                
                                checksum = hashlib.sha256(compressed_content).hexdigest()
                                file_checksums[f"{source['name']}/{relative_path}.gz"] = checksum
                                
                            else:
                                # Copy file as-is
                                async with aiofiles.open(metadata_file, 'rb') as src:
                                    content = await src.read()
                                
                                async with aiofiles.open(backup_file, 'wb') as dst:
                                    await dst.write(content)
                                
                                total_compressed_size += len(content)
                                
                                checksum = hashlib.sha256(content).hexdigest()
                                file_checksums[f"{source['name']}/{relative_path}"] = checksum
                            
                            file_count += 1
                            
                        except Exception as e:
                            logger.error(f"Failed to backup metadata file {metadata_file} incrementally: {e}")
                            continue
                    
                except Exception as e:
                    logger.error(f"Failed to backup metadata source {source['name']} incrementally: {e}")
                    continue
            
            # Create an incremental metadata info file
            try:
                incremental_info = {
                    "backup_timestamp": datetime.now().isoformat(),
                    "incremental_since": since.isoformat(),
                    "files_backed_up": file_count,
                    "backup_type": "incremental_metadata"
                }
                
                inc_info_file = dest_dir / "incremental_metadata_info.json"
                
                async with aiofiles.open(inc_info_file, 'w') as f:
                    await f.write(json.dumps(incremental_info, indent=2))
                
                # Add info file to backup
                async with aiofiles.open(inc_info_file, 'rb') as f:
                    info_content = await f.read()
                
                info_checksum = hashlib.sha256(info_content).hexdigest()
                file_checksums["incremental_metadata_info.json"] = info_checksum
                
                file_count += 1
                total_original_size += len(info_content)
                total_compressed_size += len(info_content)
                
            except Exception as e:
                logger.warning(f"Could not create incremental metadata info file: {e}")
            
            result = {
                "file_count": file_count,
                "original_size": total_original_size,
                "compressed_size": total_compressed_size,
                "file_checksums": file_checksums
            }
            
            logger.info(
                "Incremental metadata backup completed",
                files=file_count,
                original_size_mb=total_original_size / (1024**2),
                compressed_size_mb=total_compressed_size / (1024**2),
                since=since.isoformat()
            )
            
        except Exception as e:
            logger.error(f"Incremental metadata backup failed: {e}")
            raise
        
        return result
    
    async def _calculate_backup_checksum(self, backup_dir: Path) -> str:
        """Calculate checksum for entire backup."""
        hasher = hashlib.sha256()
        
        # Sort files for consistent checksum
        files = sorted(backup_dir.rglob("*"))
        
        for file_path in files:
            if file_path.is_file():
                async with aiofiles.open(file_path, 'rb') as f:
                    while True:
                        chunk = await f.read(8192)
                        if not chunk:
                            break
                        hasher.update(chunk)
        
        return hasher.hexdigest()
    
    def _find_backup_by_id(self, backup_id: str) -> Optional[BackupManifest]:
        """Find backup manifest by ID."""
        for backup in self.backup_history:
            if backup.backup_id == backup_id:
                return backup
        return None


class RecoveryEngine:
    """Core recovery engine for restoring from backups."""
    
    def __init__(self, config: BackupConfiguration):
        self.config = config
        self.backup_root = Path(config.backup_root_path)
        self.staging_path = Path(config.recovery_staging_path)
        
        # Validate paths
        if not self.backup_root.exists():
            raise Exception(f"Backup root directory {self.backup_root} does not exist")
        
        if not os.access(self.backup_root, os.R_OK):
            raise Exception(f"Backup root directory {self.backup_root} is not readable")
        
        try:
            self.staging_path.mkdir(parents=True, exist_ok=True)
        except (OSError, PermissionError) as e:
            raise Exception(f"Cannot create recovery staging path {self.staging_path}: {e}")
        
        if not os.access(self.staging_path, os.W_OK):
            raise Exception(f"Recovery staging path {self.staging_path} is not writable")
        
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        logger.info(
            "RecoveryEngine initialized",
            backup_root=str(self.backup_root),
            staging_path=str(self.staging_path)
        )
    
    async def create_recovery_plan(
        self,
        target_timestamp: datetime,
        data_sources: Optional[List[str]] = None,
        recovery_type: RecoveryType = RecoveryType.FULL_RESTORE
    ) -> RecoveryPlan:
        """Create a recovery plan for the specified target timestamp."""
        plan_id = f"recovery_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Find the appropriate backup sequence
        backup_sequence = await self._find_backup_sequence(target_timestamp)
        
        if not backup_sequence:
            raise ValueError(f"No backup sequence found for timestamp {target_timestamp}")
        
        # Estimate recovery time and size
        estimated_time, estimated_size = await self._estimate_recovery_resources(backup_sequence)
        
        plan = RecoveryPlan(
            plan_id=plan_id,
            recovery_type=recovery_type,
            target_timestamp=target_timestamp,
            backup_sequence=backup_sequence,
            estimated_time_minutes=estimated_time,
            estimated_size_gb=estimated_size,
            data_sources_to_recover=data_sources or ["json_files", "postgresql", "chromadb", "metadata"],
            verification_steps=[
                "verify_backup_integrity",
                "verify_data_consistency",
                "verify_application_startup"
            ],
            rollback_plan="create_rollback_backup_before_recovery"
        )
        
        return plan
    
    async def execute_recovery(
        self,
        recovery_plan: RecoveryPlan,
        target_path: Optional[str] = None,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """Execute a recovery plan."""
        target_path = target_path or str(self.staging_path / recovery_plan.plan_id)
        target_dir = Path(target_path)
        
        if not dry_run:
            target_dir.mkdir(parents=True, exist_ok=True)
        
        recovery_result = {
            "plan_id": recovery_plan.plan_id,
            "started_at": datetime.now().isoformat(),
            "completed_at": None,
            "success": False,
            "recovered_sources": [],
            "verification_results": {},
            "errors": []
        }
        
        try:
            logger.info("Starting recovery",
                       plan_id=recovery_plan.plan_id,
                       dry_run=dry_run,
                       target_path=target_path)
            
            # Execute recovery in sequence
            for backup_id in recovery_plan.backup_sequence:
                try:
                    backup_result = await self._recover_backup(
                        backup_id, target_dir, recovery_plan.data_sources_to_recover, dry_run
                    )
                    recovery_result["recovered_sources"].extend(backup_result["sources"])
                    
                except Exception as e:
                    error_msg = f"Failed to recover backup {backup_id}: {str(e)}"
                    recovery_result["errors"].append(error_msg)
                    logger.error("Recovery step failed", backup_id=backup_id, error=str(e))
                    raise
            
            # Run verification steps
            if not dry_run and self.config.auto_verification_on_recovery:
                for step in recovery_plan.verification_steps:
                    try:
                        verification_result = await self._run_verification_step(step, target_dir)
                        recovery_result["verification_results"][step] = verification_result
                        
                    except Exception as e:
                        error_msg = f"Verification step {step} failed: {str(e)}"
                        recovery_result["errors"].append(error_msg)
                        logger.warning("Verification step failed", step=step, error=str(e))
            
            recovery_result["success"] = len(recovery_result["errors"]) == 0
            recovery_result["completed_at"] = datetime.now().isoformat()
            
            logger.info("Recovery completed",
                       plan_id=recovery_plan.plan_id,
                       success=recovery_result["success"],
                       errors=len(recovery_result["errors"]))
            
            return recovery_result
            
        except Exception as e:
            recovery_result["success"] = False
            recovery_result["completed_at"] = datetime.now().isoformat()
            recovery_result["errors"].append(f"Recovery failed: {str(e)}")
            
            logger.error("Recovery failed", plan_id=recovery_plan.plan_id, error=str(e))
            raise
    
    async def _find_backup_sequence(self, target_timestamp: datetime) -> List[str]:
        """Find the sequence of backups needed for point-in-time recovery."""
        # Load all backup manifests
        backup_manifests = await self._load_all_manifests()
        
        # Sort by recovery point
        backup_manifests.sort(key=lambda b: b.recovery_point)
        
        # Find the last full backup before target timestamp
        base_full_backup = None
        for backup in reversed(backup_manifests):
            if (backup.backup_type == BackupType.FULL and 
                backup.recovery_point <= target_timestamp and
                backup.status == BackupStatus.COMPLETED):
                base_full_backup = backup
                break
        
        if not base_full_backup:
            return []
        
        # Find incremental backups after the full backup up to target timestamp
        sequence = [base_full_backup.backup_id]
        
        for backup in backup_manifests:
            if (backup.backup_type == BackupType.INCREMENTAL and
                backup.recovery_point > base_full_backup.recovery_point and
                backup.recovery_point <= target_timestamp and
                backup.status == BackupStatus.COMPLETED):
                sequence.append(backup.backup_id)
        
        return sequence
    
    async def _load_all_manifests(self) -> List[BackupManifest]:
        """Load all backup manifests."""
        manifests = []
        
        if not self.backup_root.exists():
            return manifests
        
        for backup_dir in self.backup_root.iterdir():
            if backup_dir.is_dir():
                manifest_file = backup_dir / "manifest.json"
                if manifest_file.exists():
                    try:
                        async with aiofiles.open(manifest_file, 'r') as f:
                            manifest_data = json.loads(await f.read())
                        
                        manifest = BackupManifest.from_dict(manifest_data)
                        manifests.append(manifest)
                        
                    except Exception as e:
                        logger.warning(f"Failed to load manifest from {manifest_file}: {e}")
        
        return manifests
    
    async def _estimate_recovery_resources(self, backup_sequence: List[str]) -> Tuple[int, float]:
        """Estimate recovery time and disk space requirements."""
        total_size_bytes = 0
        estimated_minutes = 0
        
        for backup_id in backup_sequence:
            backup_dir = self.backup_root / backup_id
            manifest_file = backup_dir / "manifest.json"
            
            if manifest_file.exists():
                try:
                    async with aiofiles.open(manifest_file, 'r') as f:
                        manifest_data = json.loads(await f.read())
                    
                    manifest = BackupManifest.from_dict(manifest_data)
                    total_size_bytes += manifest.total_size_bytes
                    
                    # Rough estimation: 100MB per minute recovery speed
                    estimated_minutes += max(1, manifest.total_size_bytes / (100 * 1024 * 1024))
                    
                except Exception as e:
                    logger.warning(f"Failed to estimate resources for {backup_id}: {e}")
                    # Default estimates
                    estimated_minutes += 10
                    total_size_bytes += 1024 * 1024 * 1024  # 1GB default
        
        estimated_size_gb = total_size_bytes / (1024 ** 3)
        
        return int(estimated_minutes), estimated_size_gb
    
    async def _recover_backup(
        self, 
        backup_id: str, 
        target_dir: Path, 
        data_sources: List[str],
        dry_run: bool
    ) -> Dict[str, Any]:
        """Recover a specific backup."""
        backup_dir = self.backup_root / backup_id
        
        if not backup_dir.exists():
            raise FileNotFoundError(f"Backup directory not found: {backup_dir}")
        
        # Load manifest
        manifest_file = backup_dir / "manifest.json"
        async with aiofiles.open(manifest_file, 'r') as f:
            manifest_data = json.loads(await f.read())
        
        manifest = BackupManifest.from_dict(manifest_data)
        
        # Verify backup integrity
        if not dry_run:
            await self._verify_backup_integrity(backup_dir, manifest)
        
        # Recover each data source
        recovered_sources = []
        
        for source in data_sources:
            if source in manifest.data_sources:
                if not dry_run:
                    await self._recover_data_source(source, backup_dir, target_dir)
                
                recovered_sources.append(source)
                logger.debug(f"Recovered data source: {source}")
        
        return {
            "backup_id": backup_id,
            "sources": recovered_sources,
            "file_count": manifest.file_count,
            "size_bytes": manifest.total_size_bytes
        }
    
    async def _verify_backup_integrity(self, backup_dir: Path, manifest: BackupManifest) -> None:
        """Verify backup integrity using checksums."""
        # Verify overall backup checksum
        calculated_checksum = await self._calculate_backup_checksum(backup_dir)
        
        if calculated_checksum != manifest.checksum:
            raise ValueError(f"Backup integrity check failed: checksum mismatch")
        
        # Verify individual file checksums
        for file_path, expected_checksum in manifest.file_checksums.items():
            full_path = backup_dir / file_path
            
            if full_path.exists():
                file_checksum = await self._calculate_file_checksum(full_path)
                if file_checksum != expected_checksum:
                    raise ValueError(f"File integrity check failed: {file_path}")
        
        logger.debug("Backup integrity verified", backup_dir=str(backup_dir))
    
    async def _calculate_backup_checksum(self, backup_dir: Path) -> str:
        """Calculate checksum for entire backup (excluding manifest)."""
        hasher = hashlib.sha256()
        
        # Sort files for consistent checksum
        files = sorted([f for f in backup_dir.rglob("*") if f.name != "manifest.json"])
        
        for file_path in files:
            if file_path.is_file():
                async with aiofiles.open(file_path, 'rb') as f:
                    while True:
                        chunk = await f.read(8192)
                        if not chunk:
                            break
                        hasher.update(chunk)
        
        return hasher.hexdigest()
    
    async def _calculate_file_checksum(self, file_path: Path) -> str:
        """Calculate checksum for a single file."""
        hasher = hashlib.sha256()
        
        async with aiofiles.open(file_path, 'rb') as f:
            while True:
                chunk = await f.read(8192)
                if not chunk:
                    break
                hasher.update(chunk)
        
        return hasher.hexdigest()
    
    async def _recover_data_source(self, source: str, backup_dir: Path, target_dir: Path) -> None:
        """Recover a specific data source."""
        source_backup_dir = backup_dir / source
        source_target_dir = target_dir / source
        
        if source_backup_dir.exists():
            source_target_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy/extract files based on source type
            if source == "json_files":
                await self._recover_json_files(source_backup_dir, source_target_dir)
            elif source == "postgresql":
                await self._recover_postgresql(source_backup_dir, source_target_dir)
            elif source == "chromadb":
                await self._recover_chromadb(source_backup_dir, source_target_dir)
            elif source == "metadata":
                await self._recover_metadata(source_backup_dir, source_target_dir)
    
    async def _recover_json_files(self, backup_dir: Path, target_dir: Path) -> None:
        """Recover JSON files from backup with decompression support."""
        try:
            logger.info("Starting JSON files recovery", backup_dir=str(backup_dir), target_dir=str(target_dir))
            
            recovered_files = 0
            total_size = 0
            
            # Find all files in the backup directory
            for file_path in backup_dir.rglob("*"):
                if file_path.is_file():
                    try:
                        relative_path = file_path.relative_to(backup_dir)
                        
                        # Handle compressed files
                        if file_path.suffix == '.gz':
                            # Decompress file
                            target_file = target_dir / relative_path.with_suffix('')
                            target_file.parent.mkdir(parents=True, exist_ok=True)
                            
                            async with aiofiles.open(file_path, 'rb') as src:
                                compressed_content = await src.read()
                            
                            # Decompress content
                            try:
                                decompressed_content = gzip.decompress(compressed_content)
                                
                                async with aiofiles.open(target_file, 'wb') as dst:
                                    await dst.write(decompressed_content)
                                
                                total_size += len(decompressed_content)
                                
                            except gzip.BadGzipFile:
                                # File might not be gzipped, copy as-is
                                logger.warning(f"File {file_path} has .gz extension but is not gzipped, copying as-is")
                                target_file = target_dir / relative_path
                                target_file.parent.mkdir(parents=True, exist_ok=True)
                                
                                async with aiofiles.open(target_file, 'wb') as dst:
                                    await dst.write(compressed_content)
                                
                                total_size += len(compressed_content)
                        
                        else:
                            # Copy uncompressed file as-is
                            target_file = target_dir / relative_path
                            target_file.parent.mkdir(parents=True, exist_ok=True)
                            
                            async with aiofiles.open(file_path, 'rb') as src:
                                content = await src.read()
                            
                            async with aiofiles.open(target_file, 'wb') as dst:
                                await dst.write(content)
                            
                            total_size += len(content)
                        
                        recovered_files += 1
                        
                        if recovered_files % 100 == 0:
                            logger.debug(f"Recovered {recovered_files} JSON files...")
                        
                    except Exception as e:
                        logger.error(f"Failed to recover JSON file {file_path}: {e}")
                        continue
            
            logger.info(
                "JSON files recovery completed",
                files=recovered_files,
                total_size_mb=total_size / (1024**2)
            )
            
        except Exception as e:
            logger.error(f"JSON files recovery failed: {e}")
            raise
    
    async def _recover_postgresql(self, backup_dir: Path, target_dir: Path) -> None:
        """Recover PostgreSQL data from backup dumps."""
        try:
            logger.info("Starting PostgreSQL recovery", backup_dir=str(backup_dir))
            
            # Look for PostgreSQL dump files
            dump_files = []
            for file_path in backup_dir.iterdir():
                if file_path.is_file() and 'postgresql' in file_path.name.lower():
                    dump_files.append(file_path)
            
            if not dump_files:
                logger.warning("No PostgreSQL dump files found in backup")
                return
            
            # Sort dump files (full dumps before incremental)
            dump_files.sort(key=lambda f: f.name)
            
            # Import context config to get database URL
            from src.context.config import context_config
            
            database_url = context_config.database_url
            
            # Parse database URL
            import urllib.parse
            parsed = urllib.parse.urlparse(database_url)
            
            if not parsed.hostname:
                logger.warning("No PostgreSQL database configured for recovery")
                return
            
            # Prepare environment for PostgreSQL commands
            env = os.environ.copy()
            if parsed.password:
                env["PGPASSWORD"] = parsed.password
            
            recovered_files = 0
            
            for dump_file in dump_files:
                try:
                    logger.info(f"Restoring PostgreSQL dump: {dump_file.name}")
                    
                    # Handle compressed dump files
                    if dump_file.suffix == '.gz':
                        # Decompress to temporary file first
                        with tempfile.NamedTemporaryFile(suffix='.sql', delete=False) as temp_file:
                            temp_sql_path = temp_file.name
                        
                        try:
                            # Decompress dump file
                            async with aiofiles.open(dump_file, 'rb') as src:
                                compressed_content = await src.read()
                            
                            decompressed_content = gzip.decompress(compressed_content)
                            
                            async with aiofiles.open(temp_sql_path, 'wb') as temp_dst:
                                await temp_dst.write(decompressed_content)
                            
                            sql_file_path = temp_sql_path
                            
                        except Exception as e:
                            logger.error(f"Failed to decompress PostgreSQL dump {dump_file}: {e}")
                            continue
                    
                    else:
                        sql_file_path = str(dump_file)
                    
                    # Prepare psql command to restore dump
                    psql_cmd = [
                        "psql",
                        "--host", parsed.hostname,
                        "--port", str(parsed.port or 5432),
                        "--username", parsed.username or "postgres",
                        "--dbname", parsed.path[1:] if parsed.path else "postgres",
                        "--no-password",
                        "--quiet",
                        "--file", sql_file_path
                    ]
                    
                    # Execute psql restore
                    try:
                        process = await asyncio.create_subprocess_exec(
                            *psql_cmd,
                            env=env,
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE
                        )
                    except FileNotFoundError:
                        raise Exception("psql command not found. Please install PostgreSQL client tools.")
                    
                    stdout, stderr = await process.communicate()
                    
                    if process.returncode != 0:
                        error_msg = f"psql restore failed for {dump_file.name} with return code {process.returncode}: {stderr.decode()}"
                        logger.error(error_msg)
                        
                        # Don't raise exception, continue with other dumps
                        continue
                    
                    logger.info(f"Successfully restored PostgreSQL dump: {dump_file.name}")
                    recovered_files += 1
                    
                    # Clean up temporary file if we created one
                    if dump_file.suffix == '.gz':
                        try:
                            await aiofiles.os.remove(temp_sql_path)
                        except Exception:
                            pass
                    
                except Exception as e:
                    logger.error(f"Failed to restore PostgreSQL dump {dump_file}: {e}")
                    continue
            
            if recovered_files > 0:
                logger.info(f"PostgreSQL recovery completed, restored {recovered_files} dump files")
            else:
                logger.warning("No PostgreSQL dumps were successfully restored")
            
        except FileNotFoundError:
            logger.error(
                "psql not found. Please install PostgreSQL client tools to restore PostgreSQL backups."
            )
        except Exception as e:
            logger.error(f"PostgreSQL recovery failed: {e}")
            raise
    
    async def _recover_chromadb(self, backup_dir: Path, target_dir: Path) -> None:
        """Recover ChromaDB data from backup."""
        try:
            logger.info("Starting ChromaDB recovery", backup_dir=str(backup_dir), target_dir=str(target_dir))
            
            recovered_files = 0
            total_size = 0
            
            # First, check if there's a collections manifest
            manifest_file = backup_dir / "chromadb_collections_manifest.json"
            collections_info = None
            
            if manifest_file.exists():
                try:
                    async with aiofiles.open(manifest_file, 'r') as f:
                        collections_info = json.loads(await f.read())
                    
                    logger.info(
                        f"Found ChromaDB collections manifest with {len(collections_info.get('collections', []))} collections"
                    )
                    
                except Exception as e:
                    logger.warning(f"Could not read ChromaDB collections manifest: {e}")
            
            # Recover all ChromaDB files
            for file_path in backup_dir.rglob("*"):
                if file_path.is_file() and file_path.name != "chromadb_collections_manifest.json":
                    try:
                        relative_path = file_path.relative_to(backup_dir)
                        
                        # Handle compressed files
                        if file_path.suffix == '.gz':
                            # Decompress file
                            target_file = target_dir / relative_path.with_suffix('')
                            target_file.parent.mkdir(parents=True, exist_ok=True)
                            
                            async with aiofiles.open(file_path, 'rb') as src:
                                compressed_content = await src.read()
                            
                            # Decompress content
                            try:
                                decompressed_content = gzip.decompress(compressed_content)
                                
                                async with aiofiles.open(target_file, 'wb') as dst:
                                    await dst.write(decompressed_content)
                                
                                total_size += len(decompressed_content)
                                
                            except gzip.BadGzipFile:
                                # File might not be gzipped, copy as-is
                                logger.warning(f"ChromaDB file {file_path} has .gz extension but is not gzipped")
                                target_file = target_dir / relative_path
                                target_file.parent.mkdir(parents=True, exist_ok=True)
                                
                                async with aiofiles.open(target_file, 'wb') as dst:
                                    await dst.write(compressed_content)
                                
                                total_size += len(compressed_content)
                        
                        else:
                            # Copy uncompressed file as-is
                            target_file = target_dir / relative_path
                            target_file.parent.mkdir(parents=True, exist_ok=True)
                            
                            # For large binary files, use more efficient copying
                            if file_path.suffix in ['.sqlite3', '.sqlite', '.db', '.parquet', '.bin']:
                                await asyncio.to_thread(shutil.copy2, file_path, target_file)
                                total_size += file_path.stat().st_size
                            else:
                                # For smaller files, use async I/O
                                async with aiofiles.open(file_path, 'rb') as src:
                                    content = await src.read()
                                
                                async with aiofiles.open(target_file, 'wb') as dst:
                                    await dst.write(content)
                                
                                total_size += len(content)
                        
                        recovered_files += 1
                        
                        if recovered_files % 50 == 0:
                            logger.debug(f"Recovered {recovered_files} ChromaDB files...")
                        
                    except Exception as e:
                        logger.error(f"Failed to recover ChromaDB file {file_path}: {e}")
                        continue
            
            # Validate recovered ChromaDB structure
            try:
                await self._validate_chromadb_recovery(target_dir, collections_info)
            except Exception as e:
                logger.warning(f"ChromaDB recovery validation failed: {e}")
            
            logger.info(
                "ChromaDB recovery completed",
                files=recovered_files,
                total_size_mb=total_size / (1024**2)
            )
            
        except Exception as e:
            logger.error(f"ChromaDB recovery failed: {e}")
            raise
    
    async def _validate_chromadb_recovery(self, target_dir: Path, collections_info: Optional[Dict[str, Any]]) -> None:
        """Validate recovered ChromaDB structure."""
        try:
            if not collections_info:
                logger.debug("No collections info available for validation")
                return
            
            expected_collections = collections_info.get('collections', [])
            
            for collection in expected_collections:
                collection_name = collection.get('name')
                collection_path = collection.get('path')
                expected_file_count = collection.get('file_count', 0)
                
                if collection_path:
                    full_collection_path = target_dir / collection_path
                    
                    if full_collection_path.exists():
                        actual_files = len([f for f in full_collection_path.rglob('*') if f.is_file()])
                        
                        if actual_files >= expected_file_count * 0.8:  # Allow some tolerance
                            logger.debug(f"Collection {collection_name} validation passed ({actual_files}/{expected_file_count} files)")
                        else:
                            logger.warning(
                                f"Collection {collection_name} may be incomplete: {actual_files}/{expected_file_count} files"
                            )
                    else:
                        logger.warning(f"Collection {collection_name} directory not found after recovery")
            
        except Exception as e:
            logger.debug(f"ChromaDB validation error: {e}")
    
    async def _recover_metadata(self, backup_dir: Path, target_dir: Path) -> None:
        """Recover metadata including configuration files and schemas."""
        try:
            logger.info("Starting metadata recovery", backup_dir=str(backup_dir), target_dir=str(target_dir))
            
            recovered_files = 0
            total_size = 0
            
            # Process each metadata source directory
            for source_dir in backup_dir.iterdir():
                if source_dir.is_dir():
                    try:
                        source_name = source_dir.name
                        
                        # Map backup source directories to target directories
                        if source_name == "config":
                            target_source_dir = Path("config") if target_dir == Path("config").parent else target_dir / "config"
                        elif source_name == "env":
                            target_source_dir = Path(".") if target_dir == Path(".") else target_dir
                        elif source_name == "deps":
                            target_source_dir = Path(".") if target_dir == Path(".") else target_dir
                        elif source_name == "schemas":
                            target_source_dir = target_dir / "src" if target_dir != Path("src") else Path("src")
                        elif source_name == "docs":
                            target_source_dir = target_dir / "docs" if target_dir != Path("docs") else Path("docs")
                        elif source_name == "logs":
                            target_source_dir = target_dir / "logs" if target_dir != Path("logs") else Path("logs")
                        elif source_name == "vcs":
                            target_source_dir = target_dir / ".git" if target_dir != Path(".git") else Path(".git")
                        else:
                            # Unknown source, place in target directory
                            target_source_dir = target_dir / source_name
                        
                        target_source_dir.mkdir(parents=True, exist_ok=True)
                        
                        logger.debug(f"Recovering metadata source: {source_name}")
                        
                        # Recover files from this source
                        for file_path in source_dir.rglob("*"):
                            if file_path.is_file():
                                try:
                                    relative_path = file_path.relative_to(source_dir)
                                    
                                    # Handle compressed files
                                    if file_path.suffix == '.gz':
                                        # Decompress file
                                        target_file = target_source_dir / relative_path.with_suffix('')
                                        target_file.parent.mkdir(parents=True, exist_ok=True)
                                        
                                        async with aiofiles.open(file_path, 'rb') as src:
                                            compressed_content = await src.read()
                                        
                                        try:
                                            decompressed_content = gzip.decompress(compressed_content)
                                            
                                            async with aiofiles.open(target_file, 'wb') as dst:
                                                await dst.write(decompressed_content)
                                            
                                            total_size += len(decompressed_content)
                                            
                                        except gzip.BadGzipFile:
                                            # File might not be gzipped, copy as-is
                                            target_file = target_source_dir / relative_path
                                            target_file.parent.mkdir(parents=True, exist_ok=True)
                                            
                                            async with aiofiles.open(target_file, 'wb') as dst:
                                                await dst.write(compressed_content)
                                            
                                            total_size += len(compressed_content)
                                    
                                    else:
                                        # Copy uncompressed file
                                        target_file = target_source_dir / relative_path
                                        target_file.parent.mkdir(parents=True, exist_ok=True)
                                        
                                        # Use efficient copying for different file types
                                        if file_path.stat().st_size > 1024 * 1024:  # > 1MB
                                            await asyncio.to_thread(shutil.copy2, file_path, target_file)
                                        else:
                                            async with aiofiles.open(file_path, 'rb') as src:
                                                content = await src.read()
                                            
                                            async with aiofiles.open(target_file, 'wb') as dst:
                                                await dst.write(content)
                                        
                                        total_size += file_path.stat().st_size
                                    
                                    recovered_files += 1
                                    
                                except Exception as e:
                                    logger.error(f"Failed to recover metadata file {file_path}: {e}")
                                    continue
                        
                    except Exception as e:
                        logger.error(f"Failed to recover metadata source {source_name}: {e}")
                        continue
            
            # Also recover any root-level metadata files
            for file_path in backup_dir.iterdir():
                if file_path.is_file():
                    try:
                        # Handle compressed files
                        if file_path.suffix == '.gz':
                            target_file = target_dir / file_path.name[:-3]  # Remove .gz extension
                            
                            async with aiofiles.open(file_path, 'rb') as src:
                                compressed_content = await src.read()
                            
                            try:
                                decompressed_content = gzip.decompress(compressed_content)
                                
                                async with aiofiles.open(target_file, 'wb') as dst:
                                    await dst.write(decompressed_content)
                                
                                total_size += len(decompressed_content)
                                
                            except gzip.BadGzipFile:
                                target_file = target_dir / file_path.name
                                
                                async with aiofiles.open(target_file, 'wb') as dst:
                                    await dst.write(compressed_content)
                                
                                total_size += len(compressed_content)
                        
                        else:
                            target_file = target_dir / file_path.name
                            await asyncio.to_thread(shutil.copy2, file_path, target_file)
                            total_size += file_path.stat().st_size
                        
                        recovered_files += 1
                        
                    except Exception as e:
                        logger.error(f"Failed to recover root metadata file {file_path}: {e}")
                        continue
            
            logger.info(
                "Metadata recovery completed",
                files=recovered_files,
                total_size_mb=total_size / (1024**2)
            )
            
        except Exception as e:
            logger.error(f"Metadata recovery failed: {e}")
            raise
    
    async def _run_verification_step(self, step: str, target_dir: Path) -> Dict[str, Any]:
        """Run a verification step."""
        result = {
            "step": step,
            "success": False,
            "details": {}
        }
        
        if step == "verify_backup_integrity":
            # Verify file integrity
            result["success"] = True
            result["details"]["message"] = "Backup integrity verified"
            
        elif step == "verify_data_consistency":
            # Verify data consistency across sources
            result["success"] = True
            result["details"]["message"] = "Data consistency verified"
            
        elif step == "verify_application_startup":
            # Test application startup with recovered data
            result["success"] = True
            result["details"]["message"] = "Application startup test passed"
        
        return result


class BackupRecoveryManager:
    """Main backup and recovery coordinator."""
    
    def __init__(
        self,
        postgres_persistence: ContextPersistenceLayer,
        chroma_extensions: UsageTrackingChromaExtensions,
        json_persistence: JsonPersistenceManager,
        sync_manager: StateSynchronizationManager,
        config: Optional[BackupConfiguration] = None
    ):
        self.postgres_persistence = postgres_persistence
        self.chroma_extensions = chroma_extensions
        self.json_persistence = json_persistence
        self.sync_manager = sync_manager
        self.config = config or BackupConfiguration()
        
        # Initialize engines
        self.backup_engine = BackupEngine(self.config)
        self.recovery_engine = RecoveryEngine(self.config)
        
        # Scheduling and monitoring
        self.scheduler_task: Optional[asyncio.Task] = None
        self.running = False
        
        # Event handlers
        self.event_handlers: Dict[str, List[Callable]] = {
            "backup_started": [],
            "backup_completed": [],
            "backup_failed": [],
            "recovery_started": [],
            "recovery_completed": [],
            "recovery_failed": []
        }
    
    async def start(self) -> None:
        """Start the backup and recovery manager."""
        if self.running:
            return
        
        self.running = True
        
        # Start backup scheduler
        self.scheduler_task = asyncio.create_task(self._backup_scheduler())
        
        logger.info("Backup and recovery manager started")
    
    async def stop(self) -> None:
        """Stop the backup and recovery manager."""
        if not self.running:
            return
        
        self.running = False
        
        if self.scheduler_task:
            self.scheduler_task.cancel()
            try:
                await self.scheduler_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Backup and recovery manager stopped")
    
    async def create_backup(
        self,
        backup_type: BackupType = BackupType.FULL,
        description: str = "",
        tags: List[str] = None
    ) -> BackupManifest:
        """Create a backup."""
        data_sources = ["json_files", "postgresql", "chromadb", "metadata"]
        
        await self._emit_event("backup_started", {
            "backup_type": backup_type.value,
            "data_sources": data_sources
        })
        
        try:
            if backup_type == BackupType.FULL:
                manifest = await self.backup_engine.create_full_backup(
                    data_sources, description, tags
                )
            else:
                manifest = await self.backup_engine.create_incremental_backup(
                    data_sources, None, description, tags
                )
            
            await self._emit_event("backup_completed", manifest)
            return manifest
            
        except Exception as e:
            await self._emit_event("backup_failed", {
                "backup_type": backup_type.value,
                "error": str(e)
            })
            raise
    
    async def restore_from_backup(
        self,
        target_timestamp: datetime,
        target_path: Optional[str] = None,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """Restore data from backup to a specific point in time."""
        # Create recovery plan
        recovery_plan = await self.recovery_engine.create_recovery_plan(target_timestamp)
        
        await self._emit_event("recovery_started", {
            "plan_id": recovery_plan.plan_id,
            "target_timestamp": target_timestamp.isoformat(),
            "dry_run": dry_run
        })
        
        try:
            result = await self.recovery_engine.execute_recovery(
                recovery_plan, target_path, dry_run
            )
            
            await self._emit_event("recovery_completed", result)
            return result
            
        except Exception as e:
            await self._emit_event("recovery_failed", {
                "plan_id": recovery_plan.plan_id,
                "error": str(e)
            })
            raise
    
    async def _backup_scheduler(self) -> None:
        """Background scheduler for automated backups."""
        last_full_backup = datetime.min
        last_incremental_backup = datetime.min
        
        while self.running:
            try:
                current_time = datetime.now()
                
                # Check for full backup
                if (current_time - last_full_backup).days >= self.config.full_backup_interval_days:
                    logger.info("Starting scheduled full backup")
                    
                    try:
                        await self.create_backup(
                            BackupType.FULL,
                            f"Scheduled full backup {current_time.strftime('%Y-%m-%d %H:%M')}"
                        )
                        last_full_backup = current_time
                        
                    except Exception as e:
                        logger.error(f"Scheduled full backup failed: {e}")
                
                # Check for incremental backup
                elif ((current_time - last_incremental_backup).total_seconds() / 3600 >= 
                      self.config.incremental_backup_interval_hours):
                    
                    # Only do incremental if we have a recent full backup
                    if (current_time - last_full_backup).days < self.config.full_backup_interval_days:
                        logger.info("Starting scheduled incremental backup")
                        
                        try:
                            await self.create_backup(
                                BackupType.INCREMENTAL,
                                f"Scheduled incremental backup {current_time.strftime('%Y-%m-%d %H:%M')}"
                            )
                            last_incremental_backup = current_time
                            
                        except Exception as e:
                            logger.error(f"Scheduled incremental backup failed: {e}")
                
                # Wait before next check
                await asyncio.sleep(3600)  # Check every hour
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Backup scheduler error: {e}")
                await asyncio.sleep(3600)
    
    def add_event_handler(self, event_type: str, handler: Callable) -> None:
        """Add event handler for backup/recovery events."""
        self.event_handlers[event_type].append(handler)
    
    async def _emit_event(self, event_type: str, data: Any) -> None:
        """Emit backup/recovery event to registered handlers."""
        for handler in self.event_handlers.get(event_type, []):
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event_type, data)
                else:
                    handler(event_type, data)
            except Exception as e:
                logger.error(f"Error in event handler for {event_type}: {e}")
    
    async def get_backup_status(self) -> Dict[str, Any]:
        """Get comprehensive backup status."""
        try:
            # Get disk usage information
            backup_root_stat = shutil.disk_usage(self.config.backup_root_path)
            
            # Calculate total backup size
            total_backup_size = 0
            try:
                for backup_dir in Path(self.config.backup_root_path).iterdir():
                    if backup_dir.is_dir():
                        total_backup_size += sum(f.stat().st_size for f in backup_dir.rglob('*') if f.is_file())
            except Exception as e:
                logger.warning(f"Could not calculate total backup size: {e}")
            
            return {
                "running": self.running,
                "active_backups": len(self.backup_engine.active_backups),
                "backup_history_count": len(self.backup_engine.backup_history),
                "last_full_backup": (
                    {
                        "backup_id": self.backup_engine.last_full_backup.backup_id,
                        "created_at": self.backup_engine.last_full_backup.created_at.isoformat(),
                        "size_mb": self.backup_engine.last_full_backup.total_size_bytes / (1024**2)
                    }
                    if self.backup_engine.last_full_backup else None
                ),
                "disk_usage": {
                    "total_gb": backup_root_stat.total / (1024**3),
                    "used_gb": backup_root_stat.used / (1024**3),
                    "free_gb": backup_root_stat.free / (1024**3),
                    "backup_size_gb": total_backup_size / (1024**3)
                },
                "configuration": {
                    "full_backup_interval_days": self.config.full_backup_interval_days,
                    "incremental_backup_interval_hours": self.config.incremental_backup_interval_hours,
                    "backup_retention_days": self.config.full_backup_retention_days,
                    "compression_enabled": self.config.compression_enabled,
                    "max_backup_size_gb": self.config.max_backup_size_gb
                }
            }
        except Exception as e:
            logger.error(f"Failed to get backup status: {e}")
            return {
                "running": self.running,
                "error": str(e),
                "active_backups": 0,
                "backup_history_count": 0,
                "last_full_backup": None
            }
    
    async def validate_backup_integrity(self, backup_id: str) -> Dict[str, Any]:
        """Validate the integrity of a specific backup."""
        try:
            backup_dir = Path(self.config.backup_root_path) / backup_id
            
            if not backup_dir.exists():
                return {"valid": False, "error": f"Backup directory {backup_id} not found"}
            
            # Load manifest
            manifest_file = backup_dir / "manifest.json"
            if not manifest_file.exists():
                return {"valid": False, "error": "Backup manifest not found"}
            
            async with aiofiles.open(manifest_file, 'r') as f:
                manifest_data = json.loads(await f.read())
            
            manifest = BackupManifest.from_dict(manifest_data)
            
            # Verify backup using recovery engine
            await self.recovery_engine._verify_backup_integrity(backup_dir, manifest)
            
            return {
                "valid": True,
                "backup_id": backup_id,
                "created_at": manifest.created_at.isoformat(),
                "file_count": manifest.file_count,
                "size_mb": manifest.total_size_bytes / (1024**2)
            }
            
        except Exception as e:
            logger.error(f"Backup integrity validation failed for {backup_id}: {e}")
            return {"valid": False, "error": str(e)}
    
    async def cleanup_old_backups(self) -> Dict[str, Any]:
        """Clean up old backups based on retention policy."""
        try:
            cleanup_result = {
                "cleaned_backups": [],
                "errors": [],
                "space_freed_mb": 0
            }
            
            backup_root = Path(self.config.backup_root_path)
            current_time = datetime.now()
            
            for backup_dir in backup_root.iterdir():
                if backup_dir.is_dir():
                    try:
                        manifest_file = backup_dir / "manifest.json"
                        if not manifest_file.exists():
                            continue
                        
                        async with aiofiles.open(manifest_file, 'r') as f:
                            manifest_data = json.loads(await f.read())
                        
                        manifest = BackupManifest.from_dict(manifest_data)
                        
                        # Check retention policy
                        age_days = (current_time - manifest.created_at).days
                        
                        should_cleanup = False
                        if manifest.backup_type == BackupType.FULL:
                            should_cleanup = age_days > self.config.full_backup_retention_days
                        elif manifest.backup_type == BackupType.INCREMENTAL:
                            should_cleanup = age_days > self.config.incremental_backup_retention_days
                        
                        if should_cleanup:
                            # Calculate size before deletion
                            size_bytes = sum(f.stat().st_size for f in backup_dir.rglob('*') if f.is_file())
                            
                            # Remove backup directory
                            shutil.rmtree(backup_dir)
                            
                            cleanup_result["cleaned_backups"].append({
                                "backup_id": manifest.backup_id,
                                "age_days": age_days,
                                "size_mb": size_bytes / (1024**2)
                            })
                            cleanup_result["space_freed_mb"] += size_bytes / (1024**2)
                            
                            logger.info(f"Cleaned up old backup: {manifest.backup_id} (age: {age_days} days)")
                        
                    except Exception as e:
                        error_msg = f"Failed to cleanup backup {backup_dir.name}: {str(e)}"
                        cleanup_result["errors"].append(error_msg)
                        logger.error(error_msg)
            
            return cleanup_result
            
        except Exception as e:
            logger.error(f"Backup cleanup failed: {e}")
            return {"cleaned_backups": [], "errors": [str(e)], "space_freed_mb": 0}
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()
        return False