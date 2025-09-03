"""High-performance JSON Lines file persistence for usage tracking with offline access.

Uses JSON Lines (.jsonl) format where each line is a valid JSON object.
This allows for efficient appending without reading/rewriting entire files,
providing O(1) write operations and scalable performance for large datasets.
"""

import json
import asyncio
import aiofiles
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict, deque
import hashlib
import gzip
import pickle
from enum import Enum

from ..ai_providers.models import UsageRecord, ProviderType
from config.logging_config import get_logger

logger = get_logger(__name__)


class CompressionType(Enum):
    """File compression types."""
    NONE = "none"
    GZIP = "gzip"
    PICKLE = "pickle"


class PartitionStrategy(Enum):
    """Data partitioning strategies."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    BY_USER = "by_user"
    BY_PROVIDER = "by_provider"


@dataclass
class FileMetadata:
    """Metadata for JSON files."""
    file_path: str
    created_at: datetime
    last_modified: datetime
    record_count: int
    size_bytes: int
    compressed: bool
    compression_type: CompressionType
    partition_key: str
    schema_version: str
    checksum: str


@dataclass 
class PersistenceConfig:
    """Configuration for JSON persistence."""
    base_path: str = "./data/usage_tracking"
    partition_strategy: PartitionStrategy = PartitionStrategy.DAILY
    compression_type: CompressionType = CompressionType.GZIP
    max_file_size_mb: int = 10
    max_records_per_file: int = 10000
    enable_indexing: bool = True
    enable_backup: bool = True
    backup_retention_days: int = 30
    write_buffer_size: int = 1000
    flush_interval_seconds: int = 30
    concurrent_writers: int = 4
    enable_checksums: bool = True
    schema_version: str = "1.0.0"


class JsonPersistenceManager:
    """High-performance JSON file persistence with optimization features."""
    
    def __init__(self, config: Optional[PersistenceConfig] = None):
        self.config = config or PersistenceConfig()
        self.base_path = Path(self.config.base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Threading and concurrency
        self.executor = ThreadPoolExecutor(max_workers=self.config.concurrent_writers)
        self.write_lock = threading.RLock()
        self.metadata_lock = threading.RLock()
        
        # Write buffers for batching
        self.write_buffers: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.buffer_locks: Dict[str, threading.Lock] = defaultdict(threading.Lock)
        
        # File metadata cache
        self.file_metadata: Dict[str, FileMetadata] = {}
        self.metadata_file = self.base_path / "metadata.json"
        
        # Indexing for fast lookups
        self.indices: Dict[str, Dict[str, List[str]]] = {
            "user_id": defaultdict(list),
            "provider_type": defaultdict(list), 
            "date": defaultdict(list),
            "model": defaultdict(list)
        }
        self.index_file = self.base_path / "indices.json"
        
        # Background tasks
        self._flush_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Performance metrics
        self.metrics = {
            "writes_total": 0,
            "writes_successful": 0,
            "reads_total": 0,
            "reads_successful": 0,
            "files_created": 0,
            "bytes_written": 0,
            "bytes_read": 0,
            "compression_savings": 0.0,
            "avg_write_time": 0.0,
            "avg_read_time": 0.0
        }
        
        # Initialize
        self._load_metadata()
        self._load_indices()
    
    async def start(self) -> None:
        """Start the persistence manager with background tasks."""
        if self._running:
            return
        
        self._running = True
        
        # Start periodic flush task
        self._flush_task = asyncio.create_task(self._periodic_flush())
        
        logger.info("JSON persistence manager started", 
                   base_path=str(self.base_path),
                   partition_strategy=self.config.partition_strategy.value)
    
    async def stop(self) -> None:
        """Stop the persistence manager and flush remaining data."""
        if not self._running:
            return
        
        self._running = False
        
        # Cancel background tasks
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        
        # Flush all buffers
        await self._flush_all_buffers()
        
        # Shutdown thread pool
        self.executor.shutdown(wait=True)
        
        logger.info("JSON persistence manager stopped")
    
    def _generate_partition_key(self, record: UsageRecord) -> str:
        """Generate partition key based on strategy."""
        timestamp = record.timestamp
        
        if self.config.partition_strategy == PartitionStrategy.DAILY:
            return timestamp.strftime("%Y-%m-%d")
        elif self.config.partition_strategy == PartitionStrategy.WEEKLY:
            year, week, _ = timestamp.isocalendar()
            return f"{year}-W{week:02d}"
        elif self.config.partition_strategy == PartitionStrategy.MONTHLY:
            return timestamp.strftime("%Y-%m")
        elif self.config.partition_strategy == PartitionStrategy.BY_USER:
            user_id = record.metadata.get("user_id", "unknown")
            return f"user_{user_id}"
        elif self.config.partition_strategy == PartitionStrategy.BY_PROVIDER:
            return f"provider_{record.provider_type.value}"
        else:
            return "default"
    
    def _generate_file_path(self, partition_key: str, file_index: int = 0) -> Path:
        """Generate file path for a partition."""
        filename = f"{partition_key}_{file_index:03d}.jsonl"
        if self.config.compression_type != CompressionType.NONE:
            filename += f".{self.config.compression_type.value}"
        
        return self.base_path / "partitions" / filename
    
    def _calculate_checksum(self, data: bytes) -> str:
        """Calculate file checksum."""
        return hashlib.sha256(data).hexdigest()
    
    async def store_usage_record(self, record: UsageRecord) -> None:
        """Store a single usage record (buffered write)."""
        try:
            partition_key = self._generate_partition_key(record)
            
            # Convert record to dict
            record_dict = {
                "request_id": record.request_id,
                "session_id": record.session_id,
                "provider_type": record.provider_type.value,
                "model": record.model,
                "input_tokens": record.input_tokens,
                "output_tokens": record.output_tokens,
                "cost": record.cost,
                "latency_ms": record.latency_ms,
                "success": record.success,
                "error_message": record.error_message,
                "timestamp": record.timestamp.isoformat(),
                "metadata": record.metadata
            }
            
            # Add to write buffer
            with self.buffer_locks[partition_key]:
                self.write_buffers[partition_key].append(record_dict)
            
            # Flush if buffer is full
            if len(self.write_buffers[partition_key]) >= self.config.write_buffer_size:
                await self._flush_buffer(partition_key)
            
            self.metrics["writes_total"] += 1
            
        except Exception as e:
            logger.error("Failed to store usage record", record_id=record.request_id, error=str(e))
            raise
    
    async def store_usage_records_batch(self, records: List[UsageRecord]) -> None:
        """Store multiple usage records efficiently."""
        try:
            # Group records by partition
            partitioned_records: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            
            for record in records:
                partition_key = self._generate_partition_key(record)
                record_dict = {
                    "request_id": record.request_id,
                    "session_id": record.session_id,
                    "provider_type": record.provider_type.value,
                    "model": record.model,
                    "input_tokens": record.input_tokens,
                    "output_tokens": record.output_tokens,
                    "cost": record.cost,
                    "latency_ms": record.latency_ms,
                    "success": record.success,
                    "error_message": record.error_message,
                    "timestamp": record.timestamp.isoformat(),
                    "metadata": record.metadata
                }
                partitioned_records[partition_key].append(record_dict)
            
            # Store each partition concurrently
            tasks = []
            for partition_key, partition_records in partitioned_records.items():
                task = asyncio.create_task(self._store_partition_records(partition_key, partition_records))
                tasks.append(task)
            
            await asyncio.gather(*tasks)
            
            self.metrics["writes_total"] += len(records)
            
        except Exception as e:
            logger.error("Failed to store usage records batch", batch_size=len(records), error=str(e))
            raise
    
    async def _store_partition_records(self, partition_key: str, records: List[Dict[str, Any]]) -> None:
        """Store records for a specific partition."""
        with self.buffer_locks[partition_key]:
            self.write_buffers[partition_key].extend(records)
        
        # Flush if buffer is large
        if len(self.write_buffers[partition_key]) >= self.config.write_buffer_size:
            await self._flush_buffer(partition_key)
    
    async def _flush_buffer(self, partition_key: str) -> None:
        """Flush write buffer for a partition to disk."""
        if not self.write_buffers[partition_key]:
            return
        
        try:
            start_time = asyncio.get_event_loop().time()
            
            # Get records from buffer
            with self.buffer_locks[partition_key]:
                records_to_write = self.write_buffers[partition_key].copy()
                self.write_buffers[partition_key].clear()
            
            if not records_to_write:
                return
            
            # Find appropriate file
            file_path = await self._get_writable_file(partition_key, len(records_to_write))
            
            # Prepare data in JSON Lines format (one record per line)
            lines = []
            for record in records_to_write:
                record_with_meta = {
                    "schema_version": self.config.schema_version,
                    "partition_key": partition_key,
                    "timestamp": datetime.now().isoformat(),
                    **record
                }
                lines.append(json.dumps(record_with_meta, separators=(',', ':')))
            
            # Join lines and encode
            json_data = '\n'.join(lines) + '\n'
            json_bytes = json_data.encode('utf-8')
            
            if self.config.compression_type == CompressionType.GZIP:
                compressed_bytes = gzip.compress(json_bytes)
                final_data = compressed_bytes
                compression_savings = 1.0 - (len(compressed_bytes) / len(json_bytes))
            elif self.config.compression_type == CompressionType.PICKLE:
                pickled_data = pickle.dumps(records_to_write)
                final_data = pickled_data
                compression_savings = 1.0 - (len(pickled_data) / len(json_bytes))
            else:
                final_data = json_bytes
                compression_savings = 0.0
            
            # Write to file
            await self._write_file_data(file_path, final_data, compression_savings)
            
            # Update indices
            await self._update_indices(partition_key, records_to_write, str(file_path))
            
            # Update metrics
            write_time = asyncio.get_event_loop().time() - start_time
            self.metrics["writes_successful"] += len(records_to_write)
            self.metrics["bytes_written"] += len(final_data)
            self.metrics["compression_savings"] += compression_savings
            self.metrics["avg_write_time"] = (self.metrics["avg_write_time"] * 0.9) + (write_time * 0.1)
            
            logger.debug("Buffer flushed", 
                        partition_key=partition_key,
                        records=len(records_to_write),
                        file_path=str(file_path),
                        write_time=write_time)
            
        except Exception as e:
            logger.error("Failed to flush buffer", partition_key=partition_key, error=str(e))
            
            # Put records back in buffer if write failed
            with self.buffer_locks[partition_key]:
                self.write_buffers[partition_key] = records_to_write + self.write_buffers[partition_key]
            raise
    
    async def _get_writable_file(self, partition_key: str, additional_records: int) -> Path:
        """Get a writable file for the partition, creating new if needed."""
        # Find existing files for this partition
        existing_files = [
            meta for meta in self.file_metadata.values() 
            if meta.partition_key == partition_key
        ]
        
        # Check if any existing file can accommodate new records
        for file_meta in existing_files:
            if (file_meta.record_count + additional_records <= self.config.max_records_per_file and
                file_meta.size_bytes < self.config.max_file_size_mb * 1024 * 1024):
                return Path(file_meta.file_path)
        
        # Create new file
        file_index = len(existing_files)
        new_file_path = self._generate_file_path(partition_key, file_index)
        new_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        return new_file_path
    
    async def _write_file_data(self, file_path: Path, data: bytes, compression_savings: float) -> None:
        """Write data to file and update metadata."""
        # Use append mode for scalable JSON Lines format
        # This prevents reading and rewriting the entire file
        file_mode = 'ab' if file_path.exists() else 'wb'
        async with aiofiles.open(file_path, file_mode) as f:
            await f.write(data)
        
        # Update metadata
        file_stat = file_path.stat()
        checksum = self._calculate_checksum(data) if self.config.enable_checksums else ""
        
        with self.metadata_lock:
            if str(file_path) in self.file_metadata:
                # Update existing metadata
                meta = self.file_metadata[str(file_path)]
                meta.last_modified = datetime.now()
                meta.size_bytes = file_stat.st_size
                meta.checksum = checksum
            else:
                # Create new metadata
                partition_key = file_path.stem.rsplit('_', 1)[0]
                meta = FileMetadata(
                    file_path=str(file_path),
                    created_at=datetime.now(),
                    last_modified=datetime.now(),
                    record_count=0,  # Will be updated by caller
                    size_bytes=file_stat.st_size,
                    compressed=self.config.compression_type != CompressionType.NONE,
                    compression_type=self.config.compression_type,
                    partition_key=partition_key,
                    schema_version=self.config.schema_version,
                    checksum=checksum
                )
                self.file_metadata[str(file_path)] = meta
                self.metrics["files_created"] += 1
        
        # Save metadata periodically
        await self._save_metadata()
    
    async def _update_indices(self, partition_key: str, records: List[Dict[str, Any]], file_path: str) -> None:
        """Update search indices for fast lookups."""
        if not self.config.enable_indexing:
            return
        
        for record in records:
            # Index by user_id
            user_id = record.get("metadata", {}).get("user_id", "unknown")
            if file_path not in self.indices["user_id"][user_id]:
                self.indices["user_id"][user_id].append(file_path)
            
            # Index by provider_type
            provider_type = record.get("provider_type", "unknown")
            if file_path not in self.indices["provider_type"][provider_type]:
                self.indices["provider_type"][provider_type].append(file_path)
            
            # Index by date
            timestamp = record.get("timestamp", "")
            if timestamp:
                date = timestamp[:10]  # YYYY-MM-DD
                if file_path not in self.indices["date"][date]:
                    self.indices["date"][date].append(file_path)
            
            # Index by model
            model = record.get("model", "unknown")
            if file_path not in self.indices["model"][model]:
                self.indices["model"][model].append(file_path)
        
        # Save indices periodically
        await self._save_indices()
    
    async def query_usage_records(
        self,
        user_id: Optional[str] = None,
        provider_type: Optional[str] = None,
        date_range: Optional[Tuple[datetime, datetime]] = None,
        model: Optional[str] = None,
        limit: int = 1000,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Query usage records with filtering."""
        try:
            start_time = asyncio.get_event_loop().time()
            
            # Find relevant files using indices
            relevant_files = await self._find_relevant_files(user_id, provider_type, date_range, model)
            
            if not relevant_files:
                return []
            
            # Read and filter records from files
            all_records = []
            tasks = []
            
            for file_path in relevant_files:
                task = asyncio.create_task(
                    self._read_and_filter_file(file_path, user_id, provider_type, date_range, model)
                )
                tasks.append(task)
            
            file_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in file_results:
                if isinstance(result, Exception):
                    logger.error("Failed to read file during query", error=str(result))
                    continue
                all_records.extend(result)
            
            # Sort by timestamp (newest first)
            all_records.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
            
            # Apply pagination
            paginated_records = all_records[offset:offset + limit]
            
            # Update metrics
            read_time = asyncio.get_event_loop().time() - start_time
            self.metrics["reads_total"] += 1
            self.metrics["reads_successful"] += 1
            self.metrics["avg_read_time"] = (self.metrics["avg_read_time"] * 0.9) + (read_time * 0.1)
            
            logger.debug("Query completed",
                        files_searched=len(relevant_files),
                        records_found=len(all_records),
                        returned=len(paginated_records),
                        query_time=read_time)
            
            return paginated_records
            
        except Exception as e:
            logger.error("Failed to query usage records", error=str(e))
            raise
    
    async def _find_relevant_files(
        self,
        user_id: Optional[str],
        provider_type: Optional[str],
        date_range: Optional[Tuple[datetime, datetime]],
        model: Optional[str]
    ) -> List[str]:
        """Find files relevant to the query using indices."""
        if not self.config.enable_indexing:
            # Return all files if indexing is disabled
            return list(self.file_metadata.keys())
        
        candidate_files = set()
        
        # Filter by user_id
        if user_id and user_id in self.indices["user_id"]:
            candidate_files.update(self.indices["user_id"][user_id])
        elif user_id:
            return []  # No files for this user
        
        # Filter by provider_type
        if provider_type:
            if provider_type in self.indices["provider_type"]:
                if candidate_files:
                    candidate_files &= set(self.indices["provider_type"][provider_type])
                else:
                    candidate_files = set(self.indices["provider_type"][provider_type])
            else:
                return []  # No files for this provider
        
        # Filter by date range
        if date_range:
            start_date, end_date = date_range
            date_files = set()
            
            current_date = start_date.date()
            while current_date <= end_date.date():
                date_str = current_date.isoformat()
                if date_str in self.indices["date"]:
                    date_files.update(self.indices["date"][date_str])
                current_date += timedelta(days=1)
            
            if candidate_files:
                candidate_files &= date_files
            else:
                candidate_files = date_files
        
        # Filter by model
        if model:
            if model in self.indices["model"]:
                if candidate_files:
                    candidate_files &= set(self.indices["model"][model])
                else:
                    candidate_files = set(self.indices["model"][model])
            else:
                return []  # No files for this model
        
        # If no specific filters, return all files
        if not candidate_files and not any([user_id, provider_type, date_range, model]):
            candidate_files = set(self.file_metadata.keys())
        
        return list(candidate_files)
    
    async def _read_and_filter_file(
        self,
        file_path: str,
        user_id: Optional[str],
        provider_type: Optional[str],
        date_range: Optional[Tuple[datetime, datetime]],
        model: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Read and filter records from a single file."""
        try:
            path = Path(file_path)
            if not path.exists():
                return []
            
            # Read file data
            async with aiofiles.open(path, 'rb') as f:
                file_data = await f.read()
            
            # Decompress if needed
            file_meta = self.file_metadata.get(file_path)
            if file_meta and file_meta.compressed:
                if file_meta.compression_type == CompressionType.GZIP:
                    decompressed_data = gzip.decompress(file_data)
                    text_data = decompressed_data.decode('utf-8')
                elif file_meta.compression_type == CompressionType.PICKLE:
                    data = pickle.loads(file_data)
                    # Handle legacy format
                    records = data.get("records", [])
                else:
                    text_data = file_data.decode('utf-8')
            else:
                text_data = file_data.decode('utf-8')
            
            # Verify checksum if enabled
            if (self.config.enable_checksums and file_meta and 
                file_meta.checksum and file_meta.checksum != self._calculate_checksum(file_data)):
                logger.warning("Checksum mismatch detected", file_path=file_path)
            
            # Parse JSON Lines format (or legacy format)
            records = []
            if 'text_data' in locals():
                for line in text_data.strip().split('\n'):
                    if line:
                        try:
                            record = json.loads(line)
                            # Remove metadata fields to get clean record
                            record.pop('schema_version', None)
                            record.pop('partition_key', None)
                            records.append(record)
                        except json.JSONDecodeError:
                            # Try legacy format
                            data = json.loads(text_data)
                            records = data.get("records", [])
                            break
            
            # Apply filters
            filtered_records = []
            for record in records:
                if self._record_matches_filters(record, user_id, provider_type, date_range, model):
                    filtered_records.append(record)
            
            self.metrics["bytes_read"] += len(file_data)
            return filtered_records
            
        except Exception as e:
            logger.error("Failed to read file", file_path=file_path, error=str(e))
            return []
    
    def _record_matches_filters(
        self,
        record: Dict[str, Any],
        user_id: Optional[str],
        provider_type: Optional[str],
        date_range: Optional[Tuple[datetime, datetime]],
        model: Optional[str]
    ) -> bool:
        """Check if a record matches the query filters."""
        # Check user_id
        if user_id:
            record_user_id = record.get("metadata", {}).get("user_id", "unknown")
            if record_user_id != user_id:
                return False
        
        # Check provider_type
        if provider_type and record.get("provider_type") != provider_type:
            return False
        
        # Check date range
        if date_range:
            start_date, end_date = date_range
            record_timestamp = record.get("timestamp", "")
            if record_timestamp:
                try:
                    record_date = datetime.fromisoformat(record_timestamp.replace('Z', '+00:00'))
                    if not (start_date <= record_date <= end_date):
                        return False
                except ValueError:
                    return False
        
        # Check model
        if model and record.get("model") != model:
            return False
        
        return True
    
    async def _periodic_flush(self) -> None:
        """Periodically flush all buffers."""
        while self._running:
            try:
                await asyncio.sleep(self.config.flush_interval_seconds)
                await self._flush_all_buffers()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in periodic flush", error=str(e))
    
    async def _flush_all_buffers(self) -> None:
        """Flush all write buffers to disk."""
        if not self.write_buffers:
            return
        
        # Get all partition keys with data
        partition_keys = [k for k, v in self.write_buffers.items() if v]
        
        if not partition_keys:
            return
        
        # Flush all partitions concurrently
        tasks = []
        for partition_key in partition_keys:
            task = asyncio.create_task(self._flush_buffer(partition_key))
            tasks.append(task)
        
        await asyncio.gather(*tasks, return_exceptions=True)
        
        logger.debug("All buffers flushed", partitions=len(partition_keys))
    
    def _load_metadata(self) -> None:
        """Load file metadata from disk."""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    metadata_data = json.load(f)
                
                for file_path, meta_dict in metadata_data.items():
                    self.file_metadata[file_path] = FileMetadata(
                        file_path=meta_dict["file_path"],
                        created_at=datetime.fromisoformat(meta_dict["created_at"]),
                        last_modified=datetime.fromisoformat(meta_dict["last_modified"]),
                        record_count=meta_dict["record_count"],
                        size_bytes=meta_dict["size_bytes"],
                        compressed=meta_dict["compressed"],
                        compression_type=CompressionType(meta_dict["compression_type"]),
                        partition_key=meta_dict["partition_key"],
                        schema_version=meta_dict["schema_version"],
                        checksum=meta_dict.get("checksum", "")
                    )
                
                logger.debug("Metadata loaded", file_count=len(self.file_metadata))
        
        except Exception as e:
            logger.warning("Failed to load metadata", error=str(e))
    
    async def _save_metadata(self) -> None:
        """Save file metadata to disk."""
        try:
            metadata_data = {}
            with self.metadata_lock:
                for file_path, meta in self.file_metadata.items():
                    metadata_data[file_path] = {
                        "file_path": meta.file_path,
                        "created_at": meta.created_at.isoformat(),
                        "last_modified": meta.last_modified.isoformat(),
                        "record_count": meta.record_count,
                        "size_bytes": meta.size_bytes,
                        "compressed": meta.compressed,
                        "compression_type": meta.compression_type.value,
                        "partition_key": meta.partition_key,
                        "schema_version": meta.schema_version,
                        "checksum": meta.checksum
                    }
            
            # Write to temporary file first, then rename (atomic operation)
            temp_file = self.metadata_file.with_suffix('.tmp')
            async with aiofiles.open(temp_file, 'w') as f:
                await f.write(json.dumps(metadata_data, indent=2))
            
            temp_file.rename(self.metadata_file)
            
        except Exception as e:
            logger.error("Failed to save metadata", error=str(e))
    
    def _load_indices(self) -> None:
        """Load search indices from disk."""
        try:
            if self.index_file.exists():
                with open(self.index_file, 'r') as f:
                    index_data = json.load(f)
                
                for index_name, index_values in index_data.items():
                    self.indices[index_name] = defaultdict(list, index_values)
                
                logger.debug("Indices loaded", index_count=len(self.indices))
        
        except Exception as e:
            logger.warning("Failed to load indices", error=str(e))
    
    async def _save_indices(self) -> None:
        """Save search indices to disk."""
        try:
            index_data = {}
            for index_name, index_values in self.indices.items():
                index_data[index_name] = dict(index_values)
            
            # Write atomically
            temp_file = self.index_file.with_suffix('.tmp')
            async with aiofiles.open(temp_file, 'w') as f:
                await f.write(json.dumps(index_data, indent=2))
            
            temp_file.rename(self.index_file)
            
        except Exception as e:
            logger.error("Failed to save indices", error=str(e))
    
    async def get_storage_statistics(self) -> Dict[str, Any]:
        """Get comprehensive storage statistics."""
        stats = {
            "total_files": len(self.file_metadata),
            "total_records": sum(meta.record_count for meta in self.file_metadata.values()),
            "total_size_bytes": sum(meta.size_bytes for meta in self.file_metadata.values()),
            "compression_enabled": self.config.compression_type != CompressionType.NONE,
            "partitioning_strategy": self.config.partition_strategy.value,
            "partitions": {},
            "performance_metrics": dict(self.metrics),
            "buffer_status": {},
            "index_statistics": {}
        }
        
        # Partition statistics
        partition_stats = defaultdict(lambda: {"file_count": 0, "record_count": 0, "size_bytes": 0})
        for meta in self.file_metadata.values():
            partition_stats[meta.partition_key]["file_count"] += 1
            partition_stats[meta.partition_key]["record_count"] += meta.record_count
            partition_stats[meta.partition_key]["size_bytes"] += meta.size_bytes
        
        stats["partitions"] = dict(partition_stats)
        
        # Buffer status
        for partition_key, buffer in self.write_buffers.items():
            stats["buffer_status"][partition_key] = {
                "pending_records": len(buffer),
                "estimated_size": len(buffer) * 500  # Rough estimate
            }
        
        # Index statistics
        for index_name, index_values in self.indices.items():
            stats["index_statistics"][index_name] = {
                "unique_values": len(index_values),
                "total_references": sum(len(files) for files in index_values.values())
            }
        
        return stats
    
    async def cleanup_old_files(self, retention_days: int = 365) -> Dict[str, Any]:
        """Clean up old files based on retention policy."""
        cleanup_stats = {
            "files_deleted": 0,
            "bytes_freed": 0,
            "partitions_cleaned": set(),
            "errors": []
        }
        
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        
        files_to_delete = []
        for file_path, meta in list(self.file_metadata.items()):
            if meta.created_at < cutoff_date:
                files_to_delete.append((file_path, meta))
        
        for file_path, meta in files_to_delete:
            try:
                # Delete file
                path = Path(file_path)
                if path.exists():
                    path.unlink()
                
                # Remove from metadata
                with self.metadata_lock:
                    del self.file_metadata[file_path]
                
                # Update statistics
                cleanup_stats["files_deleted"] += 1
                cleanup_stats["bytes_freed"] += meta.size_bytes
                cleanup_stats["partitions_cleaned"].add(meta.partition_key)
                
                logger.debug("Cleaned up old file", file_path=file_path, age_days=(datetime.now() - meta.created_at).days)
                
            except Exception as e:
                error_msg = f"Failed to delete {file_path}: {str(e)}"
                cleanup_stats["errors"].append(error_msg)
                logger.error("Failed to cleanup file", file_path=file_path, error=str(e))
        
        # Rebuild indices after cleanup
        if files_to_delete:
            await self._rebuild_indices()
            await self._save_metadata()
            await self._save_indices()
        
        cleanup_stats["partitions_cleaned"] = list(cleanup_stats["partitions_cleaned"])
        
        logger.info("File cleanup completed", **cleanup_stats)
        return cleanup_stats
    
    async def _rebuild_indices(self) -> None:
        """Rebuild search indices from existing files."""
        # Clear existing indices
        self.indices = {
            "user_id": defaultdict(list),
            "provider_type": defaultdict(list),
            "date": defaultdict(list),
            "model": defaultdict(list)
        }
        
        # Rebuild from all existing files
        for file_path in self.file_metadata.keys():
            try:
                records = await self._read_all_records_from_file(file_path)
                await self._update_indices("", records, file_path)  # Empty partition key since we're rebuilding
            except Exception as e:
                logger.error("Failed to rebuild index for file", file_path=file_path, error=str(e))
        
        logger.info("Indices rebuilt", file_count=len(self.file_metadata))
    
    async def _read_all_records_from_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Read all records from a file without filtering."""
        return await self._read_and_filter_file(file_path, None, None, None, None)
    
    async def create_backup(self, backup_path: Optional[str] = None) -> Dict[str, Any]:
        """Create a backup of all usage data."""
        if not backup_path:
            backup_path = str(self.base_path.parent / f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        backup_path = Path(backup_path)
        backup_path.mkdir(parents=True, exist_ok=True)
        
        backup_stats = {
            "backup_path": str(backup_path),
            "files_copied": 0,
            "total_size_bytes": 0,
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "errors": []
        }
        
        try:
            # Flush all buffers first
            await self._flush_all_buffers()
            
            # Copy all data files
            for file_path, meta in self.file_metadata.items():
                try:
                    source_path = Path(file_path)
                    if source_path.exists():
                        # Preserve directory structure
                        relative_path = source_path.relative_to(self.base_path)
                        dest_path = backup_path / relative_path
                        dest_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        # Copy file
                        import shutil
                        await asyncio.get_event_loop().run_in_executor(
                            None, shutil.copy2, str(source_path), str(dest_path)
                        )
                        
                        backup_stats["files_copied"] += 1
                        backup_stats["total_size_bytes"] += meta.size_bytes
                        
                except Exception as e:
                    error_msg = f"Failed to backup {file_path}: {str(e)}"
                    backup_stats["errors"].append(error_msg)
                    logger.error("Failed to backup file", file_path=file_path, error=str(e))
            
            # Copy metadata and indices
            if self.metadata_file.exists():
                import shutil
                await asyncio.get_event_loop().run_in_executor(
                    None, shutil.copy2, str(self.metadata_file), str(backup_path / "metadata.json")
                )
            
            if self.index_file.exists():
                import shutil
                await asyncio.get_event_loop().run_in_executor(
                    None, shutil.copy2, str(self.index_file), str(backup_path / "indices.json")
                )
            
            backup_stats["end_time"] = datetime.now().isoformat()
            
            logger.info("Backup created successfully", **backup_stats)
            return backup_stats
            
        except Exception as e:
            backup_stats["end_time"] = datetime.now().isoformat()
            backup_stats["errors"].append(f"Backup failed: {str(e)}")
            logger.error("Failed to create backup", error=str(e))
            raise
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()
        return False